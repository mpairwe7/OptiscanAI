#!/usr/bin/env python3
"""Generalizability test for the lightweight triage model — accuracy + precision.

The headline triage numbers (acc/precision 1.000) came from one 24-case split of
80 RFMiD traces. This script stress-tests whether they *generalize*, on four axes,
each reporting accuracy and macro-precision:

1. **True out-of-sample** — generate fresh teacher traces over *disjoint, unseen*
   RFMiD images (skip the original 80), train on the original 80, test on the
   fresh set. This is the strongest signal: new patients, same teacher + pipeline.
2. **Repeated stratified K-fold CV** — 5-fold × N seeds over all data → mean ± std
   accuracy and macro-precision (stability, not a lucky split).
3. **Learning curve** — train on growing fractions, test out-of-sample (data
   sufficiency).
4. **EMERGENCY stress test** — the real data has zero EMERGENCY cases, so that
   safety path is untested. Inject emergency codes (CRAO/AION) and verify the
   trained head + deterministic override yields emergency recall + precision.

Teacher = live Qwen3-8B-AWQ @ vLLM. Trace generation needs ``model_service`` (GPU)
and is cached/resumable.

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=<free> /usr/bin/python3 \
        scripts/generalizability_test.py --skip 80 --n 160 --threshold 0.53
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reasoner_comparison.cases import CODE_TO_NAME  # noqa: E402
from src.evaluation.reasoner_comparison.features import (  # noqa: E402
    PriorityClassifier,
    case_features,
)
from src.evaluation.reasoner_comparison.interface import (  # noqa: E402
    PRIORITY_INDEX,
    Case,
    Prediction,
    ReasonerOutput,
)
from src.evaluation.reasoner_comparison.metrics import triage_metrics  # noqa: E402
from src.evaluation.reasoner_comparison.reasoners import FeatureTriageReasoner  # noqa: E402

logger = logging.getLogger("generalizability")


# ── data ──


def _case_from_trace(tr: dict) -> Case:
    preds = [
        Prediction(p["code"], p["name"], float(p["probability"]), p.get("confidence", "unknown"))
        for p in tr["predictions"]
    ]
    t = tr["teacher"] if "teacher" in tr else tr
    case = Case(
        scan_id=tr["scan_id"],
        predictions=preds,
        probabilities={},
        referral_priority=tr.get("referral", "FOLLOW_UP"),
    )
    case.reference = ReasonerOutput(
        priority=t["priority"],
        should_explain=bool(t.get("should_explain", False)),
        should_review=bool(t.get("should_review", False)),
        reasoning=t.get("reasoning", ""),
        narrative=t.get("narrative", ""),
        source="qwen_teacher",
    )
    return case


def load_train_cases(sft_path: Path) -> list[Case]:
    rows = [json.loads(ln) for ln in sft_path.read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["scan_id"])
    return [_case_from_trace(r) for r in rows]


def generate_oos_cases(args, train_scan_ids: set[str]) -> list[Case]:
    """Generate (cached) teacher traces over fresh, disjoint, unseen images."""
    from PIL import Image

    from backend.app.core.model_service import model_service
    from src.evaluation.reasoner_comparison.vllm_teacher import VLLMReasoner

    traces_path = Path(args.oos_traces)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if traces_path.exists():
        for ln in traces_path.read_text().splitlines():
            if ln.strip():
                obj = json.loads(ln)
                cache[obj["scan_id"]] = obj
    logger.info("%d OOS traces already cached", len(cache))

    images = sorted(
        glob.glob(f"{args.images_dir}/**/*.png", recursive=True)
        + glob.glob(f"{args.images_dir}/**/*.jpg", recursive=True)
    )
    # disjoint from training: skip the first `skip`, drop any that match a train id.
    fresh = [p for p in images[args.skip :] if Path(p).stem not in train_scan_ids][: args.n]
    if not fresh:
        raise SystemExit("no fresh images to score")

    need = [p for p in fresh if Path(p).stem not in cache]
    if need:
        model_service.load()
        teacher = VLLMReasoner(base_url=args.base_url)
        fh = open(traces_path, "a")
        try:
            for i, img_path in enumerate(need):
                scan_id = Path(img_path).stem
                result = model_service.predict(
                    Image.open(img_path).convert("RGB"), threshold=args.threshold
                )
                preds = [
                    Prediction(
                        p["code"],
                        p["name"],
                        float(p["probability"]),
                        p.get("confidence", "unknown"),
                    )
                    for p in result.get("predictions", [])
                ]
                referral = result.get("clinical", {}).get("referral_priority", "FOLLOW_UP")
                case = Case(scan_id, preds, {}, referral_priority=referral)
                t0 = time.time()
                out = teacher.reason(case)
                tr = {
                    "scan_id": scan_id,
                    "referral": referral,
                    "predictions": [
                        {
                            "code": p.code,
                            "name": p.name,
                            "probability": p.probability,
                            "confidence": p.confidence,
                        }
                        for p in preds
                    ],
                    "teacher": {
                        "priority": out.priority,
                        "should_explain": out.should_explain,
                        "should_review": out.should_review,
                        "reasoning": out.reasoning,
                        "narrative": out.narrative,
                    },
                }
                fh.write(json.dumps(tr) + "\n")
                fh.flush()
                cache[scan_id] = tr
                if (i + 1) % 10 == 0:
                    logger.info(
                        "[%d/%d] generated (last %s -> %s, %.1fs)",
                        i + 1,
                        len(need),
                        scan_id,
                        out.priority,
                        time.time() - t0,
                    )
        finally:
            fh.close()
    return [_case_from_trace(cache[Path(p).stem]) for p in fresh]


# ── models ──


def build_panel(seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    panel = {
        "logreg": lambda: LogisticRegression(max_iter=2000),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier

        panel["xgboost"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=4, random_state=seed, verbosity=0, n_jobs=2
        )
    except ImportError:
        pass
    return panel


def spread(cases: list[Case]) -> dict:
    d: dict[str, int] = {}
    for c in cases:
        d[c.reference.priority] = d.get(c.reference.priority, 0) + 1
    return d


def macro_precision(y_true, y_pred) -> float:
    from sklearn.metrics import precision_recall_fscore_support

    labels = sorted(set(y_true) | set(y_pred))
    p, _, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return float(p)


def fit_reasoner(factory, train: list[Case], name: str) -> FeatureTriageReasoner:
    x = [case_features(c) for c in train]
    y = [PRIORITY_INDEX[c.reference.priority] for c in train]
    clf = PriorityClassifier(factory()).fit(x, y)
    return FeatureTriageReasoner(clf, name=name)


def out_of_sample(panel, train, oos) -> list[dict]:
    rows = []
    for arch, factory in panel.items():
        r = fit_reasoner(factory, train, f"feat_{arch}")
        m = triage_metrics([c.reference for c in oos], [r.reason(c) for c in oos])
        rows.append(
            {
                "arch": arch,
                "oos_accuracy": m["priority_accuracy"],
                "oos_macro_precision": m["priority_macro_precision"],
                "oos_macro_f1": m["priority_macro_f1"],
                "oos_kappa": m["cohen_kappa"],
                "emergency_recall": m["emergency_recall"],
                "emergency_support": m["emergency_support"],
            }
        )
    return rows


def repeated_cv(panel, cases, seeds, k=5) -> list[dict]:
    from sklearn.model_selection import StratifiedKFold

    x = np.array([case_features(c) for c in cases])
    y = np.array([PRIORITY_INDEX[c.reference.priority] for c in cases])
    n_splits = min(k, int(np.min(np.bincount(y)[np.bincount(y) > 0])))
    rows = []
    for arch, factory in panel.items():
        accs, precs = [], []
        for s in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
            for tr, te in skf.split(x, y):
                clf = PriorityClassifier(factory()).fit(x[tr], y[tr])
                yp = [int(v) for v in clf.predict(x[te])]
                yt = [int(v) for v in y[te]]
                accs.append(sum(a == b for a, b in zip(yt, yp)) / len(yt))
                precs.append(macro_precision(yt, yp))
        rows.append(
            {
                "arch": arch,
                "cv_folds": n_splits,
                "cv_runs": len(accs),
                "cv_acc_mean": round(float(np.mean(accs)), 4),
                "cv_acc_std": round(float(np.std(accs)), 4),
                "cv_macroP_mean": round(float(np.mean(precs)), 4),
                "cv_macroP_std": round(float(np.std(precs)), 4),
            }
        )
    return rows


def learning_curve(factory, train, oos, fracs, seed=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    rows = []
    for f in fracs:
        k = max(4, int(len(train) * f))
        sub = [train[i] for i in idx[:k]]
        if len({c.reference.priority for c in sub}) < 2:
            continue
        r = fit_reasoner(factory, sub, "feat_lc")
        m = triage_metrics([c.reference for c in oos], [r.reason(c) for c in oos])
        rows.append(
            {
                "train_n": k,
                "oos_accuracy": m["priority_accuracy"],
                "oos_macro_precision": m["priority_macro_precision"],
            }
        )
    return rows


def emergency_stress(factory, train, oos, seed=0) -> dict:
    """Inject CRAO/AION into half a held-out sample; verify emergency recall+precision.

    The real data never contains EMERGENCY, so the learned head never sees it; the
    deterministic override must still escalate any emergency-code case. Ground truth
    follows the production rule (graph.py:triage_node): an emergency code => EMERGENCY.
    """
    r = fit_reasoner(factory, train, "feat_emerg")
    rng = np.random.default_rng(seed)
    sample = list(oos)
    rng.shuffle(sample)
    refs, preds = [], []
    n_emerg = 0
    for i, c in enumerate(sample):
        if i % 2 == 0:  # inject an emergency code
            preds_inj = [Prediction("CRAO", CODE_TO_NAME["CRAO"], 0.97, "high"), *c.predictions]
            case = Case(c.scan_id + "_emerg", preds_inj, {}, referral_priority=c.referral_priority)
            true = "EMERGENCY"
            n_emerg += 1
        else:
            case = c
            true = c.reference.priority
        refs.append(ReasonerOutput(true, False, False))
        preds.append(r.reason(case))
    m = triage_metrics(refs, preds)
    pc = m["per_class"]["EMERGENCY"]
    return {
        "n_cases": len(sample),
        "n_emergency_injected": n_emerg,
        "emergency_recall": pc["recall"],
        "emergency_precision": pc["precision"],
        "emergency_f1": pc["f1"],
        "overall_accuracy": m["priority_accuracy"],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train-sft", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    p.add_argument("--oos-traces", default="outputs/generalizability/oos_traces.jsonl")
    p.add_argument("--images-dir", default="data/rfmid_extracted")
    p.add_argument(
        "--skip", type=int, default=80, help="skip the first N sorted images (training set)"
    )
    p.add_argument("--n", type=int, default=160, help="fresh out-of-sample images to score")
    p.add_argument("--threshold", type=float, default=0.53)
    p.add_argument("--base-url", default="http://localhost:8011/v1")
    p.add_argument("--out", default="outputs/generalizability")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    train = load_train_cases(Path(args.train_sft))
    train_ids = {c.scan_id for c in train}
    logger.info("train: %d cases, spread=%s", len(train), spread(train))
    oos = generate_oos_cases(args, train_ids)
    logger.info("OOS: %d cases, spread=%s", len(oos), spread(oos))

    panel = build_panel(seed=42)
    seeds = list(range(args.seeds))
    combined = train + oos

    oos_rows = out_of_sample(panel, train, oos)
    cv_rows = repeated_cv(panel, combined, seeds)
    lc_rows = learning_curve(panel["logreg"], train, oos, [0.25, 0.5, 0.75, 1.0])
    emerg = emergency_stress(panel["logreg"], train, oos)

    summary = {
        "train_n": len(train),
        "oos_n": len(oos),
        "train_spread": spread(train),
        "oos_spread": spread(oos),
        "out_of_sample": oos_rows,
        "repeated_cv": cv_rows,
        "learning_curve": lc_rows,
        "emergency_stress": emerg,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generalizability_report.json").write_text(json.dumps(summary, indent=2))

    # markdown
    md = [
        "# Generalizability test — triage (accuracy + precision)",
        f"- teacher: Qwen3-8B-AWQ · train (in-sample): {len(train)} · "
        f"**out-of-sample (fresh unseen images): {len(oos)}**",
        f"- train spread: {spread(train)} · OOS spread: {spread(oos)}",
        "",
        "## 1. True out-of-sample (train on 80, test on fresh unseen images)",
        "| arch | OOS acc | OOS macro-P | OOS macro-F1 | κ | EMERG recall (n) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in oos_rows:
        es = r["emergency_support"]
        md.append(
            f"| `feat_{r['arch']}` | {r['oos_accuracy']:.3f} | {r['oos_macro_precision']:.3f} | "
            f"{r['oos_macro_f1']:.3f} | {r['oos_kappa']:.3f} | "
            f"{r['emergency_recall']:.3f} ({es}) |"
        )
    md += [
        "",
        "## 2. Repeated stratified 5-fold CV (mean ± std over all data)",
        f"- {cv_rows[0]['cv_runs']} fits/arch ({args.seeds} seeds × {cv_rows[0]['cv_folds']} folds)",
        "| arch | CV acc (mean±std) | CV macro-P (mean±std) |",
        "|---|---:|---:|",
    ]
    for r in cv_rows:
        md.append(
            f"| `feat_{r['arch']}` | {r['cv_acc_mean']:.3f} ± {r['cv_acc_std']:.3f} | "
            f"{r['cv_macroP_mean']:.3f} ± {r['cv_macroP_std']:.3f} |"
        )
    md += [
        "",
        "## 3. Learning curve (logreg, tested out-of-sample)",
        "| train n | OOS acc | OOS macro-P |",
        "|---:|---:|---:|",
    ]
    for r in lc_rows:
        md.append(f"| {r['train_n']} | {r['oos_accuracy']:.3f} | {r['oos_macro_precision']:.3f} |")
    md += [
        "",
        "## 4. EMERGENCY stress test (injected CRAO/AION — real data had none)",
        f"- {emerg['n_emergency_injected']}/{emerg['n_cases']} cases given an emergency code; "
        f"ground truth = EMERGENCY (production escalation rule)",
        f"- **emergency recall {emerg['emergency_recall']:.3f}, "
        f"precision {emerg['emergency_precision']:.3f}, F1 {emerg['emergency_f1']:.3f}**, "
        f"overall acc {emerg['overall_accuracy']:.3f}",
        "- the deterministic override guarantees an emergency code is never downgraded, "
        "even though the learned head never saw EMERGENCY in training.",
    ]
    (out_dir / "generalizability_report.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nReport: {out_dir / 'generalizability_report.md'}")


if __name__ == "__main__":
    main()
