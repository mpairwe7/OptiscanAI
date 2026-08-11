#!/usr/bin/env python3
"""Sweep lightweight *feature-based* triage architectures against the CNN.

The image-based ``cnn_triage`` underperforms the deterministic rules (macro-F1
0.520 / macro-precision 0.492) because it reconstructs triage from pixels, while
the teacher actually derives it from the classifier's *structured* output. This
script trains a panel of tiny tabular models on that structured signal
(:func:`features.case_features`) and scores them with the *same* metrics and the
*same* held-out split as the real comparison, so the numbers are directly
comparable. It also runs 5-fold cross-validation (generalizability) and a
no-referral ablation (how much is learnable from findings alone).

Each model is KB-scale and sub-millisecond, so the open question is purely
accuracy/precision — the goal is >0.75 accuracy with macro-precision at least the
CNN's, ideally matching the rules (1.000).

    PYTHONPATH=. /usr/bin/python3 scripts/sweep_triage_architectures.py \
        --sft outputs/reasoner_comparison_real/sft_data.jsonl \
        --out outputs/reasoner_comparison_real
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
from src.evaluation.reasoner_comparison.metrics import (  # noqa: E402
    evaluate_gates,
    ops_metrics,
    triage_metrics,
)
from src.evaluation.reasoner_comparison.reasoners import FeatureTriageReasoner  # noqa: E402

logger = logging.getLogger("triage_sweep")


def load_cases(sft_path: Path) -> list[Case]:
    rows = [json.loads(ln) for ln in sft_path.read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["scan_id"])  # deterministic, matches CNN eval order
    cases: list[Case] = []
    for r in rows:
        preds = [
            Prediction(p["code"], p["name"], float(p["probability"]), p.get("confidence", "unknown"))
            for p in r["predictions"]
        ]
        t = r["teacher"]
        case = Case(
            scan_id=r["scan_id"],
            predictions=preds,
            probabilities={},
            referral_priority=r.get("referral", "FOLLOW_UP"),
        )
        case.reference = ReasonerOutput(
            priority=t["priority"],
            should_explain=bool(t.get("should_explain", False)),
            should_review=bool(t.get("should_review", False)),
            reasoning=t.get("reasoning", ""),
            narrative=t.get("narrative", ""),
            source="qwen_teacher",
        )
        cases.append(case)
    return cases


def build_estimators(seed: int) -> dict[str, tuple]:
    """name -> (factory, extra_deps). Factories are zero-arg, return a fresh base."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    panel: dict[str, tuple] = {
        "logreg": (lambda: LogisticRegression(max_iter=2000, C=1.0), ()),
        "decision_tree": (lambda: DecisionTreeClassifier(max_depth=6, random_state=seed), ()),
        "random_forest": (
            lambda: RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=seed, n_jobs=1
            ),
            (),
        ),
        "hist_gboost": (
            lambda: HistGradientBoostingClassifier(max_depth=4, random_state=seed),
            (),
        ),
        "mlp": (
            lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=seed),
            (),
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        panel["lightgbm"] = (
            lambda: LGBMClassifier(
                n_estimators=200, max_depth=4, verbose=-1, random_state=seed, n_jobs=1
            ),
            ("lightgbm",),
        )
    except ImportError:
        logger.warning("lightgbm not available — skipping")
    try:
        from xgboost import XGBClassifier

        panel["xgboost"] = (
            lambda: XGBClassifier(
                n_estimators=200, max_depth=4, random_state=seed, verbosity=0, n_jobs=2
            ),
            ("xgboost",),
        )
    except ImportError:
        logger.warning("xgboost not available — skipping")
    return panel


def macro_precision(y_true: list[int], y_pred: list[int]) -> float:
    """Macro precision over observed labels (matches the harness convention)."""
    from sklearn.metrics import precision_recall_fscore_support

    labels = sorted(set(y_true) | set(y_pred))
    p, _, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return float(p)


def evaluate_on_split(reasoner: FeatureTriageReasoner, test: list[Case]) -> dict:
    refs = [c.reference for c in test]
    preds = [reasoner.reason(c) for c in test]
    triage = triage_metrics(refs, preds)
    ops = ops_metrics(reasoner.info(), preds)
    gate = evaluate_gates(triage, {}, ops)
    return {"triage": triage, "ops": ops, "gate": gate}


def cross_val(factory, X, y, seed: int) -> dict:
    """5-fold stratified CV accuracy + macro-precision on all data."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    X, y = np.asarray(X), np.asarray(y)
    n_splits = min(5, int(np.min(np.bincount(y)[np.bincount(y) > 0])))
    if n_splits < 2:
        return {"cv_accuracy": None, "cv_macro_precision": None, "cv_folds": 0}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, precs = [], []
    for tr, te in skf.split(X, y):
        clf = PriorityClassifier(factory()).fit(X[tr], y[tr])
        yp = [int(v) for v in clf.predict(X[te])]
        yt = [int(v) for v in y[te]]
        accs.append(sum(a == b for a, b in zip(yt, yp)) / len(yt))
        precs.append(macro_precision(yt, yp))
    return {
        "cv_accuracy": round(float(np.mean(accs)), 4),
        "cv_macro_precision": round(float(np.mean(precs)), 4),
        "cv_folds": n_splits,
    }


def run_panel(cases: list[Case], include_referral: bool, seed: int, cut_frac: float) -> list[dict]:
    X = [case_features(c, include_referral=include_referral) for c in cases]
    y = [PRIORITY_INDEX[c.reference.priority] for c in cases]
    cut = int(len(cases) * cut_frac)
    Xtr, ytr, test = X[:cut], y[:cut], cases[cut:]

    rows: list[dict] = []
    for name, (factory, deps) in build_estimators(seed).items():
        clf = PriorityClassifier(factory()).fit(Xtr, ytr)
        reasoner = FeatureTriageReasoner(
            clf,
            name=f"feat_{name}" + ("" if include_referral else "_noref"),
            include_referral=include_referral,
            extra_deps=deps,
        )
        split = evaluate_on_split(reasoner, test)
        cv = cross_val(factory, X, y, seed)
        t, o = split["triage"], split["ops"]
        rows.append(
            {
                "arch": name,
                "include_referral": include_referral,
                "holdout_accuracy": t["priority_accuracy"],
                "holdout_macro_f1": t["priority_macro_f1"],
                "holdout_macro_precision": t["priority_macro_precision"],
                "cohen_kappa": t["cohen_kappa"],
                "cv_accuracy": cv["cv_accuracy"],
                "cv_macro_precision": cv["cv_macro_precision"],
                "cv_folds": cv["cv_folds"],
                "size_mb": round(o["size_mb"], 4),
                "latency_p95_ms": o["latency_p95_ms"],
                "gate_pass": split["gate"]["passed"],
                "extra_deps": list(deps),
                "_estimator": clf,
                "_reasoner_name": reasoner.name,
            }
        )
    rows.sort(key=lambda r: (r["holdout_macro_f1"], -r["size_mb"]), reverse=True)
    return rows


def fmt_table(rows: list[dict], title: str) -> str:
    lines = [
        f"### {title}",
        "",
        "| arch | hold acc | hold macroF1 | hold macroP | κ | CV acc | CV macroP | size MB | p95 ms | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in rows:
        cva = "—" if r["cv_accuracy"] is None else f"{r['cv_accuracy']:.3f}"
        cvp = "—" if r["cv_macro_precision"] is None else f"{r['cv_macro_precision']:.3f}"
        lines.append(
            f"| `{r['arch']}` | {r['holdout_accuracy']:.3f} | {r['holdout_macro_f1']:.3f} | "
            f"{r['holdout_macro_precision']:.3f} | {r['cohen_kappa']:.3f} | {cva} | {cvp} | "
            f"{r['size_mb']:.3f} | {r['latency_p95_ms']:.2f} | "
            f"{'✅' if r['gate_pass'] else '❌'} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sft", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    p.add_argument("--out", default="outputs/reasoner_comparison_real")
    p.add_argument("--model-out", default="outputs/triage_model")
    p.add_argument("--cut-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    cases = load_cases(Path(args.sft))
    dist: dict[str, int] = {}
    for c in cases:
        dist[c.reference.priority] = dist.get(c.reference.priority, 0) + 1
    logger.info("loaded %d cases; teacher priority spread=%s", len(cases), dist)

    full = run_panel(cases, include_referral=True, seed=args.seed, cut_frac=args.cut_frac)
    ablation = run_panel(cases, include_referral=False, seed=args.seed, cut_frac=args.cut_frac)

    # CNN / rule reference rows (from the executed real comparison).
    ref_note = (
        "**Reference (real run, same split):** "
        "`rule_baseline` acc 1.000 / macroF1 1.000 / macroP 1.000 (size 0, gate ✅); "
        "`cnn_triage` acc 0.792 / macroF1 0.520 / macroP 0.492 (size 6.1 MB, gate ❌)."
    )
    md = "\n\n".join(
        [
            "# Lightweight triage architecture sweep",
            f"- teacher: Qwen3-8B-AWQ · cases: {len(cases)} · split: "
            f"{int(len(cases) * args.cut_frac)} train / {len(cases) - int(len(cases) * args.cut_frac)} test · "
            f"5-fold CV on all {len(cases)}",
            ref_note,
            fmt_table(full, "With referral feature (realistic inference input)"),
            fmt_table(ablation, "Ablation — findings only, no referral feature"),
            "_Every feature model is KB-scale and sub-ms, clearing the 60 MB / 1800 ms "
            "edge gates by orders of magnitude; an emergency code deterministically "
            "escalates to EMERGENCY regardless of the learned head._",
        ]
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "triage_sweep.md").write_text(md + "\n")

    def _clean(rows):
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    (out_dir / "triage_sweep.json").write_text(
        json.dumps(
            {"n_cases": len(cases), "spread": dist, "with_referral": _clean(full),
             "no_referral": _clean(ablation)},
            indent=2,
        )
    )

    # Save the best model (highest held-out macro-F1, smallest size on ties).
    best = full[0]
    import joblib

    model_dir = Path(args.model_out)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["_estimator"], model_dir / "triage_model.joblib")
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "arch": best["arch"],
                "reasoner_name": best["_reasoner_name"],
                "include_referral": True,
                "holdout_accuracy": best["holdout_accuracy"],
                "holdout_macro_f1": best["holdout_macro_f1"],
                "holdout_macro_precision": best["holdout_macro_precision"],
                "cv_accuracy": best["cv_accuracy"],
                "cv_macro_precision": best["cv_macro_precision"],
                "size_mb": best["size_mb"],
                "extra_deps": best["extra_deps"],
                "feature_order": "see features.feature_names(include_referral=True)",
            },
            indent=2,
        )
    )

    print("\n" + md + "\n")
    print(f"best -> {best['arch']} (held-out macroF1={best['holdout_macro_f1']:.3f}, "
          f"macroP={best['holdout_macro_precision']:.3f}); saved to {model_dir}")


if __name__ == "__main__":
    main()
