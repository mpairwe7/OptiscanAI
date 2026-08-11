#!/usr/bin/env python3
"""Real reasoner comparison using the live self-hosted Qwen as teacher.

End-to-end, CPU-friendly pipeline:

    RFMiD image --model_service--> predictions --Qwen(localhost:8011)--> teacher
    triage+narrative (cached to traces.jsonl) --> train TriageCNN --> compare
    {rule_baseline, cnn_triage, qwen_teacher(reference)} on a held-out split.

Teacher calls dominate the wall-clock, so trace generation is the slow part; it is
cached and resumable (re-runs skip scan_ids already in traces.jsonl). This script
runs the 3-way {rule_baseline, cnn_triage, qwen_teacher} comparison and writes the
traces the DistilledQwen path consumes. The full 4-way comparison that adds the
distilled narrator lives in ``scripts/run_final_comparison.py`` (runs under system
python on GPU); see docs/28-reasoner-cnn-vs-distilledqwen.md §0 for executed
results.

    PYTHONPATH=. python scripts/run_real_qwen_comparison.py --n 160 --img-size 128 --epochs 18
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reasoner_comparison.cases import CODE_TO_NAME as SYNTH_NAMES  # noqa: E402
from src.evaluation.reasoner_comparison.cnn import train_triage_cnn  # noqa: E402
from src.evaluation.reasoner_comparison.interface import (  # noqa: E402
    Case,
    Prediction,
    Reasoner,
    ReasonerOutput,
)
from src.evaluation.reasoner_comparison.reasoners import (
    CNNTriageReasoner,
    RuleReasoner,
)  # noqa: E402
from src.evaluation.reasoner_comparison.runner import run_comparison  # noqa: E402
from src.evaluation.reasoner_comparison.vllm_teacher import VLLMReasoner  # noqa: E402

logger = logging.getLogger("real_qwen_comparison")


class PrecomputedReasoner(Reasoner):
    """Replays stored teacher outputs (the reference) — no network, real latency."""

    offline = False
    generates_narrative = True
    extra_deps = ("self-hosted vLLM endpoint",)

    def __init__(self, name: str, by_scan: dict[str, ReasonerOutput]):
        self.name = name
        self._by_scan = by_scan

    def _reason(self, case: Case) -> ReasonerOutput:
        return self._by_scan[case.scan_id]


def _load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                cache[obj["scan_id"]] = obj
    return cache


def build_cases(args) -> tuple[list[Case], dict[str, str]]:
    from PIL import Image

    from backend.app.core.model_service import model_service

    model_service.load()  # predict() does NOT auto-load; without this, 0 detections
    images = sorted(
        glob.glob(f"{args.images_dir}/**/*.png", recursive=True)
        + glob.glob(f"{args.images_dir}/**/*.jpg", recursive=True)
    )[: args.n]
    if not images:
        raise SystemExit(f"no images under {args.images_dir}")

    traces_path = Path(args.traces)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(traces_path)
    logger.info("%d images; %d teacher traces already cached", len(images), len(cache))

    teacher = None  # lazy: only construct if we actually need to call it
    code_to_name = dict(SYNTH_NAMES)
    cases: list[Case] = []
    fh = open(traces_path, "a")
    try:
        for i, img_path in enumerate(images):
            scan_id = Path(img_path).stem
            result = model_service.predict(
                Image.open(img_path).convert("RGB"), threshold=args.threshold
            )
            preds = [
                Prediction(
                    p["code"], p["name"], float(p["probability"]), p.get("confidence", "unknown")
                )
                for p in result.get("predictions", [])
            ]
            for p in preds:
                code_to_name[p.code] = p.name
            probs = {
                k: (v["probability"] if isinstance(v, dict) else float(v))
                for k, v in result.get("all_probabilities", {}).items()
            }
            referral = result.get("clinical", {}).get("referral_priority", "FOLLOW_UP")
            case = Case(
                scan_id=scan_id,
                predictions=preds,
                probabilities=probs,
                referral_priority=referral,
                image=img_path,
            )

            trace = cache.get(scan_id)
            if trace is None:
                if teacher is None:
                    teacher = VLLMReasoner(base_url=args.base_url)
                t0 = time.time()
                out = teacher.reason(case)
                trace = {
                    "scan_id": scan_id,
                    "image_path": img_path,
                    "referral": referral,
                    "detected": case.detected_codes,
                    # Store the exact predictions the teacher saw so downstream SFT
                    # data reuses them (predict() is non-deterministic via MC-dropout).
                    "predictions": [
                        {
                            "code": p.code,
                            "name": p.name,
                            "probability": p.probability,
                            "confidence": p.confidence,
                        }
                        for p in preds
                    ],
                    "priority": out.priority,
                    "should_explain": out.should_explain,
                    "should_review": out.should_review,
                    "reasoning": out.reasoning,
                    "narrative": out.narrative,
                    "teacher_latency_ms": round((time.time() - t0) * 1000, 1),
                }
                fh.write(json.dumps(trace) + "\n")
                fh.flush()
                logger.info(
                    "[%d/%d] %s -> %s (%.1fs)",
                    i + 1,
                    len(images),
                    scan_id,
                    out.priority,
                    (time.time() - t0),
                )

            ref = ReasonerOutput(
                priority=trace["priority"],
                should_explain=bool(trace.get("should_explain", False)),
                should_review=bool(trace.get("should_review", False)),
                reasoning=trace.get("reasoning", ""),
                narrative=trace.get("narrative", ""),
                source="qwen_teacher",
                latency_ms=float(trace.get("teacher_latency_ms", 0.0)),
            )
            case.reference = ref
            cases.append(case)
    finally:
        fh.close()
    return cases, code_to_name


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--images-dir", default="data/rfmid_extracted")
    p.add_argument("--n", type=int, default=160)
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="detection threshold; raise it to thin over-detection into a varied triage mix",
    )
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--base-url", default="http://localhost:8011/v1")
    p.add_argument("--out", default="outputs/reasoner_comparison_real")
    p.add_argument("--traces", default="outputs/reasoner_comparison_real/traces.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    import torch

    torch.set_num_threads(max(1, args.threads))

    cases, code_to_name = build_cases(args)
    cut = int(len(cases) * 0.7)
    train, test = cases[:cut], cases[cut:]
    logger.info(
        "training TriageCNN on %d real cases (%d test) for %d epochs…",
        len(train),
        len(test),
        args.epochs,
    )
    model = train_triage_cnn(
        train, epochs=args.epochs, img_size=args.img_size, device=args.device, seed=args.seed
    )

    refs_by_scan = {c.scan_id: c.reference for c in test}
    candidates = [
        RuleReasoner(),
        CNNTriageReasoner(model, img_size=args.img_size, device=args.device),
        PrecomputedReasoner("qwen_teacher", refs_by_scan),
    ]
    payload = run_comparison(
        test,
        candidates,
        code_to_name,
        args.out,
        mode="real",
        teacher_source="qwen_teacher (Qwen3-8B-AWQ @ vLLM)",
    )
    print("\n=== REAL comparison (teacher = live Qwen3-8B-AWQ) ===")
    print(f"cases: train={len(train)} test={len(test)}")
    for name, r in payload["reasoners"].items():
        t, o, g = r["triage"], r["ops"], r["gate"]
        print(
            f"  {name:14s} macroF1={t.get('priority_macro_f1', 0):.3f} EMERG_rec={t.get('emergency_recall', 0):.3f} "
            f"grounding={r['narrative'].get('grounding', 0):.3f} words={r['narrative'].get('avg_words', 0):.0f} "
            f"size={o.get('size_mb', 0):.1f}MB gate={'PASS' if g['passed'] else 'FAIL'}"
        )
    print(f"\nReport: {Path(args.out) / 'report.md'}")


if __name__ == "__main__":
    main()
