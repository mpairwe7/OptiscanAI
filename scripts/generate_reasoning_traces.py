#!/usr/bin/env python3
"""Generate teacher reasoning traces — the (net-new) label/distillation dataset.

The CNN and the DistilledQwen both need a *teacher signal*: for each screening
case, the priority/flags/reasoning and a clinical narrative the teacher (the
production LLM — Gemini, or a self-hosted Qwen endpoint) would produce.
This script writes that as JSONL, one object per line:

    {"scan_id","detected","priority","should_explain","should_review",
     "reasoning","narrative"}

Modes:

    # Real: run the teacher LLM over RFMiD images (needs model_service + a
    # provider key). This is the multi-hour / API-cost step quantified in
    # docs/28-reasoner-cnn-vs-distilledqwen.md.
    PYTHONPATH=. python scripts/generate_reasoning_traces.py --mode llm \
        --images data/rfmid/train --out outputs/reasoner_comparison/traces.jsonl

    # Synthetic: emit traces from the deterministic stand-in teacher (no LLM,
    # no GPU) — for wiring up and testing the downstream pipeline only.
    PYTHONPATH=. python scripts/generate_reasoning_traces.py --mode synthetic \
        --n 200 --out outputs/reasoner_comparison/traces_synthetic.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("generate_traces")


def _trace_line(scan_id: str, detected: list[str], out) -> str:
    return json.dumps(
        {
            "scan_id": scan_id,
            "detected": detected,
            "priority": out.priority,
            "should_explain": out.should_explain,
            "should_review": out.should_review,
            "reasoning": out.reasoning,
            "narrative": out.narrative,
        }
    )


def gen_synthetic(args) -> int:
    from src.evaluation.reasoner_comparison.cases import make_synthetic_cases

    cases = make_synthetic_cases(n=args.n, seed=args.seed, with_images=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for c in cases:
            fh.write(_trace_line(c.scan_id, c.detected_codes, c.reference) + "\n")
    logger.info("wrote %d synthetic traces -> %s", len(cases), out_path)
    return len(cases)


def gen_llm(args) -> int:
    from PIL import Image

    from backend.app.core.model_service import model_service
    from src.evaluation.reasoner_comparison.cases import Case, Prediction
    from src.evaluation.reasoner_comparison.reasoners import LLMReasoner

    teacher = LLMReasoner(name="teacher")  # raises if no provider configured
    images = sorted(Path(args.images).glob("*.png")) + sorted(Path(args.images).glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images under {args.images}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as fh:
        for img_path in images:
            result = model_service.predict(Image.open(img_path).convert("RGB"))
            predictions = [
                Prediction(
                    code=p["code"],
                    name=p["name"],
                    probability=float(p["probability"]),
                    confidence=p.get("confidence", "unknown"),
                )
                for p in result.get("predictions", [])
            ]
            probs = {
                k: (v["probability"] if isinstance(v, dict) else float(v))
                for k, v in result.get("all_probabilities", {}).items()
            }
            case = Case(
                scan_id=img_path.stem,
                predictions=predictions,
                probabilities=probs,
                referral_priority=result.get("clinical", {}).get("referral_priority", "FOLLOW_UP"),
            )
            out = teacher.reason(case)
            fh.write(_trace_line(case.scan_id, case.detected_codes, out) + "\n")
            written += 1
            if written % 25 == 0:
                logger.info("…%d traces", written)
    logger.info("wrote %d LLM teacher traces -> %s", written, out_path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["llm", "synthetic"], default="synthetic")
    p.add_argument("--out", required=True)
    p.add_argument("--images", default=None, help="RFMiD image dir (llm mode)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n", type=int, default=200, help="synthetic case count")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    if args.mode == "llm":
        if not args.images:
            raise SystemExit("--mode llm requires --images")
        gen_llm(args)
    else:
        gen_synthetic(args)


if __name__ == "__main__":
    main()
