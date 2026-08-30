#!/usr/bin/env python3
"""Build the DistilledQwen SFT dataset from teacher traces.

The traces written by ``run_real_qwen_comparison.py`` already embed the exact
predictions the teacher saw (``predictions``/``referral``) plus its answer
(``priority``/flags/``reasoning``/``narrative``), so this step needs no
``model_service`` and is deterministic — it just reshapes each trace into an SFT
row whose ``prompt`` is built by the shared ``build_distill_prompt`` (identical to
inference) and whose ``target`` is the teacher's JSON answer.

    PYTHONPATH=. python scripts/build_sft_dataset.py \
        --traces outputs/reasoner_comparison_real/traces.jsonl \
        --out outputs/reasoner_comparison_real/sft_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reasoner_comparison.interface import Case, Prediction  # noqa: E402
from src.evaluation.reasoner_comparison.reasoners import build_distill_prompt  # noqa: E402

logger = logging.getLogger("build_sft")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--traces", default="outputs/reasoner_comparison_real/traces.jsonl")
    p.add_argument("--out", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    traces = [
        json.loads(line) for line in Path(args.traces).read_text().splitlines() if line.strip()
    ]
    logger.info("loaded %d teacher traces", len(traces))

    rows = []
    skipped = 0
    for tr in sorted(traces, key=lambda t: t["scan_id"]):  # deterministic order
        if "predictions" not in tr:
            skipped += 1  # legacy trace without stored predictions
            continue
        preds = [
            Prediction(
                p["code"], p["name"], float(p["probability"]), p.get("confidence", "unknown")
            )
            for p in tr["predictions"]
        ]
        case = Case(
            scan_id=tr["scan_id"],
            predictions=preds,
            probabilities={},
            referral_priority=tr.get("referral", "FOLLOW_UP"),
        )
        target = json.dumps(
            {
                "priority": tr["priority"],
                "should_explain": bool(tr.get("should_explain", False)),
                "should_review": bool(tr.get("should_review", False)),
                "reasoning": tr.get("reasoning", ""),
                "narrative": tr.get("narrative", ""),
            }
        )
        rows.append(
            {
                "scan_id": tr["scan_id"],
                "image_path": tr.get("image_path", ""),
                "referral": tr.get("referral", "FOLLOW_UP"),
                "predictions": tr["predictions"],
                "teacher": {
                    "priority": tr["priority"],
                    "should_explain": bool(tr.get("should_explain", False)),
                    "should_review": bool(tr.get("should_review", False)),
                    "reasoning": tr.get("reasoning", ""),
                    "narrative": tr.get("narrative", ""),
                    "teacher_latency_ms": float(tr.get("teacher_latency_ms", 0.0)),
                },
                "prompt": build_distill_prompt(case),
                "target": target,
            }
        )

    if skipped:
        logger.warning("skipped %d legacy traces without stored predictions", skipped)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    logger.info("wrote %d SFT rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
