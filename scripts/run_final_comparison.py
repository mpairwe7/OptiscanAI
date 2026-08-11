#!/usr/bin/env python3
"""Final 4-way reasoner comparison (runs under system python, on GPU).

Reconstructs cases from the self-contained ``sft_data.jsonl`` (no ``model_service``
needed), trains the TriageCNN on GPU, and scores every candidate on one held-out
split against the live-Qwen teacher reference:

    rule_baseline · cnn_triage · distilled_qwen · qwen_teacher(reference)

The test split matches ``distill_qwen_reasoner.py`` (same ``--val-frac`` over the
same file order), so the DistilledQwen is evaluated on data it never trained on.

    PYTHONPATH=. /usr/bin/python3 scripts/run_final_comparison.py \
        --data outputs/reasoner_comparison_real/sft_data.jsonl \
        --distilled-dir outputs/distilled_qwen --device cuda:2 \
        --img-size 224 --epochs 25 --out outputs/reasoner_comparison_real
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reasoner_comparison.cnn import train_triage_cnn  # noqa: E402
from src.evaluation.reasoner_comparison.interface import (
    Case,
    Prediction,
    ReasonerOutput,
)  # noqa: E402
from src.evaluation.reasoner_comparison.reasoners import (  # noqa: E402
    CNNTriageReasoner,
    DistilledLLMReasoner,
    PrecomputedReasoner,
    RuleReasoner,
)
from src.evaluation.reasoner_comparison.runner import run_comparison  # noqa: E402

logger = logging.getLogger("final_comparison")


def rows_to_cases(rows: list[dict]) -> tuple[list[Case], dict[str, str]]:
    code_to_name: dict[str, str] = {}
    cases: list[Case] = []
    for r in rows:
        preds = [
            Prediction(
                p["code"], p["name"], float(p["probability"]), p.get("confidence", "unknown")
            )
            for p in r["predictions"]
        ]
        for p in preds:
            code_to_name[p.code] = p.name
        t = r["teacher"]
        cases.append(
            Case(
                scan_id=r["scan_id"],
                predictions=preds,
                probabilities={p.code: p.probability for p in preds},
                referral_priority=r["referral"],
                image=r["image_path"],
                reference=ReasonerOutput(
                    priority=t["priority"],
                    should_explain=bool(t.get("should_explain", False)),
                    should_review=bool(t.get("should_review", False)),
                    reasoning=t.get("reasoning", ""),
                    narrative=t.get("narrative", ""),
                    source="qwen_teacher",
                    latency_ms=float(t.get("teacher_latency_ms", 0.0)),
                ),
            )
        )
    return cases, code_to_name


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    p.add_argument("--distilled-dir", default="outputs/distilled_qwen")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--val-frac", type=float, default=0.3)
    p.add_argument("--out", default="outputs/reasoner_comparison_real")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    cases, code_to_name = rows_to_cases(rows)
    cut = int(len(cases) * (1 - args.val_frac))
    train, test = cases[:cut], cases[cut:]
    logger.info("cases: train=%d test=%d device=%s", len(train), len(test), device)

    logger.info("training TriageCNN on GPU (%d epochs, img=%d)…", args.epochs, args.img_size)
    model = train_triage_cnn(
        train,
        epochs=args.epochs,
        img_size=args.img_size,
        device=device,
        pretrained=True,
        seed=args.seed,
    )

    reasoners: list = [
        RuleReasoner(),
        CNNTriageReasoner(model, img_size=args.img_size, device=device),
        PrecomputedReasoner({c.scan_id: c.reference for c in test}, name="qwen_teacher"),
    ]
    if Path(args.distilled_dir).exists():
        try:
            reasoners.insert(2, DistilledLLMReasoner(args.distilled_dir, device=device))
            logger.info("DistilledQwen loaded from %s", args.distilled_dir)
        except Exception as e:
            logger.warning("DistilledQwen skipped: %s", e)
    else:
        logger.warning("no distilled model at %s — running without it", args.distilled_dir)

    payload = run_comparison(
        test,
        reasoners,
        code_to_name,
        args.out,
        mode="real",
        teacher_source="qwen_teacher (Qwen3-8B-AWQ @ vLLM)",
    )
    print("\n=== FINAL real comparison (teacher = live Qwen3-8B-AWQ) ===")
    print(f"cases: train={len(train)} test={len(test)}")
    for name, r in payload["reasoners"].items():
        t, o, g, n = r["triage"], r["ops"], r["gate"], r["narrative"]
        print(
            f"  {name:15s} macroF1={t.get('priority_macro_f1', 0):.3f} "
            f"EMERG={t.get('emergency_recall', 0):.3f}(n={t.get('emergency_support', 0)}) "
            f"acc={t.get('priority_accuracy', 0):.3f} "
            f"grounding={n.get('grounding', 0):.3f} words={n.get('avg_words', 0):.0f} "
            f"size={o.get('size_mb', 0):.0f}MB gate={'PASS' if g['passed'] else 'FAIL'}"
        )
    print(f"\nReport: {Path(args.out) / 'report.md'}")


if __name__ == "__main__":
    main()
