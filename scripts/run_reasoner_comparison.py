#!/usr/bin/env python3
"""Run the clinical-reasoner comparison (CNN vs DistilledQwen vs baselines).

Two modes:

    # Offline smoke run — synthetic cases, micro-trained CNN, no GPU/keys needed.
    # Validates the harness end-to-end and proves the CNN can learn the triage map.
    PYTHONPATH=. python scripts/run_reasoner_comparison.py --mode smoke

    # Real run (GATED — needs RFMiD images + teacher traces; multi-hour CNN train).
    PYTHONPATH=. python scripts/run_reasoner_comparison.py --mode real \
        --images data/rfmid/test --traces outputs/reasoner_comparison/traces.jsonl \
        --epochs 40 --img-size 224 --device cuda:2 \
        --distilled-dir outputs/distilled_qwen   # optional, if trained

Outputs: outputs/reasoner_comparison/{results.json,report.md}.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reasoner_comparison.cases import (  # noqa: E402
    CODE_TO_NAME,
    load_real_cases,
    make_synthetic_cases,
)
from src.evaluation.reasoner_comparison.cnn import train_triage_cnn  # noqa: E402
from src.evaluation.reasoner_comparison.reasoners import (  # noqa: E402
    CNNTriageReasoner,
    DistilledLLMReasoner,
    LLMReasoner,
    RuleReasoner,
)
from src.evaluation.reasoner_comparison.runner import run_comparison  # noqa: E402

logger = logging.getLogger("reasoner_comparison")


def _split(cases, frac: float = 0.7):
    cut = int(len(cases) * frac)
    return cases[:cut], cases[cut:]


def _maybe_llm(reasoners: list) -> None:
    try:
        reasoners.append(LLMReasoner(name="llm_teacher"))
        logger.info("LLM teacher reasoner enabled")
    except Exception as e:  # no keys / no provider — expected in smoke
        logger.info("LLM teacher reasoner skipped: %s", e)


def _maybe_distilled(reasoners: list, model_dir: str | None, device: str) -> None:
    if not model_dir:
        return
    try:
        reasoners.append(DistilledLLMReasoner(model_dir, device=device))
        logger.info("DistilledQwen reasoner enabled from %s", model_dir)
    except Exception as e:
        logger.warning("DistilledQwen reasoner skipped: %s", e)


def run_smoke(args) -> dict:
    import torch

    torch.set_num_threads(max(1, args.threads))
    logger.info("generating %d synthetic cases (img_size=%d)…", args.n, args.img_size)
    cases = make_synthetic_cases(n=args.n, seed=args.seed, img_size=args.img_size)
    train, test = _split(cases)
    logger.info("micro-training TriageCNN on %d cases for %d epochs…", len(train), args.epochs)
    model = train_triage_cnn(
        train, epochs=args.epochs, img_size=args.img_size, device=args.device, seed=args.seed
    )

    reasoners: list = [
        RuleReasoner(),
        CNNTriageReasoner(model, img_size=args.img_size, device=args.device),
    ]
    _maybe_llm(reasoners)
    _maybe_distilled(reasoners, args.distilled_dir, args.device)

    return run_comparison(
        test, reasoners, CODE_TO_NAME, args.out, mode="smoke", teacher_source="synthetic_teacher"
    )


def run_real(args) -> dict:
    if not (args.images and args.traces):
        raise SystemExit("real mode requires --images and --traces")
    logger.info("loading real cases from %s …", args.images)
    cases = load_real_cases(args.images, args.traces, limit=args.limit)
    if not cases:
        raise SystemExit("no real cases loaded (check images/traces)")
    train, test = _split(cases)

    if args.cnn_checkpoint:
        import torch

        from src.evaluation.reasoner_comparison.cnn import TriageCNN

        model = TriageCNN()
        model.load_state_dict(torch.load(args.cnn_checkpoint, map_location=args.device))
    else:
        logger.info("training TriageCNN on %d real cases for %d epochs…", len(train), args.epochs)
        model = train_triage_cnn(
            train, epochs=args.epochs, img_size=args.img_size, device=args.device, pretrained=True
        )

    reasoners: list = [
        RuleReasoner(),
        CNNTriageReasoner(model, img_size=args.img_size, device=args.device),
    ]
    _maybe_llm(reasoners)
    _maybe_distilled(reasoners, args.distilled_dir, args.device)

    return run_comparison(
        test, reasoners, CODE_TO_NAME, args.out, mode="real", teacher_source="llm_teacher"
    )


def _print_summary(payload: dict) -> None:
    print("\n=== Reasoner comparison summary ===")
    print(f"mode={payload['mode']} teacher={payload['teacher_source']} cases={payload['n_cases']}")
    for name, r in payload["reasoners"].items():
        t, o, g = r["triage"], r["ops"], r["gate"]
        print(
            f"  {name:16s} macroF1={t.get('priority_macro_f1', 0):.3f} "
            f"EMERG_recall={t.get('emergency_recall', 0):.3f} "
            f"size={o.get('size_mb', 0):.1f}MB p95={o.get('latency_p95_ms', 0):.1f}ms "
            f"narrates={o.get('generates_narrative')} gate={'PASS' if g['passed'] else 'FAIL'}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["smoke", "real"], default="smoke")
    p.add_argument("--out", default="outputs/reasoner_comparison")
    p.add_argument("--n", type=int, default=240, help="synthetic cases (smoke)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--img-size", type=int, default=64, help="64 for smoke, 224 for real")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (smoke)")
    p.add_argument("--distilled-dir", default=None, help="trained DistilledQwen checkpoint dir")
    # real-mode
    p.add_argument("--images", default=None)
    p.add_argument("--traces", default=None)
    p.add_argument("--cnn-checkpoint", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    payload = run_real(args) if args.mode == "real" else run_smoke(args)
    _print_summary(payload)
    print(f"\nReport: {Path(args.out) / 'report.md'}")
    print(f"JSON:   {Path(args.out) / 'results.json'}")


if __name__ == "__main__":
    main()
