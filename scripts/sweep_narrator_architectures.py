#!/usr/bin/env python3
"""Compare narrator base-model architectures (size/latency/quality tradeoff).

Companion to ``quantize_distilled_reasoner.py`` (which sweeps precision on one
base model). This sweeps **base model** instead: the distilled Qwen2.5-0.5B
narrator (988 MB bf16) is the current pick, but its size floor is set by a
151936-token vocabulary (~272 MB embedding table alone) rather than by depth.
HuggingFaceTB SmolLM2 checkpoints share the same LoRA-SFT recipe
(``distill_qwen_reasoner.py --base-model ...``) but use a 49152-token
vocabulary, so this checks whether a smaller-vocab base model gets closer to
(or clears) the 60 MB edge gate without giving up grounding.

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 /usr/bin/python3 \
        scripts/sweep_narrator_architectures.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reasoner_comparison.interface import (  # noqa: E402
    Case,
    Prediction,
    ReasonerOutput,
)
from src.evaluation.reasoner_comparison.metrics import (  # noqa: E402
    DEFAULT_GATES,
    SERVER_GATES,
    evaluate_gates,
    narrative_metrics,
    ops_metrics,
    triage_metrics,
)
from src.evaluation.reasoner_comparison.reasoners import DistilledLLMReasoner  # noqa: E402

logger = logging.getLogger("sweep_narrator")


def load_test_cases(sft_path: Path, cut_frac: float) -> tuple[list[Case], dict[str, str]]:
    rows = [json.loads(ln) for ln in sft_path.read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["scan_id"])
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
    cut = int(len(cases) * cut_frac)
    return cases[cut:], code_to_name


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    The held-out split is small (n=24), so a bare point estimate hides how weak
    the evidence is: 0.917 vs 1.000 is a two-case difference. Wilson is the
    standard choice at small n / proportions near 0 or 1, where the normal
    approximation misbehaves.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3))


def measure(reasoner: DistilledLLMReasoner, cases: list[Case], code_to_name: dict) -> dict:
    refs = [c.reference for c in cases]
    preds = [reasoner.reason(c) for c in cases]
    triage = triage_metrics(refs, preds)
    narrative = narrative_metrics(cases, preds, code_to_name)
    ops = ops_metrics(reasoner.info(), preds)
    edge_gate = evaluate_gates(triage, narrative, ops, DEFAULT_GATES)
    server_gate = evaluate_gates(triage, narrative, ops, SERVER_GATES)
    # grounding on the *template* fallback is 1.0 by construction (it only
    # states detected diseases), so it can mask a model that rarely emits a
    # usable generative narrative. Report the generation success rate
    # alongside grounding so that distinction stays visible.
    n_generated = sum(1 for p in preds if p.narrative_generated)
    return {
        "size_mb": round(reasoner.size_mb(), 1),
        "triage_macro_f1": triage["priority_macro_f1"],
        "grounding": narrative["grounding"],
        "avg_words": narrative["avg_words"],
        "latency_p50_ms": ops["latency_p50_ms"],
        "latency_p95_ms": ops["latency_p95_ms"],
        "edge_gate_pass": edge_gate["passed"],
        "server_gate_pass": server_gate["passed"],
        "generation_rate": round(n_generated / len(preds), 3) if preds else 0.0,
        "generation_rate_ci95": wilson_ci(n_generated, len(preds)),
        "n_generated": n_generated,
        "n_cases": len(preds),
        "sample": preds[0].narrative[:200] if preds else "",
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sft", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    p.add_argument("--out", default="outputs/reasoner_comparison_real/narrator_arch_sweep.json")
    p.add_argument("--cut-frac", type=float, default=0.7)
    # The teacher's JSON targets run to ~252 tokens (p50 197, p90 230). A 200-token
    # cap truncates ~40% of them mid-JSON, which shows up as a "parse failure" and
    # silently penalises whichever model writes longer prose — a harness artifact,
    # not a model property. Budget must clear the longest target with headroom.
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--models",
        default=(
            "qwen0.5b=outputs/distilled_qwen,"
            "smollm2_360m=outputs/distilled_smollm2_360m,"
            "smollm2_135m=outputs/distilled_smollm2_135m"
        ),
        help="comma-separated name=checkpoint_dir pairs; every model is scored at every --precision",
    )
    p.add_argument("--precisions", default="bf16,nf4", help="comma-separated: bf16,fp32,nf4")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible — set CUDA_VISIBLE_DEVICES to a free GPU")

    cases, code_to_name = load_test_cases(Path(args.sft), args.cut_frac)
    logger.info("test cases: %d (device=%s)", len(cases), args.device)

    precision_opts = {
        "fp32": {"load_4bit": False, "dtype": None},
        "bf16": {"load_4bit": False, "dtype": "bfloat16"},
        "nf4": {"load_4bit": True, "dtype": None},
    }
    candidates = []
    for pair in args.models.split(","):
        name, _, model_dir = pair.partition("=")
        if not Path(model_dir).is_dir():
            logger.warning("skipping %s — no checkpoint at %s", name, model_dir)
            continue
        for prec in args.precisions.split(","):
            candidates.append((f"{name}_{prec}", model_dir, precision_opts[prec]))
    if not candidates:
        raise SystemExit("no checkpoints found — train one via scripts/distill_qwen_reasoner.py")
    results: dict[str, dict] = {}
    for variant, model_dir, opts in candidates:
        logger.info("loading %s from %s …", variant, model_dir)
        reasoner = DistilledLLMReasoner(
            model_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            name=f"distilled_{variant}",
            **opts,
        )
        results[variant] = measure(reasoner, cases, code_to_name)
        logger.info(
            "%s -> %s",
            variant,
            {
                k: results[variant][k]
                for k in ("size_mb", "grounding", "latency_p95_ms", "edge_gate_pass")
            },
        )
        del reasoner
        torch.cuda.empty_cache()

    summary = {
        "n_test": len(cases),
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "variants": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print("\n=== Narrator base-model architecture sweep ===")
    print(f"cases: {len(cases)}  device: {args.device}  max_new_tokens: {args.max_new_tokens}\n")
    hdr = (
        f"{'variant':20s} {'size MB':>9s} {'p50 ms':>9s} {'p95 ms':>9s} "
        f"{'F1':>6s} {'ground':>7s} {'gen%':>6s} {'gen 95% CI':>14s} {'edge':>5s} {'server':>7s}"
    )
    print(hdr)
    for name, r in results.items():
        lo, hi = r["generation_rate_ci95"]
        print(
            f"{name:20s} {r['size_mb']:>9.1f} {r['latency_p50_ms']:>9.1f} {r['latency_p95_ms']:>9.1f} "
            f"{r['triage_macro_f1']:>6.3f} {r['grounding']:>7.3f} {r['generation_rate']:>6.1%} "
            f"{f'[{lo:.2f},{hi:.2f}]':>14s} "
            f"{'PASS' if r['edge_gate_pass'] else 'FAIL':>5s} "
            f"{'PASS' if r['server_gate_pass'] else 'FAIL':>7s}"
        )
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
