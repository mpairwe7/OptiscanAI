#!/usr/bin/env python3
"""4-bit quantize the DistilledQwen narrator and measure the gate impact.

Loads the trained narrator (``outputs/distilled_qwen``) twice on the same GPU —
once as-saved (fp baseline) and once with **bitsandbytes NF4** double-quant — and
scores both over the held-out test split: footprint (MB), p50/p95 latency,
narrative grounding, and the deployment gate. NF4 is the reliable, HF-integrated
4-bit path; its kernels are CUDA-only, so this must run on a GPU.

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 /usr/bin/python3 \
        scripts/quantize_distilled_reasoner.py --model outputs/distilled_qwen
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
    SERVER_GATES,
    evaluate_gates,
    narrative_metrics,
    ops_metrics,
    triage_metrics,
)
from src.evaluation.reasoner_comparison.reasoners import DistilledLLMReasoner  # noqa: E402

logger = logging.getLogger("quantize")


def load_test_cases(sft_path: Path, cut_frac: float) -> tuple[list[Case], dict[str, str]]:
    rows = [json.loads(ln) for ln in sft_path.read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["scan_id"])
    code_to_name: dict[str, str] = {}
    cases: list[Case] = []
    for r in rows:
        preds = [
            Prediction(p["code"], p["name"], float(p["probability"]), p.get("confidence", "unknown"))
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


def measure(reasoner: DistilledLLMReasoner, cases: list[Case], code_to_name: dict) -> dict:
    refs = [c.reference for c in cases]
    preds = [reasoner.reason(c) for c in cases]
    triage = triage_metrics(refs, preds)
    narrative = narrative_metrics(cases, preds, code_to_name)
    ops = ops_metrics(reasoner.info(), preds)
    edge_gate = evaluate_gates(triage, narrative, ops)
    server_gate = evaluate_gates(triage, narrative, ops, SERVER_GATES)
    return {
        "size_mb": round(reasoner.size_mb(), 1),
        "triage_macro_f1": triage["priority_macro_f1"],
        "grounding": narrative["grounding"],
        "avg_words": narrative["avg_words"],
        "latency_p50_ms": ops["latency_p50_ms"],
        "latency_p95_ms": ops["latency_p95_ms"],
        "edge_gate_pass": edge_gate["passed"],
        "server_gate_pass": server_gate["passed"],
        "server_gate_checks": server_gate["checks"],
        "sample": preds[0].narrative[:200] if preds else "",
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="outputs/distilled_qwen")
    p.add_argument("--sft", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    p.add_argument("--out", default="outputs/reasoner_comparison_real/quant_report.json")
    p.add_argument("--cut-frac", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible — set CUDA_VISIBLE_DEVICES to a free GPU")

    cases, code_to_name = load_test_cases(Path(args.sft), args.cut_frac)
    logger.info("test cases: %d (device=%s)", len(cases), args.device)

    # Server hosting: bf16-on-GPU is the natural best (smallest full-precision
    # footprint, fastest); fp32 and NF4 4-bit are kept for comparison.
    variants = [
        ("fp32_baseline", {"load_4bit": False, "dtype": None}),
        ("bf16_gpu", {"load_4bit": False, "dtype": "bfloat16"}),
        ("nf4_4bit", {"load_4bit": True, "dtype": None}),
    ]
    results: dict[str, dict] = {}
    for variant, opts in variants:
        logger.info("loading %s …", variant)
        reasoner = DistilledLLMReasoner(
            args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            name=f"distilled_qwen_{variant}",
            **opts,
        )
        results[variant] = measure(reasoner, cases, code_to_name)
        logger.info("%s -> %s", variant, {k: results[variant][k] for k in
                    ("size_mb", "grounding", "latency_p95_ms", "server_gate_pass")})
        del reasoner
        torch.cuda.empty_cache()

    # Best server option: among variants passing the server gate, lowest p95
    # latency (tie-break on smaller footprint).
    passing = {k: v for k, v in results.items() if v["server_gate_pass"]}
    best = min(
        passing or results,
        key=lambda k: (results[k]["latency_p95_ms"], results[k]["size_mb"]),
    )
    summary = {
        "n_test": len(cases),
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "gate_profile": "server (no edge size/latency caps; safety + grounding kept)",
        "variants": results,
        "best_server_option": best,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print("\n=== DistilledQwen narrator on a SERVER (edge size/latency gate removed) ===")
    print(f"cases: {len(cases)}  device: {args.device}  max_new_tokens: {args.max_new_tokens}\n")
    hdr = (f"{'variant':14s} {'size MB':>9s} {'p50 ms':>9s} {'p95 ms':>9s} "
           f"{'grounding':>10s} {'words':>6s} {'server':>7s} {'edge':>5s}")
    print(hdr)
    for name, r in results.items():
        print(f"{name:14s} {r['size_mb']:>9.1f} {r['latency_p50_ms']:>9.1f} {r['latency_p95_ms']:>9.1f} "
              f"{r['grounding']:>10.3f} {r['avg_words']:>6.0f} "
              f"{'PASS' if r['server_gate_pass'] else 'FAIL':>7s} "
              f"{'PASS' if r['edge_gate_pass'] else 'FAIL':>5s}")
    print(f"\nbest server option: {best} "
          f"(size {results[best]['size_mb']:.0f} MB, p95 {results[best]['latency_p95_ms']:.0f} ms, "
          f"grounding {results[best]['grounding']:.3f})")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
