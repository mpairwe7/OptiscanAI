#!/usr/bin/env python3
"""Claim-level faithfulness audit for distilled narrators.

``metrics.narrative_metrics`` scores ``grounding`` = "every disease *named* in
the narrative was actually detected". That is necessary but far from
sufficient: it is blind to everything the narrative asserts that isn't a
disease name. A narrator can score grounding 1.000 while inventing clinical
severity ("potentially life-threatening"), misquoting probabilities, or
dropping findings the teacher reported.

This audit decomposes each narrative into checkable claims and scores them
against the case's structured predictions **and** the teacher's own narrative
for that same case (the distillation target), reporting per-claim rates with
Wilson 95% CIs. That mirrors current practice for clinical text generation,
where per-claim grounding/faithfulness is scored rather than an answer-level
mean.

Checks per case:
  * ``severity_escalation`` — the narrative uses acuity/severity language the
    teacher did *not* use for this case (hallucinated urgency).
  * ``prob_fidelity``       — every ``NN%`` quoted matches a real predicted
    probability (±1pp).
  * ``omission``            — findings the teacher named that the student dropped.
  * ``disease_grounding``   — the existing lexical check, for reference.

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 /usr/bin/python3 \
        scripts/audit_narrative_faithfulness.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reasoner_comparison.reasoners import DistilledLLMReasoner  # noqa: E402

logger = logging.getLogger("faithfulness")

# Acuity/severity vocabulary. A student using one of these for a case where the
# teacher did not is asserting clinical urgency that its own supervision signal
# never supported — the failure mode lexical grounding cannot see.
SEVERITY_TERMS = [
    "life-threatening",
    "life threatening",
    "blindness",
    "irreversible",
    "permanent vision loss",
    "emergency",
    "immediate",
    "immediately",
    "sight-threatening",
    "acute",
    "rapid",
    "severe",
    "critical",
]


def load_cases(sft_path: Path, cut_frac: float) -> list[dict]:
    rows = [json.loads(ln) for ln in sft_path.read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["scan_id"])
    return rows[int(len(rows) * cut_frac) :]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3))


def audit_one(narrative: str, row: dict) -> dict:
    """Score one narrative against its case's structured facts + teacher text."""
    text = narrative.lower()
    teacher = row["teacher"].get("narrative", "").lower()

    invented = [t for t in SEVERITY_TERMS if t in text and t not in teacher]

    # every NN% quoted must match a real predicted probability (±1pp)
    real = [round(p["probability"] * 100) for p in row["predictions"]]
    quoted = [int(m) for m in re.findall(r"(\d{1,3})\s?%", narrative)]
    bad_probs = [q for q in quoted if not any(abs(q - r) <= 1 for r in real)]

    # findings the teacher named but the student dropped
    teacher_named = [p["name"] for p in row["predictions"] if p["name"].lower() in teacher]
    omitted = [n for n in teacher_named if n.lower() not in text]

    # existing lexical grounding, recomputed here for a like-for-like column
    all_names = [p["name"].lower() for p in row["predictions"]]
    mentioned = [n for n in all_names if n in text]
    ungrounded = [
        n
        for n in re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}", narrative)
        if n.lower() not in all_names and len(n.split()) > 1 and n.lower() in teacher
    ]

    return {
        "invented_severity": invented,
        "bad_probs": bad_probs,
        "quoted_probs": quoted,
        "omitted": omitted,
        "n_mentioned": len(mentioned),
        "n_ungrounded_phrases": len(ungrounded),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sft", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    p.add_argument("--out", default="outputs/reasoner_comparison_real/faithfulness_audit.json")
    p.add_argument("--cut-frac", type=float, default=0.7)
    # See sweep_narrator_architectures.py: 200 truncates ~40% of teacher-length
    # targets mid-JSON and confounds generation rate with verbosity.
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--models",
        default=(
            "qwen0.5b=outputs/distilled_qwen,"
            "smollm2_360m=outputs/distilled_smollm2_360m,"
            "smollm2_135m=outputs/distilled_smollm2_135m"
        ),
    )
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")

    rows = load_cases(Path(args.sft), args.cut_frac)
    logger.info("auditing %d held-out cases", len(rows))

    from src.evaluation.reasoner_comparison.interface import Case, Prediction

    results: dict[str, dict] = {}
    dumps: dict[str, list] = {}
    for pair in args.models.split(","):
        name, _, model_dir = pair.partition("=")
        if not Path(model_dir).is_dir():
            logger.warning("skipping %s (no checkpoint)", name)
            continue
        logger.info("loading %s …", name)
        reasoner = DistilledLLMReasoner(
            model_dir, device=args.device, max_new_tokens=args.max_new_tokens,
            dtype="bfloat16", name=name,
        )
        per_case, dump = [], []
        for row in rows:
            preds = [
                Prediction(q["code"], q["name"], float(q["probability"]), q.get("confidence", "unknown"))
                for q in row["predictions"]
            ]
            case = Case(
                scan_id=row["scan_id"], predictions=preds, probabilities={},
                referral_priority=row.get("referral", "FOLLOW_UP"),
            )
            out = reasoner.reason(case)
            a = audit_one(out.narrative, row)
            a["generated"] = out.narrative_generated
            per_case.append(a)
            dump.append({"scan_id": row["scan_id"], "generated": out.narrative_generated,
                         "narrative": out.narrative, "teacher": row["teacher"].get("narrative", "")})
        n = len(per_case)
        gen = [c for c in per_case if c["generated"]]  # only model-authored text is auditable
        ng = len(gen)
        n_sev = sum(1 for c in gen if c["invented_severity"])
        n_prob = sum(1 for c in gen if c["bad_probs"])
        n_omit = sum(1 for c in gen if c["omitted"])
        results[name] = {
            "n_cases": n,
            "n_generated": ng,
            "generation_rate": round(ng / n, 3),
            "severity_escalation_rate": round(n_sev / ng, 3) if ng else None,
            "severity_escalation_ci95": wilson_ci(n_sev, ng),
            "prob_infidelity_rate": round(n_prob / ng, 3) if ng else None,
            "prob_infidelity_ci95": wilson_ci(n_prob, ng),
            "omission_rate": round(n_omit / ng, 3) if ng else None,
            "omission_ci95": wilson_ci(n_omit, ng),
            "invented_terms": sorted({t for c in gen for t in c["invented_severity"]}),
            "example_bad_probs": [c["bad_probs"] for c in gen if c["bad_probs"]][:3],
        }
        dumps[name] = dump
        logger.info("%s -> %s", name, {k: results[name][k] for k in
                    ("generation_rate", "severity_escalation_rate", "prob_infidelity_rate", "omission_rate")})
        del reasoner
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps({"n_test": len(rows), "variants": results}, indent=2))
    dump_path = Path(args.out).with_name("faithfulness_narratives.json")
    dump_path.write_text(json.dumps(dumps, indent=2))

    print("\n=== Claim-level faithfulness audit (bf16, held-out split) ===")
    print(f"{'variant':14s} {'gen%':>6s} {'severity↑':>10s} {'CI95':>14s} "
          f"{'bad prob%':>10s} {'omission%':>10s}")
    for name, r in results.items():
        lo, hi = r["severity_escalation_ci95"]
        print(f"{name:14s} {r['generation_rate']:>6.1%} "
              f"{(r['severity_escalation_rate'] or 0):>10.1%} {f'[{lo:.2f},{hi:.2f}]':>14s} "
              f"{(r['prob_infidelity_rate'] or 0):>10.1%} {(r['omission_rate'] or 0):>10.1%}")
    for name, r in results.items():
        if r["invented_terms"]:
            print(f"\n{name} invented severity terms (teacher never used them): {r['invented_terms']}")
    print(f"\nReport: {args.out}\nNarratives: {dump_path}")


if __name__ == "__main__":
    main()
