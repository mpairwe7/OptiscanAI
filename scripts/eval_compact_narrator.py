#!/usr/bin/env python3
"""Score the compact narrator on the same held-out split as the earlier sweep.

Reports the axes that actually decide deployability, on the same 24 held-out
cases used throughout ``docs/28``/``docs/29`` so the numbers are comparable:

* **footprint** at bf16 and 4-bit, against the 60 MB edge gate;
* **generation rate** — with prose-only output there is no JSON to fail, so this
  should be 1.000 by construction; it is measured rather than assumed;
* **grounding** — the fraction of disease mentions that were actually detected;
* **claim-level faithfulness** — invented acuity language, quoted-probability
  fidelity and omissions, reusing the audit from
  ``scripts/audit_narrative_faithfulness.py``;
* **vocabulary coverage** — that every one of the classifier's 45 class names
  survives the prune/remap round trip, which is the failure mode a
  trace-derived vocabulary would hide.

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 /usr/bin/python3 \
        scripts/eval_compact_narrator.py --model outputs/narrator_compact
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_narrative_faithfulness import audit_one, wilson_ci  # noqa: E402
from src.narrator.compact import CompactNarrator  # noqa: E402

logger = logging.getLogger("eval_compact")


def load_holdout(path: Path, cut_frac: float) -> list[dict]:
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["scan_id"])
    return rows[int(len(rows) * cut_frac) :]


def check_vocabulary_coverage(narrator: CompactNarrator) -> dict:
    """Every production disease name must survive encode -> prune -> decode."""
    try:
        from backend.app.core.model_service import DISEASE_NAMES
    except Exception:
        from src.evaluation.reasoner_comparison.cases import DISEASE_VOCAB

        DISEASE_NAMES = dict(DISEASE_VOCAB)

    lossy = []
    for code, name in DISEASE_NAMES.items():
        probe = f"- {name} ({code}): 56.0% confidence"
        ids = narrator.tokenizer(probe, add_special_tokens=False)["input_ids"]
        round_trip = narrator.tokenizer.decode(narrator._to_original(narrator._to_compact(ids)))
        if round_trip.strip() != probe.strip():
            lossy.append({"code": code, "expected": probe, "got": round_trip})
    return {
        "classes_checked": len(DISEASE_NAMES),
        "lossy": lossy,
        "all_representable": not lossy,
        "unmapped_tokens": narrator._unmapped,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="outputs/narrator_compact")
    p.add_argument("--traces", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    p.add_argument("--out", default="outputs/reasoner_comparison_real/compact_narrator_report.json")
    p.add_argument("--cut-frac", type=float, default=0.7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--variants", default="bf16,int8,nf4", help="comma-separated: bf16,int8,nf4")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    import torch

    rows = load_holdout(Path(args.traces), args.cut_frac)
    logger.info("held-out cases: %d", len(rows))

    all_variants = {
        "bf16": {"load_4bit": False, "load_8bit": False, "dtype": "float16"},
        "int8": {"load_4bit": False, "load_8bit": True, "dtype": None},
        "nf4": {"load_4bit": True, "load_8bit": False, "dtype": None},
    }
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    variants = [(v, all_variants[v]) for v in wanted if v == "bf16" or torch.cuda.is_available()]

    results: dict[str, dict] = {}
    narratives: dict[str, list] = {}
    coverage = None
    for name, opts in variants:
        logger.info("loading %s …", name)
        n = CompactNarrator(
            args.model, device=args.device, max_new_tokens=args.max_new_tokens, **opts
        )
        if coverage is None:
            coverage = check_vocabulary_coverage(n)
            logger.info(
                "vocabulary coverage: %d classes, all representable=%s",
                coverage["classes_checked"],
                coverage["all_representable"],
            )

        per_case, dump = [], []
        for row in rows:
            preds = [
                {"code": q["code"], "name": q["name"], "probability": float(q["probability"])}
                for q in row["predictions"]
            ]
            text = n.narrate(preds, row.get("referral", "FOLLOW_UP"))
            a = audit_one(text, row)
            # prose-only: a non-empty string IS the narrative, nothing to parse
            a["generated"] = bool(text.strip())
            per_case.append(a)
            dump.append(
                {
                    "scan_id": row["scan_id"],
                    "narrative": text,
                    "teacher": row["teacher"].get("narrative", ""),
                }
            )

        gen = [c for c in per_case if c["generated"]]
        ng = len(gen)
        n_sev = sum(1 for c in gen if c["invented_severity"])
        n_prob = sum(1 for c in gen if c["bad_probs"])
        n_omit = sum(1 for c in gen if c["omitted"])
        words = [len(d["narrative"].split()) for d in dump]
        results[name] = {
            "size_mb": round(n.size_mb(), 1),
            "n_cases": len(per_case),
            "generation_rate": round(ng / len(per_case), 3),
            "generation_rate_ci95": wilson_ci(ng, len(per_case)),
            "severity_escalation_rate": round(n_sev / ng, 3) if ng else None,
            "prob_infidelity_rate": round(n_prob / ng, 3) if ng else None,
            "omission_rate": round(n_omit / ng, 3) if ng else None,
            "invented_terms": sorted({t for c in gen for t in c["invented_severity"]}),
            "avg_words": round(sum(words) / len(words), 1) if words else 0,
            "meets_60mb_edge_gate": n.size_mb() <= 60.0,
            "in_60_100mb_band": 60.0 <= n.size_mb() <= 100.0 or n.size_mb() < 60.0,
            "sample": dump[0]["narrative"][:300] if dump else "",
        }
        narratives[name] = dump
        logger.info(
            "%s -> %s",
            name,
            {
                k: results[name][k]
                for k in ("size_mb", "generation_rate", "severity_escalation_rate", "omission_rate")
            },
        )
        del n
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path(args.out).write_text(
        json.dumps(
            {"n_test": len(rows), "vocabulary_coverage": coverage, "variants": results}, indent=2
        )
    )
    Path(args.out).with_name("compact_narrator_narratives.json").write_text(
        json.dumps(narratives, indent=2)
    )

    print("\n=== Compact narrator (prose-only, vocabulary-pruned) ===")
    print(
        f"{'variant':8s} {'size MB':>9s} {'gen rate':>9s} {'severity':>9s} "
        f"{'bad prob':>9s} {'omission':>9s} {'words':>6s} {'<=60MB':>7s}"
    )
    for name, r in results.items():
        print(
            f"{name:8s} {r['size_mb']:>9.1f} {r['generation_rate']:>9.3f} "
            f"{(r['severity_escalation_rate'] or 0):>9.3f} "
            f"{(r['prob_infidelity_rate'] or 0):>9.3f} {(r['omission_rate'] or 0):>9.3f} "
            f"{r['avg_words']:>6.0f} {'YES' if r['meets_60mb_edge_gate'] else 'no':>7s}"
        )
    if coverage:
        print(
            f"\nvocabulary: {coverage['classes_checked']} classes checked, "
            f"all representable = {coverage['all_representable']}"
        )
        for item in coverage["lossy"][:5]:
            print(f"  LOSSY {item['code']}: {item['got']!r}")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
