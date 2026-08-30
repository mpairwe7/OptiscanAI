"""Run every available reasoner over a case set, score, and write a report.

Produces two artifacts under ``out_dir``:

* ``results.json`` — full metrics for every reasoner (machine-readable),
* ``report.md``    — a side-by-side comparison table + gate verdicts + samples
  (the thing to read before deciding whether to green-light full training).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .interface import Case, Reasoner
from .metrics import evaluate_gates, narrative_metrics, ops_metrics, triage_metrics

logger = logging.getLogger(__name__)


def run_comparison(
    cases: Sequence[Case],
    reasoners: Sequence[Reasoner],
    code_to_name: dict[str, str],
    out_dir: str | Path,
    *,
    mode: str = "smoke",
    teacher_source: str = "synthetic_teacher",
    gates: dict | None = None,
    n_samples: int = 3,
) -> dict:
    if any(c.reference is None for c in cases):
        raise ValueError("scoring needs a reference (teacher) on every case")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = [c.reference for c in cases]

    results: dict[str, dict] = {}
    for r in reasoners:
        logger.info("running reasoner: %s over %d cases", r.name, len(cases))
        preds = [r.reason(c) for c in cases]
        info = r.info()
        triage = triage_metrics(refs, preds)
        narrative = narrative_metrics(cases, preds, code_to_name)
        ops = ops_metrics(info, preds)
        gate = evaluate_gates(triage, narrative, ops, gates)
        results[r.name] = {
            "info": info,
            "triage": triage,
            "narrative": narrative,
            "ops": ops,
            "gate": gate,
            "samples": [
                {
                    "scan_id": cases[i].scan_id,
                    "detected": cases[i].detected_codes,
                    "reference": _out_dict(refs[i]),
                    "prediction": _out_dict(preds[i]),
                }
                for i in range(min(n_samples, len(cases)))
            ],
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "teacher_source": teacher_source,
        "n_cases": len(cases),
        "reasoners": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    report = _render_report(payload)
    (out_dir / "report.md").write_text(report)
    logger.info("wrote %s and %s", out_dir / "results.json", out_dir / "report.md")
    return payload


def _out_dict(o) -> dict:
    return {
        "priority": o.priority,
        "should_explain": o.should_explain,
        "should_review": o.should_review,
        "reasoning": o.reasoning,
        "narrative": o.narrative,
    }


def _render_report(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Reasoner comparison — CNN vs DistilledQwen vs baselines\n")
    lines.append(
        f"- **mode**: `{payload['mode']}`  ·  **teacher/reference**: `{payload['teacher_source']}`  "
        f"·  **cases**: {payload['n_cases']}  ·  **generated**: {payload['generated_at']}\n"
    )
    if payload["mode"] == "smoke":
        lines.append(
            "> ⚠️ **Smoke run** — synthetic cases + a CNN micro-trained for a few epochs on CPU. "
            "Numbers validate the *harness and learnability*, not production quality. "
            "Run `--real` with teacher traces for real numbers.\n"
        )

    # Triage table
    lines.append("## Triage (structured decision)\n")
    lines.append(
        "| reasoner | prio acc | macro F1 | EMERG recall | κ vs teacher | explain F1 | review F1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, r in payload["reasoners"].items():
        t = r["triage"]
        lines.append(
            f"| `{name}` | {t.get('priority_accuracy', 0):.3f} | {t.get('priority_macro_f1', 0):.3f} "
            f"| {t.get('emergency_recall', 0):.3f} | {t.get('cohen_kappa', 0):.3f} "
            f"| {t.get('should_explain_f1', 0):.3f} | {t.get('should_review_f1', 0):.3f} |"
        )

    # Narrative table
    lines.append("\n## Narrative (free-text report)\n")
    lines.append(
        "| reasoner | narrates? | grounding | top-finding coverage | empty rate | avg words |"
    )
    lines.append("|---|:--:|---:|---:|---:|---:|")
    for name, r in payload["reasoners"].items():
        n = r["narrative"]
        narrates = "✅" if r["ops"]["generates_narrative"] else "— (templated)"
        lines.append(
            f"| `{name}` | {narrates} | {n.get('grounding', 0):.3f} | {n.get('top_finding_coverage', 0):.3f} "
            f"| {n.get('empty_rate', 0):.3f} | {n.get('avg_words', 0):.1f} |"
        )

    # Ops + gate table
    lines.append("\n## Ops & deployment gate\n")
    lines.append("| reasoner | size MB | p50 ms | p95 ms | offline | extra deps | gate |")
    lines.append("|---|---:|---:|---:|:--:|---|:--:|")
    for name, r in payload["reasoners"].items():
        o = r["ops"]
        deps = ", ".join(o.get("extra_deps", [])) or "—"
        offline = "✅" if o.get("offline") else "❌"
        gate = "✅ PASS" if r["gate"]["passed"] else "❌ FAIL"
        lines.append(
            f"| `{name}` | {o.get('size_mb', 0):.2f} | {o.get('latency_p50_ms', 0):.2f} "
            f"| {o.get('latency_p95_ms', 0):.2f} | {offline} | {deps} | {gate} |"
        )

    # Gate detail
    lines.append("\n## Gate detail\n")
    for name, r in payload["reasoners"].items():
        checks = r["gate"]["checks"]
        bits = [
            f"{k} {c['value']}{'≥' if c['op'] == '>=' else '≤'}{c['threshold']} {'✓' if c['pass'] else '✗'}"
            for k, c in checks.items()
        ]
        lines.append(f"- `{name}`: " + "; ".join(bits))

    # Samples
    lines.append("\n## Sample outputs (first case)\n")
    for name, r in payload["reasoners"].items():
        if not r["samples"]:
            continue
        s = r["samples"][0]
        p = s["prediction"]
        lines.append(
            f"**`{name}`** → priority=`{p['priority']}` explain={p['should_explain']} "
            f"review={p['should_review']}\n  - narrative: {p['narrative'][:240]}"
        )
    s0 = next(iter(payload["reasoners"].values()))["samples"]
    if s0:
        ref = s0[0]["reference"]
        lines.append(
            f"\n**reference (`{payload['teacher_source']}`)** → priority=`{ref['priority']}` "
            f"explain={ref['should_explain']} review={ref['should_review']}\n  - narrative: {ref['narrative'][:240]}"
        )

    return "\n".join(lines) + "\n"
