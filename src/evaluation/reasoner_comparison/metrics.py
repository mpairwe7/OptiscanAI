"""Scoring for the reasoner comparison.

Three metric families, plus a pass/fail gate:

* **Triage** (structured) — accuracy, macro-F1 and per-class P/R over the 4-way
  ``priority``, the safety-critical **EMERGENCY recall**, Cohen's kappa vs the
  reference, and F1 for the ``should_explain`` / ``should_review`` flags. Every
  candidate (CNN, distilled LLM, rule baseline) is scored here.
* **Narrative** (free text) — grounding/faithfulness (no hallucinated disease
  codes), top-finding coverage, length, and empty-rate. Only narrative-capable
  candidates score meaningfully; a CNN is expected to fail coverage/grounding
  because it cannot narrate — that gap is the headline finding.
* **Ops** — size (MB), p50/p95 latency, offline capability, dependency
  footprint.

Scores are computed against a list of *reference* :class:`ReasonerOutput`
(the teacher), so the same code works whether the teacher is the production LLM
(real mode) or a synthetic stand-in (smoke mode).
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)

from .interface import PRIORITIES, PRIORITY_INDEX, Case, ReasonerOutput


def triage_metrics(refs: Sequence[ReasonerOutput], preds: Sequence[ReasonerOutput]) -> dict:
    """Structured-triage agreement of ``preds`` against ``refs``.

    The macro averages are taken over the labels that actually appear in this
    split (``y_true ∪ y_pred``), not all four priorities, so perfect agreement
    scores 1.0 and an absent class (e.g. no EMERGENCY cases) doesn't silently
    depress the score. The denominator is identical for every reasoner on the
    same reference set, so rankings stay comparable. The per-class table still
    lists all four priorities (with ``support``).
    """
    if len(refs) != len(preds):
        raise ValueError(f"length mismatch: {len(refs)} refs vs {len(preds)} preds")
    if not refs:
        return {}

    y_true = [r.priority for r in refs]
    y_pred = [p.priority for p in preds]

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=PRIORITIES, average=None, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(per_p[i]), 4),
            "recall": round(float(per_r[i]), 4),
            "f1": round(float(per_f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(PRIORITIES)
    }

    observed = sorted(set(y_true) | set(y_pred), key=lambda p: PRIORITY_INDEX[p])
    idx = [PRIORITY_INDEX[label] for label in observed]
    macro_p = sum(per_p[i] for i in idx) / len(idx) if idx else 0.0
    macro_r = sum(per_r[i] for i in idx) / len(idx) if idx else 0.0
    macro_f1 = sum(per_f1[i] for i in idx) / len(idx) if idx else 0.0

    kappa = float(cohen_kappa_score(y_true, y_pred, labels=PRIORITIES))
    if math.isnan(kappa):  # degenerate: a single class observed -> kappa undefined
        kappa = 0.0

    explain_f1 = _binary_f1([r.should_explain for r in refs], [p.should_explain for p in preds])
    review_f1 = _binary_f1([r.should_review for r in refs], [p.should_review for p in preds])

    return {
        "n": len(y_true),
        "priority_accuracy": round(accuracy, 4),
        "priority_macro_precision": round(float(macro_p), 4),
        "priority_macro_recall": round(float(macro_r), 4),
        "priority_macro_f1": round(float(macro_f1), 4),
        "emergency_recall": per_class["EMERGENCY"]["recall"],
        "emergency_support": per_class["EMERGENCY"]["support"],
        "cohen_kappa": round(kappa, 4),
        "should_explain_f1": explain_f1,
        "should_review_f1": review_f1,
        "per_class": per_class,
    }


def _binary_f1(y_true: list[bool], y_pred: list[bool]) -> float:
    return round(
        float(f1_score([int(x) for x in y_true], [int(x) for x in y_pred], zero_division=0)),
        4,
    )


def _mentions(text: str, code: str, name: str) -> bool:
    """True if ``text`` references a disease by code (word-boundary) or name."""
    if not text:
        return False
    low = text.lower()
    if name and name.lower() in low:
        return True
    return re.search(rf"\b{re.escape(code)}\b", text) is not None


def narrative_metrics(
    cases: Sequence[Case], preds: Sequence[ReasonerOutput], code_to_name: dict[str, str]
) -> dict:
    """Faithfulness / coverage of generated narratives.

    * ``grounding`` — of all disease references in the narrative, the fraction
      that correspond to *detected* diseases (1.0 = no hallucinated diseases).
    * ``top_finding_coverage`` — fraction of cases whose highest-probability
      finding is named in the narrative.
    * ``empty_rate`` — fraction of cases with no narrative (a CNN is ~1.0 here).
    """
    if len(cases) != len(preds):
        raise ValueError(f"length mismatch: {len(cases)} cases vs {len(preds)} preds")
    if not cases:
        return {}

    all_codes = list(code_to_name.items())
    grounded_total = 0
    grounded_hits = 0
    covered = 0
    empties = 0
    lengths: list[int] = []

    for case, pred in zip(cases, preds):
        text = pred.narrative or ""
        lengths.append(len(text.split()))
        if not text.strip():
            empties += 1
            continue

        detected = set(case.detected_codes)
        for code, name in all_codes:
            if _mentions(text, code, code_to_name.get(code, "")):
                grounded_total += 1
                if code in detected:
                    grounded_hits += 1
        # also count name-only references not in the vocab map as ungrounded noise
        if case.predictions:
            top = max(case.predictions, key=lambda p: p.probability)
            if _mentions(text, top.code, top.name):
                covered += 1

    # Grounding = precision of disease mentions (no hallucinated diseases). A
    # narrative that claims no diseases (e.g. a no-pathology case) has nothing
    # false to penalise, so it is perfectly grounded; whether it *should* have
    # named a finding is captured separately by top_finding_coverage.
    grounding = grounded_hits / grounded_total if grounded_total else 1.0
    return {
        "n": len(cases),
        "grounding": round(grounding, 4),
        "top_finding_coverage": round(covered / len(cases), 4),
        "empty_rate": round(empties / len(cases), 4),
        "avg_words": round(sum(lengths) / len(lengths), 1),
    }


def ops_metrics(reasoner_info: dict, preds: Sequence[ReasonerOutput]) -> dict:
    """Latency percentiles + static footprint from a reasoner's ``info()``."""
    lat = sorted(p.latency_ms for p in preds)
    return {
        "size_mb": reasoner_info.get("size_mb", 0.0),
        "offline": reasoner_info.get("offline", True),
        "generates_narrative": reasoner_info.get("generates_narrative", False),
        "extra_deps": reasoner_info.get("extra_deps", []),
        "latency_p50_ms": round(_percentile(lat, 50), 3),
        "latency_p95_ms": round(_percentile(lat, 95), 3),
    }


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


# Default deployment gates for an offline / edge reasoner. The emergency-recall
# floor is the safety-critical one: a candidate must never miss more emergencies
# than the teacher.
DEFAULT_GATES = {
    "min_emergency_recall": 1.0,  # must match teacher; missing an emergency is unacceptable
    "min_priority_macro_f1": 0.70,
    "max_size_mb": 60.0,
    "max_latency_p95_ms": 1800.0,  # mid-range Android budget (mirrors mobile target)
    "min_grounding": 0.95,  # narrative must not hallucinate diseases (narrative-capable only)
}

# Server-hosted profile: the component runs in the backend, not on a phone, so the
# artifact-size and mobile-latency budgets do not apply (size/latency become tuning
# concerns, not pass/fail). The safety-critical and quality gates — emergency
# recall, triage macro-F1, narrative grounding — still hold unchanged.
SERVER_GATES = {
    "min_emergency_recall": 1.0,
    "min_priority_macro_f1": 0.70,
    "max_size_mb": float("inf"),
    "max_latency_p95_ms": float("inf"),
    "min_grounding": 0.95,
}


def evaluate_gates(triage: dict, narrative: dict, ops: dict, gates: dict | None = None) -> dict:
    """Apply pass/fail gates; narrative gate is skipped for non-narrating models."""
    g = {**DEFAULT_GATES, **(gates or {})}
    # No emergencies in this split -> recall is undefined; don't fail the gate on it.
    emerg_recall = (
        1.0 if triage.get("emergency_support", 0) == 0 else triage.get("emergency_recall", 0.0)
    )
    checks: dict[str, dict] = {
        "emergency_recall": _check(emerg_recall, ">=", g["min_emergency_recall"]),
        "priority_macro_f1": _check(
            triage.get("priority_macro_f1", 0.0), ">=", g["min_priority_macro_f1"]
        ),
        "size_mb": _check(ops.get("size_mb", 0.0), "<=", g["max_size_mb"]),
        "latency_p95_ms": _check(ops.get("latency_p95_ms", 0.0), "<=", g["max_latency_p95_ms"]),
    }
    if ops.get("generates_narrative"):
        checks["grounding"] = _check(narrative.get("grounding", 0.0), ">=", g["min_grounding"])
    return {"passed": all(c["pass"] for c in checks.values()), "checks": checks}


def _check(value: float, op: str, threshold: float) -> dict:
    passed = value >= threshold if op == ">=" else value <= threshold
    return {"value": round(float(value), 4), "op": op, "threshold": threshold, "pass": bool(passed)}
