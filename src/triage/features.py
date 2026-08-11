"""Canonical structured-feature encoder for triage — the single source of truth.

The teacher derives referral priority from the *classifier's structured output*
(detected diseases, their probabilities, and the classifier's own referral
suggestion), not from pixels. This module encodes exactly that signal as a
fixed-length vector.

It lives in a production package on purpose. Both the training/evaluation
harness (``src.evaluation.reasoner_comparison.features``, which delegates here)
and the serving path (``src.triage.model``) import this one implementation, so
the vector a model was fitted on and the vector it is served cannot drift.
Feature order is stable and published as :data:`FEATURE_NAMES`.

The encoder takes primitives — ``(code, probability)`` pairs and a referral
string — rather than a harness dataclass, so nothing in production has to depend
on the evaluation package.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: Priority labels, in index order. Mirrors ``src/agents/graph.py``.
PRIORITIES: tuple[str, ...] = ("EMERGENCY", "URGENT", "ROUTINE", "FOLLOW_UP")
PRIORITY_INDEX: dict[str, int] = {p: i for i, p in enumerate(PRIORITIES)}

#: Sight-threatening codes that must always escalate, whatever a model predicts.
EMERGENCY_CODES: frozenset[str] = frozenset({"CRAO", "AION"})
#: Codes that warrant specialist review.
CRITICAL_CODES: frozenset[str] = frozenset({"CRAO", "AION", "CRVO", "VH", "RS"})

#: Disease-probability columns, in vector order.
#:
#: This **must** match the vocabulary the model was fitted on, column for column.
#: It is duplicated nowhere: the exported model artifact carries its own
#: ``disease_codes`` and ``feature_names``, and :mod:`src.triage.model` encodes
#: using the artifact's list and refuses to load if the names it derives differ
#: from the ones recorded at export time. This constant is only the default for
#: standalone use.
DISEASE_CODES: tuple[str, ...] = (
    "CRAO",
    "AION",
    "CRVO",
    "VH",
    "RS",
    "DR",
    "ARMD",
    "BRVO",
    "CSR",
    "MH",
    "ERM",
    "DN",
    "MYA",
    "ODC",
    "TSLN",
    "LS",
)

_LOW_CONF = 0.70
_HIGH_CONF = 0.85


def feature_names(
    include_referral: bool = True,
    disease_codes: Sequence[str] = DISEASE_CODES,
) -> list[str]:
    """Column names for :func:`encode`, in vector order."""
    names = [f"p_{code}" for code in disease_codes]
    names += [
        "n_findings",
        "max_prob",
        "mean_prob",
        "has_emergency",
        "has_critical",
        "any_low_conf",
        "n_high_conf",
    ]
    if include_referral:
        names += [f"referral_{p}" for p in PRIORITIES]
    return names


#: Public, stable name list (referral included) — for reports / introspection.
FEATURE_NAMES: list[str] = feature_names(include_referral=True)


def encode(
    findings: Iterable[tuple[str, float]],
    referral_priority: str = "FOLLOW_UP",
    include_referral: bool = True,
    disease_codes: Sequence[str] = DISEASE_CODES,
) -> list[float]:
    """Encode findings + referral as the fixed-length triage feature vector.

    ``findings`` is an iterable of ``(disease_code, probability)``. Duplicate
    codes collapse to their maximum probability, matching how the harness
    encoded training rows.
    """
    pairs: list[tuple[str, float]] = [(str(c), float(p)) for c, p in findings]

    by_code: dict[str, float] = {}
    for code, prob in pairs:
        by_code[code] = max(by_code.get(code, 0.0), prob)

    probs = [p for _, p in pairs]
    codes = {c for c, _ in pairs}

    vec = [by_code.get(code, 0.0) for code in disease_codes]
    vec += [
        float(len(pairs)),
        max(probs, default=0.0),
        (sum(probs) / len(probs)) if probs else 0.0,
        float(any(c in EMERGENCY_CODES for c in codes)),
        float(any(c in CRITICAL_CODES for c in codes)),
        float(any(p < _LOW_CONF for p in probs)),
        float(sum(p >= _HIGH_CONF for p in probs)),
    ]
    if include_referral:
        vec += [float(referral_priority == p) for p in PRIORITIES]
    return vec


def findings_from_predictions(predictions: Sequence[dict]) -> list[tuple[str, float]]:
    """Adapt the pipeline's prediction dicts to :func:`encode` input."""
    return [
        (str(p["code"]), float(p.get("probability", 0.0)))
        for p in predictions
        if p.get("code") is not None
    ]
