"""Lightweight learned triage head — serving path.

Replaces the external LLM for the *structured triage decision*. On real data the
8B teacher's priority equalled the classifier's own referral in 160/160
out-of-sample cases, so this task is deterministic and a 3 KB linear model
reproduces the teacher exactly (accuracy / macro-precision 1.000, held-out,
5-fold CV, and out-of-sample on 160 unseen images). See
``docs/28-reasoner-cnn-vs-distilledqwen.md`` §0.5 and §0.8.

Design notes:

* **No pickle.** The artifact is JSON weights (``models/triage/triage_model.json``)
  and the forward pass is one matrix multiply, so serving needs neither
  scikit-learn nor a pickle load at startup.
* **Self-validating.** The artifact carries the disease vocabulary and the
  feature-name list recorded at export time. Loading rebuilds the names from the
  artifact's own vocabulary and refuses the model if they disagree, so a change
  to the feature encoder cannot silently reorder columns at serve time.
* **Emergency is not learned.** No EMERGENCY case appeared in the training
  sample, so the head cannot emit that class at all. Escalation is enforced by a
  deterministic override here, exactly as the rule baseline does. A learned head
  can never *downgrade* a sight-threatening code.
* **Degrades to rules.** Any load or inference failure returns ``None`` so the
  caller keeps its existing deterministic behaviour rather than failing a scan.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .features import (
    CRITICAL_CODES,
    EMERGENCY_CODES,
    PRIORITIES,
    encode,
    feature_names,
    findings_from_predictions,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/triage/triage_model.json"
_LOW_CONF = 0.70


@dataclass(frozen=True)
class TriageDecision:
    """Structured triage output, shaped like the pipeline's ``triage`` dict."""

    priority: str
    should_explain: bool
    should_review: bool
    reasoning: str
    source: str

    def as_dict(self) -> dict:
        return {
            "priority": self.priority,
            "should_explain": self.should_explain,
            "should_review": self.should_review,
            "reasoning": self.reasoning,
            "source": self.source,
        }


class TriageModel:
    """JSON-weights linear classifier with a deterministic emergency override."""

    def __init__(self, spec: dict):
        if spec.get("format") != "linear-softmax-v1":
            raise ValueError(f"unsupported triage model format: {spec.get('format')!r}")

        self.priorities: list[str] = list(spec["priorities"])
        self.classes: list[int] = [int(c) for c in spec["classes"]]
        self.disease_codes: list[str] = list(spec["disease_codes"])
        self.coef: list[list[float]] = [[float(v) for v in row] for row in spec["coef"]]
        self.intercept: list[float] = [float(v) for v in spec["intercept"]]

        # Guard against silent train/serve skew: rebuild the column names from the
        # artifact's own vocabulary and require them to match what was exported.
        derived = feature_names(include_referral=True, disease_codes=self.disease_codes)
        recorded = list(spec["feature_names"])
        if derived != recorded:
            raise ValueError(
                "triage feature layout changed since export — refusing to serve. "
                f"derived {len(derived)} columns, artifact records {len(recorded)}; "
                f"first difference at index "
                f"{next((i for i, (a, b) in enumerate(zip(derived, recorded)) if a != b), 'n/a')}"
            )
        if len(self.coef[0]) != len(recorded):
            raise ValueError(
                f"weight/feature mismatch: coef has {len(self.coef[0])} columns, "
                f"features {len(recorded)}"
            )

    @classmethod
    def load(cls, path: str | Path) -> "TriageModel":
        return cls(json.loads(Path(path).read_text()))

    def _predict_priority(self, features: Sequence[float]) -> str:
        """argmax(x @ coef.T + intercept), mapped back to a priority label."""
        best_i, best_score = 0, float("-inf")
        for i, (row, b) in enumerate(zip(self.coef, self.intercept)):
            score = b + sum(w * x for w, x in zip(row, features))
            if score > best_score:
                best_i, best_score = i, score
        return self.priorities[self.classes[best_i]]

    def decide(self, predictions: Sequence[dict], referral_priority: str) -> TriageDecision:
        """Triage a case from the classifier's structured output."""
        findings = findings_from_predictions(predictions)
        codes = {c for c, _ in findings}
        probs = [p for _, p in findings]

        has_emergency = bool(codes & EMERGENCY_CODES)
        has_critical = bool(codes & CRITICAL_CODES)

        if has_emergency:
            # Never learned, never overridable: a sight-threatening code escalates.
            priority, source = "EMERGENCY", "triage_model+emergency_override"
        else:
            features = encode(
                findings,
                referral_priority=referral_priority,
                include_referral=True,
                disease_codes=self.disease_codes,
            )
            priority = self._predict_priority(features)
            source = "triage_model"

        return TriageDecision(
            priority=priority,
            should_explain=has_critical or len(findings) >= 3,
            should_review=(
                any(p < _LOW_CONF for p in probs) or len(findings) > 5 or has_emergency
            ),
            reasoning=_reasoning(len(findings), priority, has_critical, has_emergency),
            source=source,
        )


def _reasoning(n: int, priority: str, has_critical: bool, has_emergency: bool) -> str:
    """Mirror of the rule baseline's phrasing, so downstream text is unchanged."""
    if has_emergency:
        return "Sight-threatening arterial occlusion detected — immediate referral required"
    if has_critical:
        return f"Critical pathology detected among {n} findings — specialist review recommended"
    if n > 5:
        return f"Complex multi-disease presentation ({n} findings) — review for co-management"
    if n > 0:
        return f"{n} finding(s) detected at {priority} priority"
    return "No significant pathology detected"


_lock = threading.Lock()
_cached: TriageModel | None = None
_load_attempted = False


def is_enabled() -> bool:
    """``TRIAGE_MODEL_ENABLED`` (default on) gates the learned head."""
    return os.getenv("TRIAGE_MODEL_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def get_model() -> TriageModel | None:
    """Load the model once. Returns ``None`` if disabled or unavailable.

    Never raises: a missing or malformed artifact must degrade to the caller's
    deterministic rules, not fail a screening.
    """
    global _cached, _load_attempted
    if not is_enabled():
        return None
    if _cached is not None or _load_attempted:
        return _cached
    with _lock:
        if _cached is not None or _load_attempted:
            return _cached
        _load_attempted = True
        path = os.getenv("TRIAGE_MODEL_PATH", DEFAULT_MODEL_PATH)
        try:
            _cached = TriageModel.load(path)
            logger.info("triage head loaded from %s (%d features)", path, len(_cached.coef[0]))
        except FileNotFoundError:
            logger.info("no triage head at %s — using deterministic rules", path)
        except Exception as e:  # malformed / incompatible artifact
            logger.warning("triage head at %s unusable (%s) — using deterministic rules", path, e)
        return _cached


def reset_cache() -> None:
    """Drop the cached model (tests, or after swapping the artifact)."""
    global _cached, _load_attempted
    with _lock:
        _cached = None
        _load_attempted = False


__all__ = [
    "PRIORITIES",
    "DEFAULT_MODEL_PATH",
    "TriageDecision",
    "TriageModel",
    "get_model",
    "is_enabled",
    "reset_cache",
]
