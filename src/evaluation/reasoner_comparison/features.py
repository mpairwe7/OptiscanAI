"""Structured features for lightweight (non-pixel) triage models.

The teacher derives the referral ``priority`` from the *classifier's structured
output* — the detected diseases, their probabilities, and the classifier's own
referral suggestion — not from raw pixels. The image-based ``cnn_triage`` throws
that signal away and has to reconstruct it from pixels, which is why it
underperforms the deterministic rules (see
``docs/28-reasoner-cnn-vs-distilledqwen.md`` §0).

This module turns a :class:`Case` into a small fixed-length feature vector over
exactly that structured signal, so a tiny tabular model (logistic regression,
gradient-boosted trees, a small MLP) can learn the teacher's mapping. Such models
are KB-scale and sub-millisecond, so they clear the size/latency edge gates by a
wide margin — the only open question the sweep answers is accuracy/precision.

Feature order is stable and documented by :data:`FEATURE_NAMES` so a fitted model
and an inference call always agree on the column layout.
"""

from __future__ import annotations

from typing import Any

from src.triage import features as _prod

from .cases import DISEASE_VOCAB
from .interface import Case

#: Disease-probability column order, shared with the serving encoder.
_DISEASE_CODES: list[str] = [code for code, _ in DISEASE_VOCAB]


def feature_names(include_referral: bool = True) -> list[str]:
    """Column names for :func:`case_features`, in vector order."""
    return _prod.feature_names(include_referral, disease_codes=_DISEASE_CODES)


#: Public, stable name list (referral included) — for reports / introspection.
FEATURE_NAMES: list[str] = feature_names(include_referral=True)


def case_features(case: Case, include_referral: bool = True) -> list[float]:
    """Encode a :class:`Case` as a fixed-length structured feature vector.

    With ``include_referral`` the classifier's own referral suggestion is one-hot
    appended — a realistic inference-time input. The no-referral variant isolates
    how much a model can recover from the *findings alone*, which is the more
    honest generalizability probe (the referral passthrough can dominate).

    Delegates to :func:`src.triage.features.encode` — the same code the serving
    path runs — so a model fitted here and served there cannot see different
    columns.
    """
    return _prod.encode(
        [(p.code, float(p.probability)) for p in case.predictions],
        referral_priority=case.referral_priority,
        include_referral=include_referral,
        disease_codes=_DISEASE_CODES,
    )


class PriorityClassifier:
    """Wrap any sklearn-style classifier so ``predict`` returns
    :data:`interface.PRIORITY_INDEX` integers.

    The base estimator is fitted on *dense* 0-based labels (some estimators, e.g.
    XGBoost, require contiguous classes), and predictions are mapped back to the
    canonical priority index. This keeps a fitted model self-contained — a saved
    instance carries its own label mapping, and :class:`reasoners.FeatureTriageReasoner`
    can call ``.predict`` without knowing which priorities were seen in training.
    """

    def __init__(self, base: Any):
        self.base = base
        self.classes_: Any = None  # PRIORITY_INDEX values seen in training, ascending

    def fit(self, X: Any, y: Any) -> "PriorityClassifier":
        import numpy as np

        y = np.asarray([int(v) for v in y])
        self.classes_ = np.array(sorted(set(int(v) for v in y)))
        remap = {c: i for i, c in enumerate(self.classes_)}
        self.base.fit(X, np.array([remap[int(v)] for v in y]))
        return self

    def predict(self, X: Any) -> Any:
        import numpy as np

        dense = np.asarray(self.base.predict(X))
        return np.array([int(self.classes_[int(d)]) for d in dense])
