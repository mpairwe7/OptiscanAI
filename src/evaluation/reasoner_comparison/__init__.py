"""Head-to-head comparison of self-contained replacements for the external LLM
clinical reasoner (triage + narrative) used in the OptiscanAI screening pipeline.

Candidates implement the :class:`~.interface.Reasoner` contract:

* :class:`~.reasoners.RuleReasoner` — deterministic floor (no model),
* :class:`~.reasoners.CNNTriageReasoner` — a small image CNN (the "CNN" option),
* :class:`~.reasoners.DistilledLLMReasoner` — a local distilled small LLM
  (the "DistilledQwen" option),
* :class:`~.reasoners.LLMReasoner` — the production LLM teacher / oracle.

See ``docs/28-reasoner-cnn-vs-distilledqwen.md`` for the design, metrics, and the
cost/feasibility analysis. Run with ``scripts/run_reasoner_comparison.py``.
"""

from __future__ import annotations

from .interface import (
    PRIORITIES,
    Case,
    Prediction,
    Reasoner,
    ReasonerOutput,
)
from .reasoners import (
    CNNTriageReasoner,
    DistilledLLMReasoner,
    LLMReasoner,
    RuleReasoner,
)
from .runner import run_comparison

__all__ = [
    "PRIORITIES",
    "Case",
    "Prediction",
    "Reasoner",
    "ReasonerOutput",
    "RuleReasoner",
    "CNNTriageReasoner",
    "DistilledLLMReasoner",
    "LLMReasoner",
    "run_comparison",
]
