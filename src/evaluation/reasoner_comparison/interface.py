"""Common contracts for the clinical-reasoner comparison harness.

The OptiscanAI screening pipeline currently delegates two jobs to an external
LLM (the aspirational self-hosted ``Qwen/Qwen3-8B-AWQ`` runtime, today stood in
for by Claude -> Groq -> deterministic rules — see ``src/agents/graph.py``):

1. **Triage** — given the classifier's disease predictions, decide a referral
   ``priority`` plus the ``should_explain`` / ``should_review`` routing flags and
   a one-sentence ``reasoning`` string (``triage_node``).
2. **Report** — write a 3-4 sentence free-text clinical ``narrative``
   (``report_node``).

This package lets us swap that reasoner for a self-contained component and score
candidates head-to-head on the *same* task. A :class:`Reasoner` consumes a
:class:`Case` and returns a :class:`ReasonerOutput`; everything else in the
harness (metrics, runner, report) is written against these two types so a CNN, a
distilled small LLM, the rule baseline, and the teacher LLM are all
interchangeable.

The label space mirrors the production code exactly:
``PRIORITIES`` matches ``triage_node`` (the knowledge graph emits
URGENT/ROUTINE/FOLLOW_UP and ``triage_node`` escalates the emergency codes to
EMERGENCY), and ``EMERGENCY_CODES`` / ``CRITICAL_CODES`` mirror the sets in
``src/agents/graph.py``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Referral priority label space — order matters (most -> least urgent).
# Mirrors src/agents/graph.py:triage_node.
PRIORITIES: list[str] = ["EMERGENCY", "URGENT", "ROUTINE", "FOLLOW_UP"]
PRIORITY_INDEX: dict[str, int] = {p: i for i, p in enumerate(PRIORITIES)}

# Sight-threatening codes — mirror src/agents/graph.py:triage_node (lines 148-149).
EMERGENCY_CODES: frozenset[str] = frozenset({"CRAO", "AION"})
CRITICAL_CODES: frozenset[str] = frozenset({"CRAO", "AION", "CRVO", "VH", "RS"})


@dataclass(slots=True)
class Prediction:
    """A single detected disease, matching ``model_service.predict`` items."""

    code: str
    name: str
    probability: float
    confidence: str = "unknown"


@dataclass(slots=True)
class ReasonerOutput:
    """The unified output every reasoner produces.

    ``priority`` / ``should_explain`` / ``should_review`` / ``reasoning`` are the
    structured *triage* sub-task. ``narrative`` is the free-text *report*
    sub-task — a CNN cannot generate one, which is exactly the asymmetry this
    harness is built to quantify, so it defaults to an empty string.
    """

    priority: str
    should_explain: bool
    should_review: bool
    reasoning: str = ""
    narrative: str = ""
    source: str = ""
    latency_ms: float = 0.0
    narrative_generated: bool = False
    """True if ``narrative`` came from the model itself, not a template fallback
    (``source`` is overwritten with the reasoner's own name by ``reason()`` below,
    so it can't carry this)."""

    def __post_init__(self) -> None:
        if self.priority not in PRIORITY_INDEX:
            raise ValueError(
                f"priority {self.priority!r} not in {PRIORITIES}; reasoner '{self.source}'"
            )


@dataclass(slots=True)
class Case:
    """One screening case the reasoner must triage and report on.

    A case carries *both* representations so every reasoner kind can consume it:
    structured ``predictions``/``probabilities`` (rule baseline, LLM, distilled
    LLM) and an optional ``image`` tensor (the CNN). ``reference`` holds the
    teacher's output, used as ground truth for scoring and as the training target
    for the CNN / distilled LLM.
    """

    scan_id: str
    predictions: list[Prediction]
    probabilities: dict[str, float]
    referral_priority: str = "FOLLOW_UP"
    image: Any = None  # torch.Tensor [3,H,W] or path; optional (CNN only)
    history: str = ""
    reference: ReasonerOutput | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def detected_codes(self) -> list[str]:
        return [p.code for p in self.predictions]

    def disease_summary(self, top: int = 10) -> str:
        """Human-readable bullet list of findings — the LLM prompt input."""
        return "\n".join(
            f"- {p.name} ({p.code}): {p.probability:.1%} confidence [{p.confidence}]"
            for p in self.predictions[:top]
        )


class Reasoner(ABC):
    """Base class for every candidate that replaces the external LLM reasoner.

    Subclasses implement :meth:`_reason`; the public :meth:`reason` wraps it with
    latency timing and stamps ``source``/``latency_ms`` so the ops metrics are
    measured uniformly. Capability flags let the runner and report be honest
    about what each candidate can and cannot do (e.g. a CNN cannot narrate).
    """

    #: Stable identifier used in reports and result keys.
    name: str = "reasoner"
    #: Whether the candidate runs with no network / external service.
    offline: bool = True
    #: Whether the candidate emits a genuine free-text narrative.
    generates_narrative: bool = False
    #: Extra Python deps beyond the base project (for the feasibility table).
    extra_deps: tuple[str, ...] = ()

    @abstractmethod
    def _reason(self, case: Case) -> ReasonerOutput:
        """Produce a triage (+ optional narrative) for ``case``."""

    def reason(self, case: Case) -> ReasonerOutput:
        start = time.perf_counter()
        out = self._reason(case)
        out.latency_ms = (time.perf_counter() - start) * 1000.0
        out.source = self.name
        return out

    def size_mb(self) -> float:
        """On-disk / in-memory footprint in MB (0.0 for rule/remote reasoners)."""
        return 0.0

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offline": self.offline,
            "generates_narrative": self.generates_narrative,
            "size_mb": round(self.size_mb(), 3),
            "extra_deps": list(self.extra_deps),
        }
