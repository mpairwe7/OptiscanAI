"""Voice screening session state management.

Tracks the lifecycle of a single voice-based screening interaction:
CHW speaks -> ASR -> capture image -> gate + inference -> TTS response.
Target: complete in < 90 seconds via voice only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SessionPhase(str, Enum):
    """Phases of a voice screening session."""

    GREETING = "greeting"
    COLLECTING_HISTORY = "collecting_history"
    AWAITING_IMAGE = "awaiting_image"
    PROCESSING = "processing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class VoiceScreeningSession:
    """State for a single voice screening session.

    Attributes
    ----------
    session_id : str
        Unique session identifier.
    language : str
        User's chosen language ("en-ug" | "lg" | "en").
    phase : SessionPhase
        Current phase of the screening workflow.
    patient_context : dict
        Symptoms, history, and demographics collected via voice.
    transcripts : list[str]
        All transcription segments from the session.
    screening_result : dict
        Model inference result (set after processing).
    referral_generated : bool
        Whether a referral recommendation was generated.
    """

    session_id: str = ""
    language: str = "en-ug"
    phase: SessionPhase = SessionPhase.GREETING
    speech_rate: float = 1.0
    barge_in_enabled: bool = True

    # Clinical context from voice
    patient_context: dict = field(default_factory=dict)
    transcripts: list[str] = field(default_factory=list)

    # Screening results
    image_captured: bool = False
    screening_result: Optional[dict] = None
    referral_generated: bool = False

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Metrics
    total_utterances: int = 0
    total_asr_ms: float = 0.0
    total_tts_ms: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def is_complete(self) -> bool:
        return self.phase in (SessionPhase.COMPLETED, SessionPhase.ERROR)

    def add_transcript(self, text: str, asr_ms: float = 0.0) -> None:
        """Add a transcription segment."""
        if text.strip():
            self.transcripts.append(text.strip())
            self.total_utterances += 1
            self.total_asr_ms += asr_ms

    def set_patient_context(self, key: str, value: Any) -> None:
        """Set a patient context field extracted from voice."""
        self.patient_context[key] = value

    def complete(self, result: Optional[dict] = None) -> None:
        """Mark session as completed."""
        self.phase = SessionPhase.COMPLETED
        self.completed_at = time.time()
        if result:
            self.screening_result = result

    def fail(self, reason: str) -> None:
        """Mark session as failed."""
        self.phase = SessionPhase.ERROR
        self.completed_at = time.time()
        self.patient_context["error"] = reason

    def to_audit_dict(self) -> dict:
        """Convert session to audit-friendly dict."""
        return {
            "session_id": self.session_id,
            "language": self.language,
            "phase": self.phase.value,
            "elapsed_seconds": self.elapsed_seconds,
            "total_utterances": self.total_utterances,
            "image_captured": self.image_captured,
            "referral_generated": self.referral_generated,
            "has_result": self.screening_result is not None,
            "total_asr_ms": self.total_asr_ms,
            "total_tts_ms": self.total_tts_ms,
        }
