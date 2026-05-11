"""Voice Activity Detection engine using Silero VAD.

Processes incoming audio chunks in real-time and emits speech start/end
events. Supports barge-in detection during TTS playback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Guard optional dependency
_silero_available = False
try:
    import torch

    _silero_available = True
except ImportError:
    pass


@dataclass
class VADEvent:
    """Voice activity detection event."""

    event_type: str  # "speech_start" | "speech_end"
    timestamp: float = 0.0
    confidence: float = 0.0
    duration_ms: float = 0.0


class VADEngine:
    """Silero VAD engine with streaming chunk processing.

    Parameters
    ----------
    sensitivity : float
        Detection sensitivity 0.0-1.0. Higher = more sensitive.
    sample_rate : int
        Audio sample rate. Silero expects 16000 Hz.
    min_speech_ms : int
        Minimum speech duration to trigger speech_start.
    min_silence_ms : int
        Minimum silence duration to trigger speech_end.
    """

    def __init__(
        self,
        sensitivity: float = 0.6,
        sample_rate: int = 16000,
        min_speech_ms: int = 250,
        min_silence_ms: int = 700,
    ):
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._threshold = 0.3 + (1.0 - sensitivity) * 0.4  # Map to [0.3, 0.7]

        self._model = None
        self._is_speaking = False
        self._speech_start_time: float = 0.0
        self._silence_start_time: float = 0.0
        self._frames_since_speech = 0

    def initialize(self) -> None:
        """Load Silero VAD model."""
        if not _silero_available:
            logger.warning("torch not available — VAD running in stub mode")
            return

        try:
            model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            logger.info("Silero VAD loaded (threshold=%.2f)", self._threshold)
        except Exception as e:
            logger.warning("Failed to load Silero VAD: %s — using energy-based fallback", e)

    def process_chunk(self, audio_chunk: bytes) -> list[VADEvent]:
        """Process an audio chunk and return any VAD events.

        Parameters
        ----------
        audio_chunk : bytes
            Raw PCM16 audio at self.sample_rate.

        Returns
        -------
        list[VADEvent]
            Speech start/end events detected in this chunk.
        """
        events: list[VADEvent] = []
        now = time.time()

        # Convert bytes to float array
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

        # Get speech probability
        speech_prob = self._get_speech_probability(audio_np)
        is_speech = speech_prob >= self._threshold

        if is_speech and not self._is_speaking:
            # Potential speech start
            if self._speech_start_time == 0:
                self._speech_start_time = now

            elapsed_ms = (now - self._speech_start_time) * 1000
            if elapsed_ms >= self.min_speech_ms:
                self._is_speaking = True
                self._silence_start_time = 0
                events.append(VADEvent(
                    event_type="speech_start",
                    timestamp=now,
                    confidence=speech_prob,
                ))

        elif not is_speech and self._is_speaking:
            # Potential speech end
            if self._silence_start_time == 0:
                self._silence_start_time = now

            elapsed_ms = (now - self._silence_start_time) * 1000
            if elapsed_ms >= self.min_silence_ms:
                duration_ms = (now - self._speech_start_time) * 1000
                self._is_speaking = False
                self._speech_start_time = 0
                self._silence_start_time = 0
                events.append(VADEvent(
                    event_type="speech_end",
                    timestamp=now,
                    confidence=speech_prob,
                    duration_ms=duration_ms,
                ))

        elif is_speech and self._is_speaking:
            # Reset silence counter if speech resumes
            self._silence_start_time = 0

        elif not is_speech and not self._is_speaking:
            # Reset speech start counter
            self._speech_start_time = 0

        return events

    def _get_speech_probability(self, audio: np.ndarray) -> float:
        """Get speech probability from Silero or energy-based fallback."""
        if self._model is not None and _silero_available:
            import torch

            tensor = torch.from_numpy(audio)
            if len(tensor) < 512:
                return 0.0
            # Silero expects specific chunk sizes
            chunk_size = min(len(tensor), 1536)
            prob = self._model(tensor[:chunk_size], self.sample_rate).item()
            return prob

        # Energy-based fallback
        energy = np.sqrt(np.mean(audio ** 2))
        return min(energy * 10, 1.0)

    def reset(self) -> None:
        """Reset VAD state for a new session."""
        self._is_speaking = False
        self._speech_start_time = 0
        self._silence_start_time = 0
        self._frames_since_speech = 0

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
