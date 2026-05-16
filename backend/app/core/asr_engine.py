"""Automatic Speech Recognition engine using Whisper-tiny.

Supports streaming partial transcriptions and final transcription.
Optimized for Ugandan English + Luganda code-switching.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Guard optional dependencies
_whisper_available = False
try:
    from faster_whisper import WhisperModel

    _whisper_available = True
except ImportError:
    pass


@dataclass
class TranscriptionResult:
    """Result of ASR transcription."""

    text: str
    language: str = "en"
    confidence: float = 0.0
    duration_ms: float = 0.0
    is_partial: bool = False
    segments: list[dict] = None

    def __post_init__(self):
        if self.segments is None:
            self.segments = []


class ASREngine:
    """Whisper-tiny ASR engine with streaming support.

    Parameters
    ----------
    model_size : str
        Whisper model size. Default "tiny" for mobile/low-resource.
    model_path : str
        Path to fine-tuned ONNX model. Falls back to HuggingFace download.
    device : str
        Inference device. "cpu" for offline/mobile.
    compute_type : str
        Compute precision. "int8" for efficiency.
    language : str
        Default language. "en" or None for auto-detect.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        model_path: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
    ):
        self._model_size = model_size
        self._model_path = model_path
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model = None
        self._audio_buffer: list[np.ndarray] = []

    def initialize(self) -> None:
        """Load the Whisper model."""
        if not _whisper_available:
            logger.warning(
                "faster-whisper not installed — ASR running in stub mode. "
                "Install with: pip install faster-whisper"
            )
            return

        model_source = self._model_path or self._model_size
        try:
            self._model = WhisperModel(
                model_source,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info(
                "Whisper ASR loaded: model=%s, device=%s, compute=%s",
                model_source,
                self._device,
                self._compute_type,
            )
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)

    def feed_audio(self, audio_chunk: np.ndarray) -> None:
        """Buffer incoming audio chunks for transcription."""
        self._audio_buffer.append(audio_chunk)

    def transcribe_partial(self) -> Optional[TranscriptionResult]:
        """Run partial transcription on buffered audio.

        Returns a partial result every ~500ms of buffered audio.
        """
        if not self._audio_buffer:
            return None

        audio = np.concatenate(self._audio_buffer)
        duration_s = len(audio) / 16000

        # Only attempt partial every 0.5s of audio
        if duration_s < 0.5:
            return None

        return self._transcribe(audio, is_partial=True)

    def transcribe_final(self) -> TranscriptionResult:
        """Run final transcription on all buffered audio and clear buffer."""
        if not self._audio_buffer:
            return TranscriptionResult(text="", is_partial=False)

        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()

        return self._transcribe(audio, is_partial=False)

    def _transcribe(self, audio: np.ndarray, is_partial: bool = False) -> TranscriptionResult:
        """Internal transcription method."""
        t0 = time.perf_counter()

        if self._model is None:
            # Stub mode: return empty
            return TranscriptionResult(
                text="[ASR not available]",
                is_partial=is_partial,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1 if is_partial else 5,
                best_of=1 if is_partial else 3,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )

            text_parts = []
            segment_data = []
            total_confidence = 0.0
            count = 0

            for seg in segments:
                text_parts.append(seg.text.strip())
                segment_data.append(
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                        "avg_logprob": seg.avg_logprob,
                    }
                )
                total_confidence += np.exp(seg.avg_logprob)
                count += 1

            text = " ".join(text_parts)
            avg_confidence = total_confidence / max(count, 1)
            detected_lang = info.language if info else self._language

            return TranscriptionResult(
                text=text,
                language=detected_lang or "en",
                confidence=avg_confidence,
                duration_ms=(time.perf_counter() - t0) * 1000,
                is_partial=is_partial,
                segments=segment_data,
            )

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return TranscriptionResult(
                text="",
                is_partial=is_partial,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

    def reset(self) -> None:
        """Clear the audio buffer for a new utterance."""
        self._audio_buffer.clear()

    @property
    def is_available(self) -> bool:
        return self._model is not None
