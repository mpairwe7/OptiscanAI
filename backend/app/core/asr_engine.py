"""Automatic Speech Recognition engine using Whisper-tiny.

Supports streaming partial transcriptions and final transcription.
Optimized for Ugandan English + Luganda code-switching.

Local-first with an optional Sunbird AI cloud fallback: whisper runs on every
utterance, and Sunbird is consulted only when the local model is missing or
returns nothing usable, and only for a *final* transcription. Partials stay
local by design — a network round-trip per 500ms chunk would both destroy the
streaming latency the partial exists for and burn the daily quota in minutes.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.app.core.config import settings

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


# Whisper and the voice pipeline both work at 16 kHz mono; Sunbird's
# transcription endpoint takes the same, so no resampling is needed.
_SAMPLE_RATE = 16000


def _to_wav_bytes(audio: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Encode a float32 [-1, 1] mono array as a 16-bit PCM WAV.

    The cloud endpoint takes a file upload, not raw samples. Clipping before
    the int16 cast matters: values slightly outside [-1, 1] (which VAD gain can
    produce) would otherwise wrap around and arrive as loud noise.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


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
        locale: str = "en",
    ):
        self._model_size = model_size
        self._model_path = model_path
        self._device = device
        self._compute_type = compute_type
        self._language = language
        # Drives the cloud tier's language code and whether the cloud is allowed
        # at all (settings.sunbird.cloud_locales). Kept separate from
        # `language`, which is whisper's own hint and uses different codes.
        self._locale = locale
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

    def transcribe_final(self, locale: Optional[str] = None) -> TranscriptionResult:
        """Run final transcription on all buffered audio and clear buffer.

        *locale* selects the cloud tier's language for this utterance. It is a
        per-call argument because one engine instance is shared by every
        concurrent WebSocket session — storing it on the engine would let one
        caller's language leak into another's transcript.
        """
        if not self._audio_buffer:
            return TranscriptionResult(text="", is_partial=False)

        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()

        return self._transcribe(audio, is_partial=False, locale=locale)

    def _transcribe(
        self, audio: np.ndarray, is_partial: bool = False, locale: Optional[str] = None
    ) -> TranscriptionResult:
        """Internal transcription method."""
        t0 = time.perf_counter()

        if self._model is None:
            # No local model. The cloud tier can still serve a final utterance;
            # a partial has nowhere to go, so it stays a stub.
            cloud = self._transcribe_cloud(audio, is_partial, locale)
            if cloud is not None:
                cloud.duration_ms = (time.perf_counter() - t0) * 1000
                return cloud
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

            # Whisper-tiny frequently returns nothing for the Ugandan languages
            # it was not fine-tuned on. An empty final transcript is the signal
            # to escalate — the cloud model is trained on them.
            if not text.strip():
                cloud = self._transcribe_cloud(audio, is_partial, locale)
                if cloud is not None:
                    cloud.duration_ms = (time.perf_counter() - t0) * 1000
                    return cloud

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
            cloud = self._transcribe_cloud(audio, is_partial, locale)
            if cloud is not None:
                cloud.duration_ms = (time.perf_counter() - t0) * 1000
                return cloud
            return TranscriptionResult(
                text="",
                is_partial=is_partial,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

    def _transcribe_cloud(
        self, audio: np.ndarray, is_partial: bool, locale: Optional[str] = None
    ) -> Optional[TranscriptionResult]:
        """Transcribe via Sunbird, or ``None`` when the tier cannot serve this call.

        Returns ``None`` — rather than an empty result — so every caller can tell
        "the cloud had nothing to add" from "the cloud produced a transcript",
        and keep its own local result in the first case.
        """
        # Partials never reach the network: see the module docstring.
        if is_partial:
            return None
        try:
            from backend.app.core import sunbird_client
        except ImportError:  # pragma: no cover — optional tier
            return None
        if not sunbird_client.is_available():
            return None
        # English is served locally; the allowlist keeps a misconfigured locale
        # from silently sending every English utterance to the cloud.
        loc = locale or self._locale
        if loc not in settings.sunbird.cloud_locales:
            return None

        result = sunbird_client.transcribe(_to_wav_bytes(audio), locale=loc)
        if not result or not (result.get("text") or "").strip():
            return None
        logger.info(
            "ASR served by Sunbird cloud (locale=%s, accounts=%s)",
            loc,
            sunbird_client.account_summary(),
        )
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", loc),
            # The endpoint returns no per-segment logprobs, so there is no
            # honest confidence to report. 0.0 would read as "certainly wrong";
            # this is a deliberate neutral marker for "unscored".
            confidence=0.0,
            is_partial=False,
            segments=[],
        )

    def reset(self) -> None:
        """Clear the audio buffer for a new utterance."""
        self._audio_buffer.clear()

    @property
    def is_available(self) -> bool:
        return self._model is not None
