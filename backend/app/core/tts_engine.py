"""Text-to-Speech engine using Piper TTS.

Streams audio chunks back through WebSocket. Supports barge-in
(stop mid-utterance when VAD detects speech).
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

_piper_available = False
try:
    import piper

    _piper_available = True
except ImportError:
    pass


@dataclass
class TTSConfig:
    """TTS synthesis configuration."""

    language: str = "en-ug"
    speech_rate: float = 1.0
    sample_rate: int = 22050
    audio_format: str = "pcm16"  # pcm16 | wav
    chunk_size_samples: int = 4096


class TTSEngine:
    """Piper TTS engine with streaming audio output.

    Parameters
    ----------
    model_path : str
        Path to Piper ONNX voice model.
    config_path : str
        Path to Piper voice config JSON.
    sample_rate : int
        Output audio sample rate.
    """

    def __init__(
        self,
        model_path: str = "models/voice/piper-en-ug.onnx",
        config_path: Optional[str] = None,
        sample_rate: int = 22050,
    ):
        self._model_path = model_path
        self._config_path = config_path
        self._sample_rate = sample_rate
        self._voice = None
        self._is_speaking = False
        self._cancelled = False

    def initialize(self) -> None:
        """Load the Piper TTS voice model."""
        if not _piper_available:
            logger.warning(
                "piper-tts not installed — TTS running in stub mode. "
                "Install with: pip install piper-tts"
            )
            return

        model_path = Path(self._model_path)
        if not model_path.exists():
            logger.warning("TTS model not found at %s — stub mode", model_path)
            return

        try:
            self._voice = piper.PiperVoice.load(str(model_path))
            logger.info("Piper TTS loaded: %s", model_path)
        except Exception as e:
            logger.error("Failed to load Piper TTS: %s", e)

    async def synthesize_stream(
        self, text: str, config: Optional[TTSConfig] = None
    ) -> AsyncIterator[bytes]:
        """Synthesize text and yield audio chunks for streaming.

        Supports barge-in: call cancel() to stop mid-utterance.
        """
        cfg = config or TTSConfig()
        self._is_speaking = True
        self._cancelled = False

        try:
            if self._voice is None:
                # Stub: yield silence chunks
                for chunk in self._generate_stub_audio(text, cfg):
                    if self._cancelled:
                        break
                    yield chunk
                    await asyncio.sleep(0.01)
                return

            # Synthesize full audio then stream in chunks
            audio_data = self._synthesize_full(text, cfg)

            chunk_size = cfg.chunk_size_samples * 2  # 2 bytes per int16 sample
            offset = 0

            while offset < len(audio_data) and not self._cancelled:
                end = min(offset + chunk_size, len(audio_data))
                yield audio_data[offset:end]
                offset = end
                # Small delay to simulate streaming and allow barge-in checks
                await asyncio.sleep(
                    len(audio_data[offset - chunk_size : end]) / (self._sample_rate * 2)
                )

        finally:
            self._is_speaking = False

    def _synthesize_full(self, text: str, cfg: TTSConfig) -> bytes:
        """Synthesize text to complete PCM16 audio buffer."""
        if self._voice is None:
            return b""

        buf = io.BytesIO()
        try:
            self._voice.synthesize(
                text,
                buf,
                sentence_silence=0.3,
                length_scale=1.0 / max(cfg.speech_rate, 0.5),
            )
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            return b""

        return buf.getvalue()

    def _generate_stub_audio(self, text: str, cfg: TTSConfig) -> list[bytes]:
        """Generate silent audio chunks as a stub when no TTS model is loaded."""
        # ~100ms of silence per word as approximation
        word_count = max(len(text.split()), 1)
        duration_s = word_count * 0.15 / max(cfg.speech_rate, 0.5)
        total_samples = int(duration_s * cfg.sample_rate)

        chunks = []
        remaining = total_samples
        while remaining > 0:
            chunk_samples = min(remaining, cfg.chunk_size_samples)
            # Near-silent audio (tiny noise to prevent audio device sleep)
            samples = np.random.randint(-10, 10, chunk_samples, dtype=np.int16)
            chunks.append(samples.tobytes())
            remaining -= chunk_samples

        return chunks

    def cancel(self) -> None:
        """Cancel current TTS playback (barge-in)."""
        if self._is_speaking:
            self._cancelled = True
            logger.debug("TTS playback cancelled (barge-in)")

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def is_available(self) -> bool:
        return self._voice is not None
