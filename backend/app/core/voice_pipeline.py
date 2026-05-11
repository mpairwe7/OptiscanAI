"""Voice processing pipeline orchestrating VAD, ASR, and TTS.

Manages the full voice-to-screening workflow for a single WebSocket
connection. Handles barge-in, language switching, and session state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Optional

from backend.app.core.asr_engine import ASREngine, TranscriptionResult
from backend.app.core.tts_engine import TTSConfig, TTSEngine
from backend.app.core.vad_engine import VADEngine, VADEvent
from backend.app.core.voice_screening_session import (
    SessionPhase,
    VoiceScreeningSession,
)

logger = logging.getLogger(__name__)


class VoiceEvent:
    """Event emitted by the voice pipeline to the WebSocket."""

    def __init__(
        self,
        event_type: str,
        data: Optional[str] = None,
        error: Optional[str] = None,
        **kwargs,
    ):
        self.type = event_type
        self.data = data
        self.error = error
        self.timestamp = time.time()
        self.extra = kwargs

    def to_dict(self) -> dict:
        d = {"type": self.type, "timestamp": self.timestamp}
        if self.data is not None:
            d["data"] = self.data
        if self.error is not None:
            d["error"] = self.error
        d.update(self.extra)
        return d


class VoicePipeline:
    """Orchestrates VAD -> ASR -> processing -> TTS for a single session.

    Parameters
    ----------
    vad : VADEngine
        Voice activity detection engine.
    asr : ASREngine
        Automatic speech recognition engine.
    tts : TTSEngine
        Text-to-speech engine.
    session : VoiceScreeningSession
        Session state tracker.
    """

    def __init__(
        self,
        vad: VADEngine,
        asr: ASREngine,
        tts: TTSEngine,
        session: VoiceScreeningSession,
    ):
        self.vad = vad
        self.asr = asr
        self.tts = tts
        self.session = session
        self._tts_task: Optional[asyncio.Task] = None

    async def process_audio_chunk(self, chunk: bytes) -> list[VoiceEvent]:
        """Process a single audio chunk through VAD -> ASR.

        Returns list of events to send back via WebSocket.
        """
        events: list[VoiceEvent] = []

        # VAD
        import numpy as np

        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        vad_events = self.vad.process_chunk(chunk)

        for ve in vad_events:
            if ve.event_type == "speech_start":
                events.append(VoiceEvent("vad_speech_start"))
                # Barge-in: cancel TTS if speaking
                if self.session.barge_in_enabled and self.tts.is_speaking:
                    self.tts.cancel()
                    events.append(VoiceEvent("response_end", data="barge_in"))

            elif ve.event_type == "speech_end":
                events.append(VoiceEvent("vad_speech_end"))
                # Trigger final transcription
                result = self.asr.transcribe_final()
                if result.text.strip():
                    events.append(VoiceEvent(
                        "transcription",
                        data=result.text,
                        language=result.language,
                        confidence=result.confidence,
                    ))
                    self.session.add_transcript(result.text, result.duration_ms)

        # Feed audio to ASR buffer
        self.asr.feed_audio(audio_np)

        # Emit partial transcriptions periodically
        partial = self.asr.transcribe_partial()
        if partial and partial.text.strip():
            events.append(VoiceEvent(
                "partial_transcription",
                data=partial.text,
            ))

        return events

    async def handle_audio_end(self) -> list[VoiceEvent]:
        """Handle end of audio input — run final transcription."""
        events: list[VoiceEvent] = []

        result = self.asr.transcribe_final()
        if result.text.strip():
            events.append(VoiceEvent(
                "transcription",
                data=result.text,
                language=result.language,
                confidence=result.confidence,
            ))
            self.session.add_transcript(result.text, result.duration_ms)

        return events

    async def generate_response(self, text: str) -> AsyncIterator[VoiceEvent | bytes]:
        """Generate TTS response and stream audio + text events.

        Yields VoiceEvent for control messages and bytes for audio data.
        """
        yield VoiceEvent("response_start")

        tts_config = TTSConfig(
            language=self.session.language,
            speech_rate=self.session.speech_rate,
        )

        # Stream TTS audio
        t0 = time.perf_counter()
        async for audio_chunk in self.tts.synthesize_stream(text, tts_config):
            yield audio_chunk  # Raw audio bytes

        tts_ms = (time.perf_counter() - t0) * 1000
        self.session.total_tts_ms += tts_ms

        yield VoiceEvent("response_end")

    def reset(self) -> None:
        """Reset pipeline state for a new utterance."""
        self.vad.reset()
        self.asr.reset()


# ---------------------------------------------------------------------------
# Factory for creating pipeline instances
# ---------------------------------------------------------------------------

_global_vad: Optional[VADEngine] = None
_global_asr: Optional[ASREngine] = None
_global_tts: Optional[TTSEngine] = None


def init_voice_pipeline(settings: Optional[Any] = None) -> None:
    """Initialize shared voice engines (called once at app startup)."""
    global _global_vad, _global_asr, _global_tts

    from backend.app.core.config import settings as app_settings

    cfg = settings or app_settings.voice_first

    _global_vad = VADEngine(sensitivity=cfg.vad_sensitivity)
    _global_vad.initialize()

    _global_asr = ASREngine(
        model_size=cfg.asr_model.replace("whisper-", ""),
        model_path=cfg.asr_model_path if cfg.asr_model_path else None,
        language="en",
    )
    _global_asr.initialize()

    _global_tts = TTSEngine(
        model_path=cfg.tts_model_path,
    )
    _global_tts.initialize()

    logger.info("Voice pipeline initialized")


def create_pipeline(session: VoiceScreeningSession) -> VoicePipeline:
    """Create a new VoicePipeline for a WebSocket session."""
    if _global_vad is None:
        init_voice_pipeline()

    return VoicePipeline(
        vad=_global_vad,  # type: ignore
        asr=_global_asr,  # type: ignore
        tts=_global_tts,  # type: ignore
        session=session,
    )
