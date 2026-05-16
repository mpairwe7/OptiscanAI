"""Voice-first WebSocket endpoint for streaming ASR/TTS.

Implements the protocol defined by the frontend at
frontend/src/hooks/useVoiceWebSocket.ts:

Client sends:
  - binary: audio chunks (WebM/Opus, 250ms intervals)
  - JSON: {type: "config", language: "..."}
  - JSON: {type: "audio_end"}
  - JSON: {type: "image_with_transcript", image: "data:...", transcript: "..."}

Server sends:
  - JSON VoiceEvent: transcription, partial_transcription,
    response_start, response_chunk, response_end, error,
    vad_speech_start, vad_speech_end
  - binary: TTS audio chunks
"""

from __future__ import annotations

import json
import logging
import os
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from backend.app.core.config import settings
from backend.app.core.voice_screening_session import (
    SessionPhase,
    VoiceScreeningSession,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])


def _is_enabled() -> bool:
    """Check if voice-first is enabled."""
    vf = getattr(settings, "voice_first", None)
    if vf is not None and vf.enabled:
        return True
    return os.getenv("VOICE_FIRST__ENABLED", "false").lower() in ("1", "true", "yes")


@router.websocket("/v1/voice/stream")
async def voice_stream(websocket: WebSocket) -> None:
    """Bidirectional voice streaming WebSocket endpoint."""
    await websocket.accept()

    if not _is_enabled():
        await websocket.send_json({
            "type": "error",
            "error": "Voice-first is not enabled. Set VOICE_FIRST__ENABLED=true.",
        })
        await websocket.close(1008)
        return

    session_id = str(uuid4())
    logger.info("Voice session started: %s", session_id)

    # Lazy import of the heavy ASR/TTS pipeline when enabled
    from backend.app.core.voice_pipeline import create_pipeline

    session = VoiceScreeningSession(
        session_id=session_id,
        language=settings.voice_first.default_language,
        speech_rate=settings.voice_first.speech_rate,
        barge_in_enabled=settings.voice_first.barge_in_enabled,
    )
    pipeline = create_pipeline(session)

    try:
        # Send greeting
        await websocket.send_json({
            "type": "response_start",
            "timestamp": session.started_at,
        })
        greeting = _get_greeting(session.language)
        await websocket.send_json({
            "type": "response_chunk",
            "data": greeting,
        })
        await websocket.send_json({"type": "response_end"})

        session.phase = SessionPhase.COLLECTING_HISTORY

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Binary audio data
            if "bytes" in message and message["bytes"]:
                events = await pipeline.process_audio_chunk(message["bytes"])
                for event in events:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(event.to_dict())

            # JSON control messages
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "config":
                    session.language = data.get("language", session.language)
                    session.speech_rate = data.get("speech_rate", session.speech_rate)
                    session.barge_in_enabled = data.get(
                        "barge_in", session.barge_in_enabled
                    )
                    logger.info(
                        "Session %s config: lang=%s rate=%.1f",
                        session_id, session.language, session.speech_rate,
                    )

                elif msg_type == "audio_end":
                    events = await pipeline.handle_audio_end()
                    for event in events:
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_json(event.to_dict())

                    # Generate response based on transcripts
                    if session.transcripts:
                        last_text = session.transcripts[-1]
                        response = _generate_clinical_response(last_text, session)
                        async for item in pipeline.generate_response(response):
                            if websocket.client_state != WebSocketState.CONNECTED:
                                break
                            if isinstance(item, bytes):
                                await websocket.send_bytes(item)
                            else:
                                await websocket.send_json(item.to_dict())

                elif msg_type == "image_with_transcript":
                    session.image_captured = True
                    session.phase = SessionPhase.PROCESSING
                    transcript = data.get("transcript", "")
                    if transcript:
                        session.add_transcript(transcript)

                    # Acknowledge receipt
                    await websocket.send_json({
                        "type": "response_start",
                    })
                    await websocket.send_json({
                        "type": "response_chunk",
                        "data": _get_processing_message(session.language),
                    })

                    # In production: run fundus gate + inference here
                    # For now, acknowledge and let the REST API handle prediction
                    session.phase = SessionPhase.REPORTING
                    await websocket.send_json({
                        "type": "response_chunk",
                        "data": _get_result_message(session.language),
                    })
                    await websocket.send_json({"type": "response_end"})

    except WebSocketDisconnect:
        logger.info("Voice session disconnected: %s", session_id)
    except Exception as e:
        logger.error("Voice session error %s: %s", session_id, e, exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
    finally:
        session.complete()
        pipeline.reset()
        logger.info(
            "Voice session ended: %s (%.1fs, %d utterances)",
            session_id, session.elapsed_seconds, session.total_utterances,
        )


def _get_greeting(language: str) -> str:
    """Get greeting message in the appropriate language."""
    if language == "lg":
        return (
            "Nkulamusizza mu RetinalAI. Njagala okuyamba okukebera amaaso. "
            "Yogera ebikwata ku mulwadde."
        )
    return (
        "Welcome to RetinalAI Clinical Screening. "
        "Please describe any symptoms or patient history."
    )


def _get_processing_message(language: str) -> str:
    if language == "lg":
        return "Nkola ku kifaananyi... Lindako."
    return "Processing the fundus image... Please wait."


def _get_result_message(language: str) -> str:
    if language == "lg":
        return "Ekikebera kikomye. Nsaba otegeere ebivaamu."
    return "Screening complete. Please review the results on screen."


def _generate_clinical_response(transcript: str, session: VoiceScreeningSession) -> str:
    """Generate a clinical response based on the transcript and session state."""
    phase = session.phase

    if phase == SessionPhase.COLLECTING_HISTORY:
        if session.language == "lg":
            return (
                "Weebale. Kati nsaba okwasa ekifaananyi ky'amaaso g'omulwadde. "
                "Kozesa kameera okukwata."
            )
        return (
            "Thank you. Now please capture the patient's fundus image. "
            "Use the camera button to take a photo."
        )

    if phase == SessionPhase.REPORTING:
        if session.language == "lg":
            return "Ebivaamu biri ku sikulini. Nsaba otegeere."
        return "The results are displayed on screen. Please review them."

    # Default
    if session.language == "lg":
        return "Nsaba oyogere ebisingawo oba okwasa ekifaananyi."
    return "Please continue speaking or capture a fundus image."
