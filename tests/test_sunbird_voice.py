"""Tests for the Sunbird AI cloud voice tier.

Three properties matter and none is covered elsewhere:

* **the tier stays off unless asked** — it is disabled by default and must make
  no network call when unconfigured, so existing offline deployments are
  untouched by its presence;
* **voice tags never cross languages** — a tag from another locale must fall
  back to the locale's default rather than being forwarded, because the API
  either rejects it or synthesises the wrong language, and both surface to a
  patient as narration in a language they did not ask for;
* **partial transcriptions never leave the device** — a network round-trip per
  500ms chunk would destroy the streaming latency the partial exists for.
"""

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core import sunbird_client as sb
from backend.app.core.asr_engine import ASREngine, _to_wav_bytes
from backend.app.core.config import settings
from backend.app.core.tts_engine import TTSConfig, TTSEngine
from backend.app.core.voice_pipeline import normalize_locale


@pytest.fixture
def sunbird_off():
    """Restore the real (disabled) configuration after a test mutates it."""
    cfg = settings.sunbird
    saved = (cfg.enabled, cfg.api_token, cfg.fallback_api_token)
    yield cfg
    cfg.enabled, cfg.api_token, cfg.fallback_api_token = saved
    sb.reset_clients()


# ── Default state ──


def test_tier_is_disabled_by_default():
    """Shipping this must not change behaviour for anyone who has not opted in."""
    assert settings.sunbird.enabled is False
    assert sb.is_available() is False
    assert sb.account_summary() == "unavailable"


def test_enabled_without_token_is_still_unavailable(sunbird_off):
    """A half-configured tier must not count as available and 500 at call time."""
    sunbird_off.enabled = True
    assert sb.is_available() is False


def test_account_summary_names_roles(sunbird_off):
    """A single account has to be visible as such — failover needs two."""
    sunbird_off.enabled = True
    sunbird_off.api_token = SecretStr("primary-token")
    assert sb.account_summary() == "primary"
    sunbird_off.fallback_api_token = SecretStr("fallback-token")
    assert sb.account_summary() == "primary+fallback"


# ── Voice catalog ──


def test_default_voice_per_locale():
    assert sb.resolve_tts_voice("lg") == "salt_lug_0001"
    assert sb.resolve_tts_voice("ach") == "salt_ach_0001"
    assert sb.resolve_tts_voice("sw") == "waxal_swa_0006"


def test_requested_voice_honoured_when_it_belongs_to_the_locale():
    assert sb.resolve_tts_voice("lg", "waxal_lug_0004") == "waxal_lug_0004"


def test_foreign_voice_tag_falls_back_to_locale_default():
    """An Acholi tag asked for in Luganda must not be forwarded."""
    assert sb.resolve_tts_voice("lg", "salt_ach_0001") == "salt_lug_0001"


def test_unknown_locale_has_no_voice():
    assert sb.resolve_tts_voice("xx") is None
    assert sb.resolve_tts_voice("xx", "salt_lug_0001") is None


def test_every_catalog_default_is_listed_in_its_own_catalog():
    """A default that is not in its locale's catalog could never be re-selected."""
    for locale, default in sb.TTS_VOICES.items():
        assert default in sb.TTS_VOICE_CATALOG[locale], locale


def test_every_voiced_locale_maps_to_a_sunbird_code():
    for locale in sb.TTS_VOICES:
        assert locale in sb.LOCALE_TO_SUNBIRD, locale


# ── Locale normalisation ──


@pytest.mark.parametrize(
    "session_language,expected",
    [("en-ug", "en"), ("en", "en"), ("lg", "lg"), ("", "en"), ("LG", "lg")],
)
def test_normalize_locale(session_language, expected):
    """The session speaks 'en-ug'; the cloud tables key off a bare locale."""
    assert normalize_locale(session_language) == expected


# ── WAV framing ──


def test_wav_framing_is_16k_mono_pcm16():
    audio = np.sin(np.linspace(0, 100, 8000)).astype(np.float32)
    with wave.open(io.BytesIO(_to_wav_bytes(audio)), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        assert w.getnframes() == 8000


def test_wav_framing_clips_instead_of_wrapping():
    """Out-of-range samples must clip; wrapping would arrive as loud noise."""
    audio = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    pcm = np.frombuffer(_to_wav_bytes(audio)[44:], dtype=np.int16)
    assert pcm[0] == 32767
    assert pcm[1] == -32767
    assert abs(pcm[2]) < 10


# ── Engine gating ──


def test_asr_skips_cloud_when_tier_disabled():
    engine = ASREngine()
    engine._model = None
    audio = np.zeros(1600, dtype=np.float32)
    assert engine._transcribe_cloud(audio, is_partial=False, locale="lg") is None


def test_asr_never_sends_partials_to_the_cloud(sunbird_off, monkeypatch):
    """Even fully configured, a partial must not reach the network."""
    sunbird_off.enabled = True
    sunbird_off.api_token = SecretStr("token")

    called = []
    monkeypatch.setattr(sb, "transcribe", lambda *a, **k: called.append(1))

    engine = ASREngine()
    engine._model = None
    audio = np.zeros(1600, dtype=np.float32)
    assert engine._transcribe_cloud(audio, is_partial=True, locale="lg") is None
    assert called == []


def test_asr_keeps_english_local(sunbird_off, monkeypatch):
    """English is served locally; only cloud_locales may reach the network."""
    sunbird_off.enabled = True
    sunbird_off.api_token = SecretStr("token")

    called = []
    monkeypatch.setattr(sb, "transcribe", lambda *a, **k: called.append(1))

    engine = ASREngine()
    engine._model = None
    audio = np.zeros(1600, dtype=np.float32)
    assert engine._transcribe_cloud(audio, is_partial=False, locale="en") is None
    assert called == []


def test_asr_returns_cloud_transcript_for_a_cloud_locale(sunbird_off, monkeypatch):
    sunbird_off.enabled = True
    sunbird_off.api_token = SecretStr("token")
    monkeypatch.setattr(
        sb, "transcribe", lambda *a, **k: {"text": "  Oli otya  ", "language": "lug"}
    )

    engine = ASREngine()
    engine._model = None
    result = engine._transcribe_cloud(np.zeros(1600, dtype=np.float32), False, "lg")
    assert result is not None
    assert result.text == "Oli otya"
    assert result.language == "lug"
    assert result.is_partial is False


def test_asr_treats_empty_cloud_transcript_as_no_answer(sunbird_off, monkeypatch):
    """Blank text must read as 'nothing to add', not as a valid empty transcript."""
    sunbird_off.enabled = True
    sunbird_off.api_token = SecretStr("token")
    monkeypatch.setattr(sb, "transcribe", lambda *a, **k: {"text": "   "})

    engine = ASREngine()
    engine._model = None
    assert engine._transcribe_cloud(np.zeros(1600, dtype=np.float32), False, "lg") is None


def test_tts_skips_cloud_when_tier_disabled():
    engine = TTSEngine()
    assert engine._synthesize_cloud("Oli otya", TTSConfig(locale="lg")) == b""


def test_tts_non_wav_payload_degrades_instead_of_returning_garbage(sunbird_off, monkeypatch):
    """An MP3 body has no decoder in this image; it must read as no audio."""
    sunbird_off.enabled = True
    sunbird_off.api_token = SecretStr("token")
    monkeypatch.setattr(sb, "synthesize", lambda *a, **k: {"audio_url": "https://x/a.mp3"})
    monkeypatch.setattr(sb, "fetch_audio", lambda *a, **k: b"ID3\x04\x00not-a-wav")

    engine = TTSEngine()
    assert engine._synthesize_cloud("Oli otya", TTSConfig(locale="lg")) == b""


def test_tts_returns_pcm_frames_from_a_wav_payload(sunbird_off, monkeypatch):
    sunbird_off.enabled = True
    sunbird_off.api_token = SecretStr("token")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)  # matches TTSEngine's default — no resampling
        w.writeframes(np.arange(500, dtype=np.int16).tobytes())

    monkeypatch.setattr(sb, "synthesize", lambda *a, **k: {"audio_url": "https://x/a.wav"})
    monkeypatch.setattr(sb, "fetch_audio", lambda *a, **k: buf.getvalue())

    engine = TTSEngine()
    frames = engine._synthesize_cloud("Oli otya", TTSConfig(locale="lg"))
    assert len(frames) == 1000  # 500 samples x 2 bytes
    assert np.frombuffer(frames, dtype=np.int16)[10] == 10
