"""Sunbird AI cloud speech + translation for Ugandan languages.

A cloud tier behind the local whisper/piper models: :mod:`asr_engine` and
:mod:`tts_engine` run locally first and call in here only when the local model
is unavailable or returns nothing, so an offline clinic keeps working.

Ported from the URA chatbot's Sunbird integration; the endpoint shapes, voice
catalog tags and failover semantics were established against the live API there
and are kept rather than re-derived. Sync (``httpx.Client``) on purpose — both
engines are sync, and an async client would force a restructure for no gain.

API docs: https://docs.sunbird.ai
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import httpx

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_BACKOFF_BASE_S = 0.5
_RETRY_AFTER_CAP_S = 5.0

# Cached per-token clients — one connection pool per account, not per call.
_clients: dict[str, httpx.Client] = {}


# ── Language and voice tables ──

# RetinalAI locale → Sunbird ISO 639-3 code.
LOCALE_TO_SUNBIRD: dict[str, str] = {
    "en": "eng",
    "en-ug": "eng",  # the platform's default_language
    "lg": "lug",  # Luganda
    "nyn": "nyn",  # Runyankole
    "ach": "ach",  # Acholi
    "sw": "swa",  # Swahili
    "teo": "teo",  # Ateso
    "lgg": "lgg",  # Lugbara
}

# Default catalog tag per locale. `salt_*_0001` is each language's first voice.
TTS_VOICES: dict[str, str] = {
    "lg": "salt_lug_0001",
    "nyn": "salt_nyn_0001",
    "ach": "salt_ach_0001",
    "sw": "waxal_swa_0006",  # the catalog exposes no salt_swa_* tag
    "teo": "salt_teo_0001",
    "en": "salt_eng_0001",  # last resort — local piper serves English first
}

# Every speaker a caller may choose, per locale; the first is the default above.
#
# These are catalog tags for orpheus-3b-tts, NOT the numeric speaker ids that
# address spark-tts — the speech endpoint rejects an id with 400. A tag that is
# wrong or stale does not fail loudly: the request 400s and the chain degrades
# to an English voice reading Luganda, which is the failure this table exists to
# prevent. Re-verify against the live endpoint before adding one.
TTS_VOICE_CATALOG: dict[str, tuple[str, ...]] = {
    "lg": (
        "salt_lug_0001",
        "waxal_lug_0002",
        "waxal_lug_0003",
        "waxal_lug_0004",
        "waxal_lug_0005",
        "waxal_lug_0006",
        "waxal_lug_0007",
        "waxal_lug_0008",
    ),
    "ach": (
        "salt_ach_0001",
        "waxal_ach_0001",
        "waxal_ach_0005",
        "waxal_ach_0006",
        "waxal_ach_0008",
    ),
    "nyn": (
        "salt_nyn_0001",
        "waxal_nyn_0003",
        "waxal_nyn_0004",
        "waxal_nyn_0007",
        "waxal_nyn_0008",
    ),
    "sw": ("waxal_swa_0006", "waxal_swa_0007"),
    "teo": ("salt_teo_0001",),
    "en": ("salt_eng_0001",),
}

# Lugbara (lgg) is in the model's training mix for translation but exposes no
# voice id, so it is deliberately absent from the TTS tables — offering it would
# mean a language you can pick and never hear.

# Languages Sunbird's /tasks/translate serves that this platform routes.
TRANSLATION_LANGUAGES = {"eng", "lug", "nyn", "ach", "teo", "lgg", "swa"}


def resolve_tts_voice(locale: str, voice: str | None = None) -> str | None:
    """The catalog tag to synthesise *locale* with, honouring a requested voice.

    A requested tag is used only when the catalog lists it FOR THAT LOCALE.
    A tag from another language, a stale tag, or a local piper voice name that
    reached here by mistake falls back to the locale's default rather than being
    forwarded — sending a foreign tag either 400s or synthesises the wrong
    language, and both are worse than quietly using the right default.
    """
    default = TTS_VOICES.get(locale)
    if not voice:
        return default
    return voice if voice in TTS_VOICE_CATALOG.get(locale, ()) else default


# ── Accounts ──


def _cfg():
    return settings.sunbird


def _account_tokens() -> list[str]:
    """Configured account tokens in priority order (primary, then fallback)."""
    cfg = _cfg()
    tokens = (cfg.api_token.get_secret_value(), cfg.fallback_api_token.get_secret_value())
    return [t.strip() for t in tokens if t and t.strip()]


def is_available() -> bool:
    """True when the tier is enabled and at least one account is configured."""
    return bool(_cfg().enabled) and bool(_account_tokens())


def account_labels() -> str:
    """Configured accounts named by handle, for server logs.

    Kept out of :func:`account_summary` (and therefore out of health output):
    the roles are what an operator needs there, and an account handle is not
    something to publish on an endpoint just to make a log nicer.
    """
    cfg = _cfg()
    parts = []
    for role, tok, name in (
        ("primary", cfg.api_token.get_secret_value(), cfg.username),
        ("fallback", cfg.fallback_api_token.get_secret_value(), cfg.fallback_username),
    ):
        if tok and tok.strip():
            parts.append(f"{role}={name}" if name else role)
    return ", ".join(parts) or "none"


def account_summary() -> str:
    """Which accounts are configured — never the tokens themselves.

    Surfaced on health output because the failover below only works with two
    accounts and degrades silently with one: ``is_available()`` is True either
    way, so a single-account deployment looks healthy from outside right up
    until that account's daily quota returns 429 and Ugandan-language narration
    falls back to an English voice with nothing to fail over to. Naming the
    roles makes "fallback-only" visible as the misconfiguration it is.
    """
    cfg = _cfg()
    names = [
        name
        for name, tok in (
            ("primary", cfg.api_token.get_secret_value()),
            ("fallback", cfg.fallback_api_token.get_secret_value()),
        )
        if tok and tok.strip()
    ]
    return "+".join(names) if names else "unavailable"


def _client_for(token: str) -> httpx.Client:
    client = _clients.get(token)
    if client is None:
        cfg = _cfg()
        client = httpx.Client(
            base_url=cfg.api_url,
            # Bearer header, never a ?key= query string: an httpx exception
            # renders the URL, which would put the token in logs and traces.
            headers={"Authorization": f"Bearer {token}"},
            timeout=cfg.timeout_s,
        )
        _clients[token] = client
    return client


def reset_clients() -> None:
    """Close and drop cached clients so the next call re-reads configuration."""
    for client in _clients.values():
        try:
            client.close()
        except Exception:  # noqa: BLE001 — teardown must never raise
            pass
    _clients.clear()


# ── Transport ──


def _rewind_uploads(kwargs: dict[str, Any]) -> None:
    """Seek file-like upload bodies back to 0 so a retry re-sends the bytes.

    Without this a retried multipart request silently uploads an already
    consumed (empty) stream.
    """
    for value in (kwargs.get("files") or {}).values():
        stream = value[1] if isinstance(value, tuple) and len(value) > 1 else value
        if hasattr(stream, "seek"):
            try:
                stream.seek(0)
            except Exception:  # noqa: BLE001 — non-seekable; retry sends as-is
                pass


def _retry_delay(attempt: int, retry_after: str = "") -> float:
    """Exponential backoff, honouring a small server Retry-After when given."""
    if retry_after:
        try:
            return min(float(retry_after), _RETRY_AFTER_CAP_S)
        except ValueError:
            pass
    return _BACKOFF_BASE_S * (2 ** (attempt - 1))


def _post(path: str, **kwargs: Any) -> httpx.Response:
    """POST to Sunbird with bounded retry, then account failover.

    Per account: up to ``retries`` attempts with short exponential backoff on
    transport errors and retryable statuses (429/5xx, honouring small
    ``Retry-After`` values). Auth failures (401/403) skip straight to the
    fallback account. Raises the last error when every account is exhausted.
    """
    tokens = _account_tokens()
    if not tokens:
        raise RuntimeError("no Sunbird token configured")
    retries = max(1, _cfg().retries)
    last_exc: Exception | None = None

    for idx, token in enumerate(tokens):
        for attempt in range(1, retries + 1):
            _rewind_uploads(kwargs)
            try:
                resp = _client_for(token).post(path, **kwargs)
                resp.raise_for_status()
                if idx > 0:
                    logger.info("Sunbird fallback account served %s", path)
                return resp
            except httpx.HTTPStatusError as e:
                last_exc = e
                status = e.response.status_code
                if status in (401, 403):
                    logger.warning(
                        "Sunbird auth failed (%d) on account #%d for %s", status, idx + 1, path
                    )
                    break  # retrying the same credentials cannot help
                if status in _RETRYABLE_STATUSES and attempt < retries:
                    delay = _retry_delay(attempt, e.response.headers.get("Retry-After", ""))
                    logger.warning(
                        "Sunbird %d on %s (account #%d, attempt %d); retrying in %.1fs",
                        status,
                        path,
                        idx + 1,
                        attempt,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                break
            except httpx.TimeoutException as e:
                # Deliberately NOT retried on the same account. A timeout means
                # the model is slow right now — usually a cold start — and
                # waiting the identical timeout again doubles the cost for the
                # same answer while the caller's own deadline runs out. Failover
                # to the next account still happens below, so a stuck endpoint
                # gets a second chance; it just does not get two identical waits.
                last_exc = e
                logger.warning(
                    "Sunbird timeout on %s (account #%d) after %ss; not retrying "
                    "this account — failing over",
                    path,
                    idx + 1,
                    _cfg().timeout_s,
                )
                break
            except httpx.HTTPError as e:  # transport errors — a retry can help
                last_exc = e
                if attempt < retries:
                    delay = _retry_delay(attempt)
                    logger.warning(
                        "Sunbird transport error on %s (account #%d, attempt %d): %s; "
                        "retrying in %.1fs",
                        path,
                        idx + 1,
                        attempt,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                break
            except Exception as e:  # noqa: BLE001 — unexpected; fail over
                last_exc = e
                break
        if idx + 1 < len(tokens):
            logger.warning(
                "Sunbird account #%d exhausted for %s (%s); trying fallback",
                idx + 1,
                path,
                last_exc,
            )
    raise last_exc  # type: ignore[misc]


def _error_body(exc: httpx.HTTPStatusError) -> str:
    """Truncated response body for a failed request.

    A bare "400 Bad Request" says nothing about *which* field was rejected; the
    body carries the validation detail and is what makes these diagnosable.
    """
    try:
        return exc.response.text[:400]
    except Exception:  # noqa: BLE001 — body may be unreadable; status still useful
        return ""


# ── Speech-to-text ──


def transcribe(audio_bytes: bytes, locale: str = "en", filename: str = "audio.wav") -> dict | None:
    """Transcribe audio via Sunbird. Returns ``None`` when unavailable/failed.

    Returns ``{"text": str, "language": str}`` so the caller can build its own
    ``TranscriptionResult`` without importing this module's shapes.
    """
    if not is_available():
        return None
    language = LOCALE_TO_SUNBIRD.get(locale, "eng")
    try:
        files = {"audio": (filename, io.BytesIO(audio_bytes), "audio/wav")}
        # /tasks/modal/stt was retired; /tasks/audio/transcriptions replaces it.
        # `language` is REQUIRED here, so it is always sent.
        resp = _post(
            "/tasks/audio/transcriptions", files=files, data={"language": language or "eng"}
        )
        result = resp.json()
        text = result.get("output", {}).get("audio_transcription") or result.get(
            "audio_transcription", ""
        )
        logger.info("Sunbird STT completed (language=%s, chars=%d)", language, len(text))
        return {"text": text, "language": result.get("language", language)}
    except httpx.HTTPStatusError as e:
        logger.warning("Sunbird STT failed (%s): %s | body=%s", language, e, _error_body(e))
        return None
    except Exception as e:  # noqa: BLE001 — cloud tier must never break the caller
        logger.warning("Sunbird STT failed (%s): %s", language, e)
        return None


# ── Text-to-speech ──


def synthesize(text: str, locale: str = "en", voice: str | None = None) -> dict | None:
    """Synthesise *text* with a native Sunbird voice.

    Returns ``{"audio_url", "file_name", "expires_at"}`` — the endpoint hands
    back a signed URL rather than bytes. URLs expire in ~30 minutes, so the
    caller must fetch promptly.
    """
    voice = resolve_tts_voice(locale, voice)
    if not is_available() or not voice:
        return None
    lang_code = LOCALE_TO_SUNBIRD.get(locale)
    try:
        # orpheus-3b-tts is the documented model for this endpoint and the only
        # one that answers. It takes a catalog tag as `voice` and an ISO 639-3
        # `language`; numeric speaker ids belong to spark-tts and are rejected.
        payload: dict[str, Any] = {
            "text": text[:10000],
            "model": "orpheus-3b-tts",
            "voice": voice,
            "response_mode": "url",
        }
        if lang_code:
            payload["language"] = lang_code  # orpheus-only field
        data = _post("/tasks/audio/speech", json=payload).json()
        out = data.get("output", {})
        return {
            "audio_url": out.get("audio_url") or data.get("audio_url"),
            "file_name": data.get("gcs_object") or data.get("file_name"),
            # The response names this `audio_url_expires_at`; the older key is
            # still read so a rollback does not lose the field.
            "expires_at": data.get("audio_url_expires_at") or data.get("expires_at"),
        }
    except httpx.HTTPStatusError as e:
        logger.warning(
            "Sunbird TTS failed (%s, voice %s): %s | body=%s", locale, voice, e, _error_body(e)
        )
        return None
    except Exception as e:  # noqa: BLE001 — cloud tier must never break the caller
        logger.warning("Sunbird TTS failed (%s, voice %s): %s", locale, voice, e)
        return None


def fetch_audio(audio_url: str, timeout_s: float | None = None) -> bytes | None:
    """Download a signed TTS URL into bytes, or ``None`` on failure.

    Separate from :func:`synthesize` because the URL is short-lived: callers
    that only need to hand the URL to a client should not pay for the download.
    """
    try:
        resp = httpx.get(audio_url, timeout=timeout_s or _cfg().timeout_s)
        resp.raise_for_status()
        return resp.content
    except Exception as e:  # noqa: BLE001
        logger.warning("Sunbird audio fetch failed: %s", e)
        return None


# ── Translation ──


def translate(text: str, source_locale: str, target_locale: str) -> str | None:
    """Translate between two locales, or ``None`` when unavailable/unsupported."""
    if not is_available():
        return None
    src = LOCALE_TO_SUNBIRD.get(source_locale)
    tgt = LOCALE_TO_SUNBIRD.get(target_locale)
    if not src or not tgt or src == tgt:
        return None
    if src not in TRANSLATION_LANGUAGES or tgt not in TRANSLATION_LANGUAGES:
        return None
    try:
        data = _post(
            "/tasks/translate",
            json={"text": text, "source_language": src, "target_language": tgt},
        ).json()
        return data.get("output", {}).get("translated_text") or data.get("translated_text")
    except httpx.HTTPStatusError as e:
        logger.warning(
            "Sunbird translate failed (%s->%s): %s | body=%s", src, tgt, e, _error_body(e)
        )
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Sunbird translate failed (%s->%s): %s", src, tgt, e)
        return None
