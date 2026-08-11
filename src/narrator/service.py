"""Process-wide loader for the compact narrator, plus the disclosure guarantee.

Serving contract, in order of preference:

1. the local compact narrator (``src.narrator.compact``) when enabled and loadable;
2. whatever the caller already did (external LLM);
3. the grounded template.

Two things are enforced here rather than left to the model:

**The AI-disclosure statement.** The teacher traces contain no disclosure — 0 of
80 — so every distilled student inherits its absence, and swapping the template
for a generated narrative would silently drop the "AI-assisted, requires
specialist confirmation" sentence from clinical output. EU AI Act transparency
duties for AI-generated content apply from August 2026 and name patient triage
explicitly, so the sentence is appended deterministically. The proper fix is to
regenerate the traces with it mandated (``docs/29`` §3.9); this guarantees it in
the meantime and costs nothing.

**Never failing a screening.** Any load or generation error degrades to the
caller's existing behaviour.

Configuration (``docs/24-environment-variables.md``):

* ``NARRATOR_ENABLED``   — default ``false``; the narrator is opt-in.
* ``NARRATOR_MODEL_PATH``— local checkpoint dir, or a Hub repo id.
* ``NARRATOR_PRECISION`` — ``bf16`` (default/recommended), ``int8`` or ``nf4``.
* ``NARRATOR_DEVICE``    — ``cpu`` (default) or ``cuda:N``.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

DISCLOSURE = "This is an AI-assisted screening result and requires specialist confirmation."

DEFAULT_MODEL_PATH = "models/narrator"
#: Only bf16 is recommended. 4-bit reaches 54 MB but drops findings the teacher
#: reported in 42% of held-out cases while staying fluent; int8 is both larger
#: than 4-bit's band and ~8x slower at batch 1. See docs/29 §3.11.
_PRECISIONS = {
    "bf16": {"dtype": "float16", "load_4bit": False, "load_8bit": False},
    "int8": {"dtype": None, "load_4bit": False, "load_8bit": True},
    "nf4": {"dtype": None, "load_4bit": True, "load_8bit": False},
}

_lock = threading.Lock()
_cached = None
_load_attempted = False


def is_enabled() -> bool:
    """``NARRATOR_ENABLED`` — **off by default**.

    The narrator is opt-in because it has not been clinically reviewed and was
    scored on 24 held-out cases (``docs/30`` feasibility report). The grounded
    template remains the safe default.
    """
    return os.getenv("NARRATOR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def with_disclosure(text: str) -> str:
    """Append the AI-disclosure sentence unless it is already present."""
    text = (text or "").strip()
    if not text:
        return DISCLOSURE
    if "ai-assisted" in text.lower() or "specialist confirmation" in text.lower():
        return text
    if text[-1] not in ".!?":
        text += "."
    return f"{text} {DISCLOSURE}"


def get_narrator():
    """Load the narrator once. Returns ``None`` when disabled or unavailable."""
    global _cached, _load_attempted
    if not is_enabled():
        return None
    if _cached is not None or _load_attempted:
        return _cached
    with _lock:
        if _cached is not None or _load_attempted:
            return _cached
        _load_attempted = True
        path = os.getenv("NARRATOR_MODEL_PATH", DEFAULT_MODEL_PATH)
        precision = os.getenv("NARRATOR_PRECISION", "bf16").strip().lower()
        if precision not in _PRECISIONS:
            logger.warning("unknown NARRATOR_PRECISION %r — using bf16", precision)
            precision = "bf16"
        try:
            from .compact import CompactNarrator

            _cached = CompactNarrator(
                path, device=os.getenv("NARRATOR_DEVICE", "cpu"), **_PRECISIONS[precision]
            )
            logger.info("narrator loaded from %s (%s, %.0f MB)", path, precision, _cached.size_mb())
        except FileNotFoundError:
            logger.info("no narrator at %s — using template", path)
        except Exception as e:
            logger.warning("narrator at %s unusable (%s) — using template", path, e)
        return _cached


def narrate(predictions: list[dict], referral_priority: str) -> str | None:
    """Generate a narrative, disclosure included. ``None`` if unavailable."""
    narrator = get_narrator()
    if narrator is None or not predictions:
        return None
    try:
        return with_disclosure(narrator.narrate(predictions, referral_priority))
    except Exception as e:  # never fail a screening on the narrator
        logger.warning("narrator generation failed (%s) — falling back", e)
        return None


def reset_cache() -> None:
    """Drop the cached narrator (tests, or after swapping the artifact)."""
    global _cached, _load_attempted
    with _lock:
        _cached = None
        _load_attempted = False
