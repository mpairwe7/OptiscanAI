"""Google Gemini LLM layer for RetinalAI agents.

Fallback chain: Gemini → Deterministic rules.

Gemini (Google AI Studio) is the sole hosted provider for clinical reasoning.
Deterministic rules guarantee the system always works without any LLM, and are
what runs whenever ``GEMINI_API_KEY`` is unset or the API errors out.

Configuration is read through :mod:`backend.app.core.config` settings rather
than ``os.environ``. pydantic-settings loads ``.env`` into the ``Settings``
object but does *not* export it into the process environment, so the previous
``os.environ`` reads silently missed every key that was set only in ``.env`` —
the agent graph fell through to deterministic rules even when a key was
configured.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# ── Provider state (lazy-initialized) ──

_gemini_client = None
_gemini_ok: bool | None = None
_active_provider: str = "none"  # gemini | none
_limiter: "_RateLimiter | None" = None


def _cfg():
    """Fetch settings lazily.

    Imported inside the call rather than at module scope: this module is
    imported during package import, and the settings object reads ``.env`` at
    its own construction time.
    """
    from backend.app.core.config import settings

    return settings


def _api_key() -> str:
    """Return the Gemini API key, unwrapped from ``SecretStr``.

    Kept in one place so the raw value never spreads through the module; it is
    passed straight to the SDK client and never logged.
    """
    raw = _cfg().gemini_api_key
    return (raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)).strip()


def _llm_timeout_ms() -> int:
    """Per-request timeout in milliseconds (google-genai takes ms, not seconds).

    Without a timeout a hung provider would block the agent indefinitely; on
    timeout the SDK raises and the Gemini → deterministic fallback kicks in.
    """
    return int(_cfg().llm_timeout_seconds * 1000)


# ── Rate limiting ──


class _RateLimiter:
    """Async sliding-window throttle for the provider's requests-per-minute cap.

    The Gemini free tier rejects anything above ``GEMINI_RPM`` requests per
    rolling minute with a 429. A screening run issues two LLM calls (triage and
    report), so even a modest burst of concurrent scans trips the cap without
    this. Waiters hold the lock while sleeping so the window is enforced
    strictly in arrival order rather than thundering-herd retried.
    """

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._rpm <= 0:  # 0 or negative disables throttling
            return
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self._rpm:
                    self._calls.append(now)
                    return
                wait = 60.0 - (now - self._calls[0])
                logger.debug("Gemini RPM cap reached — waiting %.1fs", wait)
                await asyncio.sleep(wait)


def _output_budget(max_tokens: int) -> int:
    """Total ``max_output_tokens`` to request, including thinking tokens.

    Gemini 3.x always reasons before answering and charges those thinking
    tokens against ``max_output_tokens``; ``thinking_budget=0`` is ignored by
    gemini-3.7-flash. Callers pass the size of the *answer* they want (e.g. 200
    for a triage JSON), so reasoning headroom has to be added on top. Without
    it the model spends the entire budget thinking and returns a truncated
    fragment — measured: a 200-token budget yielded 192 thinking tokens and the
    partial string ``'-level pathologies detected'``.
    """
    s = _cfg()
    return max(max_tokens + s.gemini_thinking_headroom, s.gemini_min_output_tokens)


def get_model() -> str:
    """Return the active model name."""
    if _active_provider == "gemini":
        return _cfg().gemini_model
    return "deterministic_fallback"


def get_provider() -> str:
    """Return which LLM provider is active."""
    _init_providers()
    return _active_provider


def _init_providers():
    """Lazy-initialize the Gemini client."""
    global _gemini_client, _gemini_ok, _active_provider, _limiter

    if _gemini_ok is not None:
        return  # already tried

    key = _api_key()
    if not key:
        _gemini_ok = False
        logger.info("No GEMINI_API_KEY — agents use deterministic fallback")
        return

    try:
        from google import genai
        from google.genai import types

        s = _cfg()
        _gemini_client = genai.Client(
            api_key=key,
            # The SDK sends the key as the x-goog-api-key header, never in the
            # URL query string — an exception carrying the URL would otherwise
            # leak it into logs and traces.
            http_options=types.HttpOptions(timeout=_llm_timeout_ms()),
        )
        _limiter = _RateLimiter(s.gemini_rpm)
        _gemini_ok = True
        _active_provider = "gemini"
        logger.info("Gemini async client ready (model=%s, rpm=%s)", s.gemini_model, s.gemini_rpm)
    except Exception as e:
        logger.warning(f"Gemini init failed: {e}")
        _gemini_ok = False


def reset_providers() -> None:
    """Drop cached client state so the next call re-reads configuration.

    Used by tests, which flip ``GEMINI_API_KEY`` between cases.
    """
    global _gemini_client, _gemini_ok, _active_provider, _limiter
    _gemini_client = None
    _gemini_ok = None
    _active_provider = "none"
    _limiter = None


def is_available() -> bool:
    """Check if a hosted LLM provider is available."""
    _init_providers()
    return _active_provider != "none"


# ── System prompt ──

CLINICAL_SYSTEM_PROMPT = """You are a clinical AI assistant embedded in the RetinalAI screening platform.
You analyze retinal disease predictions from a multi-label GNN classifier (45 diseases) and provide:

1. Clinical interpretation of findings in context
2. Referral priority reasoning (EMERGENCY / URGENT / ROUTINE / FOLLOW_UP)
3. Treatment considerations based on detected disease combinations
4. Risk assessment considering disease co-occurrence patterns

You have access to a clinical knowledge graph with 144 disease relationships calibrated
for Ugandan and East African disease prevalence.

Rules:
- Always state this is AI-assisted screening, not diagnosis
- Flag any sight-threatening conditions immediately
- Consider disease co-occurrence (e.g., DR + HR + CME = complex diabetic eye disease)
- Reference severity levels: 1=mild, 2=moderate, 3=severe
- Be concise and structured — this output feeds directly into clinical reports"""


# ── Unified invoke ──


async def invoke(
    prompt: str,
    system: str = CLINICAL_SYSTEM_PROMPT,
    max_tokens: int = 1024,
    tools: list[dict] | None = None,
) -> dict[str, Any]:
    """Invoke the LLM with automatic fallback: Gemini → deterministic.

    Args:
        max_tokens: size of the *answer* wanted. Reasoning headroom is added on
            top internally — see :func:`_output_budget`.

    Returns:
        {"text": str, "tool_calls": list, "model": str, "provider": str, "fallback": bool}
    """
    _init_providers()

    if _gemini_ok and _gemini_client is not None:
        result = await _invoke_gemini(prompt, system, max_tokens, tools)
        if not result["fallback"]:
            return result
        logger.info("Gemini call failed, falling back to deterministic")

    return {
        "text": "",
        "tool_calls": [],
        "model": "deterministic_fallback",
        "provider": "none",
        "fallback": True,
    }


async def call_llm(
    prompt: str,
    max_tokens: int = 1024,
    system: str = CLINICAL_SYSTEM_PROMPT,
) -> str:
    """Text-in / text-out convenience wrapper around :func:`invoke`.

    Callers that only need the response text (e.g. the voice history extractor)
    use this instead of unpacking the full result dict. Returns an empty string
    when the provider falls through to the deterministic fallback, so callers
    should treat ``""`` as "no LLM output, use your own fallback".
    """
    result = await invoke(prompt, system=system, max_tokens=max_tokens)
    return result.get("text", "")


def _to_gemini_tools(tools: list[dict]) -> list:
    """Translate the agents' tool dicts into Gemini function declarations.

    Callers describe tools as ``{"name", "description", "input_schema"}``; Gemini
    expects the JSON Schema under ``parameters``. Tools already in Gemini shape
    (``parameters`` present) pass through unchanged.
    """
    from google.genai import types

    declarations = []
    for tool in tools:
        schema = tool.get("parameters") or tool.get("input_schema") or {}
        declarations.append(
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=schema,
            )
        )
    return [types.Tool(function_declarations=declarations)]


async def _invoke_gemini(
    prompt: str, system: str, max_tokens: int, tools: list[dict] | None
) -> dict[str, Any]:
    """Call Gemini via the google-genai async API."""
    from google.genai import types

    s = _cfg()
    model = s.gemini_model
    try:
        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": _output_budget(max_tokens),
            "temperature": s.gemini_temperature,
            # Tool declarations are plain schemas, not Python callables, so the
            # SDK has nothing to auto-invoke. Disabling AFC keeps function calls
            # in the response for the caller to execute, and silences the
            # warning google-genai logs on every generate_content() call.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if tools:
            config["tools"] = _to_gemini_tools(tools)

        if _limiter is not None:
            await _limiter.acquire()

        response = await _gemini_client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
        )

        text = (response.text or "").strip()
        tool_calls = [
            {"id": fc.id or fc.name, "name": fc.name, "input": dict(fc.args or {})}
            for fc in (response.function_calls or [])
        ]

        # A budget exhausted by the reasoning pass returns a truncated fragment
        # with finish_reason=MAX_TOKENS. Callers treat any non-fallback text as
        # usable clinical output, so surface that as a fallback rather than
        # letting a half-sentence become the report narrative.
        candidates = response.candidates or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        if finish is not None and str(finish).endswith("MAX_TOKENS") and not tool_calls:
            logger.warning(
                "Gemini response truncated (finish_reason=MAX_TOKENS, %d chars) — "
                "raise gemini_thinking_headroom; discarding partial output",
                len(text),
            )
            return {
                "text": "",
                "tool_calls": [],
                "model": model,
                "provider": "gemini",
                "fallback": True,
                "error": "max_tokens_truncation",
            }

        usage = {}
        if response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "input_tokens": um.prompt_token_count,
                "output_tokens": um.candidates_token_count,
                "thinking_tokens": getattr(um, "thoughts_token_count", None),
            }

        return {
            "text": text,
            "tool_calls": tool_calls,
            "model": response.model_version or model,
            "provider": "gemini",
            "fallback": False,
            "usage": usage,
        }
    except Exception as e:
        logger.warning(f"Gemini API error: {type(e).__name__}: {e}")
        return {
            "text": "",
            "tool_calls": [],
            "model": "",
            "provider": "gemini",
            "fallback": True,
            "error": str(e),
        }
