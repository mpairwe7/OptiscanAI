"""Multi-provider LLM layer for RetinalAI agents.

Fallback chain: Claude → Groq → Deterministic rules.

Claude is preferred for medical reasoning (best tool-use, structured output).
Groq provides fast inference with Llama 3.3 70B when Claude credits are exhausted.
Deterministic rules guarantee the system always works without any LLM.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Provider state (lazy-initialized) ──

_claude_client = None
_groq_client = None
_claude_ok: bool | None = None
_groq_ok: bool | None = None
_active_provider: str = "none"  # claude | groq | none


def _get_env(key: str, default: str = "") -> str:
    """Read env lazily — .env loaded by pydantic-settings after module import."""
    return os.environ.get(key, default)


def get_model() -> str:
    """Return the active model name."""
    if _active_provider == "claude":
        return _get_env("AGENT_MODEL", "claude-sonnet-4-20250514")
    if _active_provider == "groq":
        return _get_env("GROQ_MODEL", "llama-3.3-70b-versatile")
    return "deterministic_fallback"


def get_provider() -> str:
    """Return which LLM provider is active."""
    _init_providers()
    return _active_provider


def _init_providers():
    """Lazy-initialize LLM clients. Try Claude first, then Groq."""
    global _claude_client, _groq_client, _claude_ok, _groq_ok, _active_provider

    if _claude_ok is not None or _groq_ok is not None:
        return  # already tried

    # 1. Try Claude
    claude_key = _get_env("ANTHROPIC_API_KEY")
    if claude_key:
        try:
            import anthropic

            _claude_client = anthropic.AsyncAnthropic(api_key=claude_key)
            _claude_ok = True
            _active_provider = "claude"
            logger.info(
                f"Claude async client ready (model={_get_env('AGENT_MODEL', 'claude-sonnet-4-20250514')})"
            )
        except Exception as e:
            logger.warning(f"Claude init failed: {e}")
            _claude_ok = False
    else:
        _claude_ok = False

    # 2. Try Groq as fallback
    groq_key = _get_env("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import AsyncGroq

            _groq_client = AsyncGroq(api_key=groq_key)
            _groq_ok = True
            if _active_provider == "none":
                _active_provider = "groq"
            logger.info(
                f"Groq async client ready (model={_get_env('GROQ_MODEL', 'llama-3.3-70b-versatile')})"
            )
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")
            _groq_ok = False
    else:
        _groq_ok = False

    if _active_provider == "none":
        logger.info("No LLM provider available — agents use deterministic fallback")


def is_available() -> bool:
    """Check if any LLM provider is available."""
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
    """Invoke LLM with automatic fallback: Claude → Groq → deterministic.

    Returns:
        {"text": str, "tool_calls": list, "model": str, "provider": str, "fallback": bool}
    """
    _init_providers()

    # 1. Try Claude
    if _claude_ok and _claude_client is not None:
        result = await _invoke_claude(prompt, system, max_tokens, tools)
        if not result["fallback"]:
            return result
        logger.info("Claude call failed, falling back to Groq")

    # 2. Try Groq
    if _groq_ok and _groq_client is not None:
        result = await _invoke_groq(prompt, system, max_tokens)
        if not result["fallback"]:
            return result
        logger.info("Groq call failed, falling back to deterministic")

    # 3. Deterministic fallback
    return {
        "text": "",
        "tool_calls": [],
        "model": "deterministic_fallback",
        "provider": "none",
        "fallback": True,
    }


async def _invoke_claude(
    prompt: str, system: str, max_tokens: int, tools: list[dict] | None
) -> dict[str, Any]:
    """Call Claude via Anthropic async SDK."""
    try:
        kwargs: dict[str, Any] = {
            "model": _get_env("AGENT_MODEL", "claude-sonnet-4-20250514"),
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            kwargs["tools"] = tools

        response = await _claude_client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        return {
            "text": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "model": response.model,
            "provider": "claude",
            "fallback": False,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
    except Exception as e:
        logger.warning(f"Claude API error: {e}")
        return {
            "text": "",
            "tool_calls": [],
            "model": "",
            "provider": "claude",
            "fallback": True,
            "error": str(e),
        }


async def _invoke_groq(prompt: str, system: str, max_tokens: int) -> dict[str, Any]:
    """Call Groq via async SDK (OpenAI-compatible chat completions)."""
    try:
        model = _get_env("GROQ_MODEL", "llama-3.3-70b-versatile")
        temperature = float(_get_env("GROQ_TEMPERATURE", "0.3"))

        response = await _groq_client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )

        text = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return {
            "text": text,
            "tool_calls": [],  # Groq tool_calls handled separately if needed
            "model": response.model or model,
            "provider": "groq",
            "fallback": False,
            "usage": usage,
        }
    except Exception as e:
        logger.warning(f"Groq API error: {e}")
        return {
            "text": "",
            "tool_calls": [],
            "model": "",
            "provider": "groq",
            "fallback": True,
            "error": str(e),
        }
