"""Teacher adapter for the self-hosted Qwen vLLM endpoint.

Talks to the OpenAI-compatible ``/v1`` server documented in
``docs/21-vllm-gpu7-awq-optimization.md`` (host port 8011) using only the
standard library, so it adds no dependency. This is the real teacher/oracle for
``--mode real``: it produces the same triage JSON + narrative the production
pipeline would, and is the source of training labels for the CNN / DistilledQwen.

Qwen3 is a reasoning model; we disable its <think> stage via
``chat_template_kwargs.enable_thinking=false`` (passed through by vLLM) so it
returns the requested JSON directly instead of a long reasoning preamble.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .interface import PRIORITY_INDEX, Case, Reasoner, ReasonerOutput
from .reasoners import _extract_json, _rule_triage, template_narrative

logger = logging.getLogger(__name__)


class VLLMReasoner(Reasoner):
    """OpenAI-compatible chat-completions client for the self-hosted Qwen."""

    offline = False  # depends on the local vLLM service being up
    generates_narrative = True
    extra_deps = ("self-hosted vLLM endpoint",)

    def __init__(
        self,
        base_url: str = "http://localhost:8011/v1",
        model: str | None = None,
        name: str = "qwen_teacher",
        timeout: float = 90.0,
        temperature: float = 0.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.timeout = timeout
        self.temperature = temperature
        self.model = model or self._discover_model()
        logger.info("VLLMReasoner ready: %s @ %s", self.model, self.base_url)

    def _discover_model(self) -> str:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=self.timeout) as r:
                data = json.loads(r.read())
            return data["data"][0]["id"]
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"cannot reach vLLM /models at {self.base_url}: {e}") from e

    def _chat(self, prompt: str, max_tokens: int) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                # Qwen3: skip the reasoning stage so we get the JSON/report directly.
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read())
        return payload["choices"][0]["message"]["content"] or ""

    def _reason(self, case: Case) -> ReasonerOutput:
        priority, should_explain, should_review, reasoning = _rule_triage(case)  # fallback seed
        triage_prompt = (
            f"Triage this retinal screening result. {len(case.predictions)} diseases detected.\n"
            f"Referral priority from classifier: {case.referral_priority}\n\n"
            f"Detected diseases:\n{case.disease_summary()}\n\n"
            "Respond with exactly this JSON (no markdown, no commentary):\n"
            '{"priority": "EMERGENCY|URGENT|ROUTINE|FOLLOW_UP", "should_explain": true/false, '
            '"should_review": true/false, "reasoning": "one sentence clinical reasoning"}'
        )
        try:
            t = _extract_json(self._chat(triage_prompt, max_tokens=400))
            if t.get("priority") in PRIORITY_INDEX:
                priority = t["priority"]
            should_explain = bool(t.get("should_explain", should_explain))
            should_review = bool(t.get("should_review", should_review))
            reasoning = t.get("reasoning", reasoning)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("vLLM triage failed for %s (%s) — using rule seed", case.scan_id, e)

        narrative = template_narrative(case, priority, reasoning)
        report_prompt = (
            "Write a concise clinical screening report (3-4 sentences) for this retinal scan.\n\n"
            f"Detected: {', '.join(f'{p.name} ({p.probability:.0%})' for p in case.predictions[:8]) or 'none'}\n"
            f"Referral: {priority}\nTriage reasoning: {reasoning}\n\n"
            "Document findings for the referring ophthalmologist; be specific about disease codes "
            "and probabilities; end with a clear recommendation. Output only the report text."
        )
        try:
            text = self._chat(report_prompt, max_tokens=400).strip()
            if text:
                narrative = text
        except (urllib.error.URLError, KeyError, IndexError) as e:
            logger.warning("vLLM report failed for %s (%s) — using template", case.scan_id, e)

        return ReasonerOutput(
            priority=priority,
            should_explain=should_explain,
            should_review=should_review,
            reasoning=reasoning,
            narrative=narrative,
        )
