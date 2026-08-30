"""Concrete reasoner adapters, all behind the :class:`Reasoner` interface.

* :class:`RuleReasoner` — the deterministic floor (mirrors
  ``src/agents/graph.py`` triage fallback + template narrative). Zero added size,
  always available; this is the honest "no LLM at all" baseline.
* :class:`CNNTriageReasoner` — wraps a trained :class:`TriageCNN`. Emits the
  structured triage from the image; reuses the template for narrative text.
* :class:`LLMReasoner` — wraps ``src.agents.llm.invoke`` (Claude/Groq, or a
  self-hosted Qwen endpoint if wired). Used as the teacher / upper bound in real
  mode; needs network, so the smoke path skips it.
* :class:`DistilledLLMReasoner` — wraps a local HuggingFace causal LM (the
  distilled small "Qwen"). Generates both triage and narrative offline. Requires
  ``transformers`` + a checkpoint; constructed lazily and skipped when absent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import re
from typing import Any

from .interface import (
    CRITICAL_CODES,
    EMERGENCY_CODES,
    PRIORITIES,
    PRIORITY_INDEX,
    Case,
    Reasoner,
    ReasonerOutput,
)

logger = logging.getLogger(__name__)


# ── Shared deterministic pieces (mirror src/agents/graph.py) ──


def rule_reasoning(n: int, priority: str, has_critical: bool, has_emergency: bool) -> str:
    if has_emergency:
        return "Sight-threatening arterial occlusion detected — immediate referral required"
    if has_critical:
        return f"Critical pathology detected among {n} findings — specialist review recommended"
    if n > 5:
        return f"Complex multi-disease presentation ({n} findings) — review for co-management"
    if n > 0:
        return f"{n} finding(s) detected at {priority} priority"
    return "No significant pathology detected"


def template_narrative(case: Case, priority: str, reasoning: str) -> str:
    """Mirror src/agents/graph.py:_template_narrative — grounded by construction."""
    if not case.predictions:
        return "AI screening found no significant retinal pathology. Routine follow-up recommended."
    top = max(case.predictions, key=lambda p: p.probability)
    return (
        f"AI screening identified {len(case.predictions)} finding(s). "
        f"Primary finding: {top.name} ({top.code}) at {top.probability:.0%} confidence. "
        f"Referral priority: {priority}. {reasoning} "
        "This is an AI-assisted screening result and requires specialist confirmation."
    )


def _rule_triage(case: Case) -> tuple[str, bool, bool, str]:
    codes = case.detected_codes
    has_emergency = any(c in EMERGENCY_CODES for c in codes)
    has_critical = any(c in CRITICAL_CODES for c in codes)
    n = len(case.predictions)
    low_conf = any(p.probability < 0.70 for p in case.predictions)
    priority = "EMERGENCY" if has_emergency else case.referral_priority
    if priority not in PRIORITY_INDEX:
        priority = "FOLLOW_UP"
    should_explain = has_critical or n >= 3
    should_review = low_conf or n > 5 or has_emergency
    return (
        priority,
        should_explain,
        should_review,
        rule_reasoning(n, priority, has_critical, has_emergency),
    )


def build_distill_prompt(case: Case) -> str:
    """The DistilledQwen prompt — shared by SFT data prep and inference.

    Training and inference MUST use byte-identical prompts, so both the
    ``scripts/build_sft_dataset.py`` label builder and
    :class:`DistilledLLMReasoner` call this one function.
    """
    return (
        f"Retinal screening, {len(case.predictions)} diseases detected. "
        f"Classifier referral: {case.referral_priority}.\nFindings:\n{case.disease_summary()}\n\n"
        'Return JSON: {"priority": "...", "should_explain": bool, "should_review": bool, '
        '"reasoning": "one sentence", "narrative": "3-4 sentence clinical report"}'
    )


# ── Adapters ──


class RuleReasoner(Reasoner):
    name = "rule_baseline"
    offline = True
    generates_narrative = False
    extra_deps = ()

    def _reason(self, case: Case) -> ReasonerOutput:
        priority, should_explain, should_review, reasoning = _rule_triage(case)
        return ReasonerOutput(
            priority=priority,
            should_explain=should_explain,
            should_review=should_review,
            reasoning=reasoning,
            narrative=template_narrative(case, priority, reasoning),
        )


class CNNTriageReasoner(Reasoner):
    """Structured triage from a trained :class:`TriageCNN`; templated narrative."""

    name = "cnn_triage"
    offline = True
    generates_narrative = False  # structured triage only — narrative is templated
    extra_deps = ()  # timm is already a base dependency

    def __init__(self, model: Any, img_size: int = 64, device: str = "cpu"):
        import torch

        from .interface import PRIORITIES

        self._torch = torch
        self._priorities = PRIORITIES
        self.model = model.to(device).eval()
        self.img_size = img_size
        self.device = device

    def size_mb(self) -> float:
        return self.model.size_mb()

    def _reason(self, case: Case) -> ReasonerOutput:
        from .cnn import _to_image_tensor

        x = _to_image_tensor(case.image, self.img_size).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            logit_prio, logit_exp, logit_rev = self.model(x)
        priority = self._priorities[int(logit_prio.argmax(dim=-1).item())]
        should_explain = bool(self._torch.sigmoid(logit_exp).item() >= 0.5)
        should_review = bool(self._torch.sigmoid(logit_rev).item() >= 0.5)
        # Reasoning + narrative reuse the deterministic template (CNN gives the
        # decision; text is templated — the report flags this).
        _, _, _, reasoning = _rule_triage(case)
        return ReasonerOutput(
            priority=priority,
            should_explain=should_explain,
            should_review=should_review,
            reasoning=reasoning,
            narrative=template_narrative(case, priority, reasoning),
        )


class FeatureTriageReasoner(Reasoner):
    """Structured-feature triage from a tiny fitted estimator; templated narrative.

    Consumes the classifier's structured output (detected diseases + probabilities
    + referral) via :func:`features.case_features` rather than pixels, so a
    KB-scale tabular model (logistic regression, gradient-boosted trees, small
    MLP) can learn the teacher's triage mapping. The fitted ``estimator`` predicts
    a :data:`interface.PRIORITY_INDEX` integer; an emergency code always
    escalates to EMERGENCY as a deterministic safety override so a learned model
    can never *downgrade* a sight-threatening finding it under-sampled.

    The narrative reuses the grounded template (this candidate decides triage, not
    prose — the report flags that), matching :class:`CNNTriageReasoner`.
    """

    offline = True
    generates_narrative = False
    extra_deps = ()  # scikit-learn is already a base dependency; lgbm/xgb optional

    def __init__(
        self,
        estimator: Any,
        name: str = "feature_triage",
        include_referral: bool = True,
        emergency_override: bool = True,
        extra_deps: tuple[str, ...] = (),
    ):
        self.name = name
        self.estimator = estimator
        self.include_referral = include_referral
        self.emergency_override = emergency_override
        self.extra_deps = extra_deps
        self._size_mb = len(pickle.dumps(estimator)) / 1e6

    def size_mb(self) -> float:
        return self._size_mb

    def _reason(self, case: Case) -> ReasonerOutput:
        from .features import case_features

        feats = case_features(case, include_referral=self.include_referral)
        idx = int(self.estimator.predict([feats])[0])
        priority = PRIORITIES[idx]
        if self.emergency_override and any(c in EMERGENCY_CODES for c in case.detected_codes):
            priority = "EMERGENCY"
        # Flags/reasoning/narrative reuse the deterministic pieces (this candidate
        # owns the priority decision; text is templated — the report flags this).
        _, should_explain, should_review, reasoning = _rule_triage(case)
        return ReasonerOutput(
            priority=priority,
            should_explain=should_explain,
            should_review=should_review,
            reasoning=reasoning,
            narrative=template_narrative(case, priority, reasoning),
        )


def _run_sync(coro):
    """Run an async coroutine from sync code (fresh loop, benchmark-safe)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _extract_json(text: str) -> dict:
    """Best-effort JSON parse of an LLM triage reply (mirrors graph.py handling)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].replace("json", "", 1).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = json.loads(match.group(0) if match else text)
    if not isinstance(parsed, dict):
        # a small/degraded model can emit valid JSON that isn't an object (e.g. a
        # bare quoted string) — treat that the same as a parse failure so callers
        # fall back to the rule-seeded triage instead of crashing on `.get`.
        raise json.JSONDecodeError("parsed JSON is not an object", text, 0)
    return parsed


class LLMReasoner(Reasoner):
    """Wraps the production LLM layer (``src.agents.llm``) as the teacher / oracle."""

    offline = False
    generates_narrative = True
    extra_deps = ("anthropic|groq (provider SDK)",)

    def __init__(self, name: str = "llm_teacher"):
        self.name = name
        from src.agents import llm

        self._llm = llm
        if not llm.is_available():
            raise RuntimeError("no LLM provider available (set ANTHROPIC_API_KEY/GROQ_API_KEY)")

    def _reason(self, case: Case) -> ReasonerOutput:
        priority, should_explain, should_review, reasoning = _rule_triage(case)  # fallback seed
        triage_prompt = (
            f"Triage this retinal screening result. {len(case.predictions)} diseases detected.\n"
            f"Referral priority from classifier: {case.referral_priority}\n\n"
            f"Detected diseases:\n{case.disease_summary()}\n\n"
            "Respond with exactly this JSON (no markdown):\n"
            '{"priority": "EMERGENCY|URGENT|ROUTINE|FOLLOW_UP", "should_explain": true/false, '
            '"should_review": true/false, "reasoning": "one sentence clinical reasoning"}'
        )
        resp = _run_sync(self._llm.invoke(triage_prompt, max_tokens=200))
        if not resp["fallback"]:
            try:
                t = _extract_json(resp["text"])
                priority = t.get("priority", priority)
                if priority not in PRIORITY_INDEX:
                    priority = "FOLLOW_UP"
                should_explain = bool(t.get("should_explain", should_explain))
                should_review = bool(t.get("should_review", should_review))
                reasoning = t.get("reasoning", reasoning)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning("LLM triage parse failed (%s) — using rule seed", e)

        narrative = template_narrative(case, priority, reasoning)
        report_prompt = (
            "Write a concise clinical screening report (3-4 sentences) for this retinal scan.\n\n"
            f"Detected: {', '.join(f'{p.name} ({p.probability:.0%})' for p in case.predictions[:8])}\n"
            f"Referral: {priority}\nTriage reasoning: {reasoning}\n\n"
            "Document findings for the referring ophthalmologist; be specific about disease codes "
            "and probabilities; end with a clear recommendation."
        )
        rep = _run_sync(self._llm.invoke(report_prompt, max_tokens=300))
        if not rep["fallback"] and rep["text"].strip():
            narrative = rep["text"].strip()

        return ReasonerOutput(
            priority=priority,
            should_explain=should_explain,
            should_review=should_review,
            reasoning=reasoning,
            narrative=narrative,
        )


class DistilledLLMReasoner(Reasoner):
    """Local distilled small-LLM ("DistilledQwen") via HuggingFace transformers."""

    name = "distilled_qwen"
    offline = True
    generates_narrative = True
    extra_deps = ("transformers", "accelerate")

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        max_new_tokens: int = 256,
        load_4bit: bool = False,
        dtype: str | None = None,
        name: str | None = None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "DistilledLLMReasoner needs `transformers` (and a trained checkpoint). "
                "Install the distill extra and train via scripts/ before enabling."
            ) from e

        import torch

        if name:
            self.name = name
        self._torch = torch
        self.load_4bit = load_4bit
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # eager attention avoids the Triton/flash path (some hosts lack the
        # toolchain Triton needs to JIT-compile its CUDA kernels).
        kwargs: dict[str, Any] = {"attn_implementation": "eager"}
        if load_4bit:
            # bitsandbytes NF4 + double-quant: the reliable, HF-integrated 4-bit
            # path. Weights are packed to ~0.5 byte/param; kernels are CUDA-only,
            # so a 4-bit load requires a GPU at inference.
            from transformers import BitsAndBytesConfig

            if device == "cpu" or not torch.cuda.is_available():
                raise RuntimeError("load_4bit requires a CUDA device (bitsandbytes kernels)")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = {"": device}
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).eval()
            self.extra_deps = ("transformers", "accelerate", "bitsandbytes")
        else:
            if dtype:
                kwargs["torch_dtype"] = getattr(torch, dtype)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_dir = model_dir

    def size_mb(self) -> float:
        total = 0
        for p in self.model.parameters():
            # bitsandbytes Params4bit store packed uint8; numel*element_size on the
            # packed storage already reflects the true 4-bit footprint.
            total += p.numel() * p.element_size()
        for b in self.model.buffers():
            total += b.numel() * b.element_size()
        return total / 1e6

    def _generate(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def _reason(self, case: Case) -> ReasonerOutput:
        priority, should_explain, should_review, reasoning = _rule_triage(case)  # fallback seed
        prompt = build_distill_prompt(case)
        narrative_generated = False
        try:
            t = _extract_json(self._generate(prompt))
            priority = (
                t.get("priority", priority) if t.get("priority") in PRIORITY_INDEX else priority
            )
            should_explain = bool(t.get("should_explain", should_explain))
            should_review = bool(t.get("should_review", should_review))
            reasoning = t.get("reasoning", reasoning)
            generated_narrative = t.get("narrative")
            narrative = generated_narrative or template_narrative(case, priority, reasoning)
            narrative_generated = bool(generated_narrative)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("distilled LLM parse failed (%s) — using rule seed", e)
            narrative = template_narrative(case, priority, reasoning)

        return ReasonerOutput(
            priority=priority,
            should_explain=should_explain,
            should_review=should_review,
            reasoning=reasoning,
            narrative=narrative,
            narrative_generated=narrative_generated,
        )


class PrecomputedReasoner(Reasoner):
    """Replays stored teacher outputs as a comparison row (the reference).

    Triage scores are trivially perfect (it generated the reference), so its
    value in the report is the *narrative* metrics: the real teacher text's
    grounding / coverage / length, which the template and DistilledQwen are
    measured against. Keyed by ``scan_id``.
    """

    offline = False
    generates_narrative = True
    extra_deps = ("self-hosted vLLM endpoint",)

    def __init__(self, by_scan: dict[str, ReasonerOutput], name: str = "qwen_teacher"):
        self.name = name
        self._by_scan = by_scan

    def _reason(self, case: Case) -> ReasonerOutput:
        out = self._by_scan[case.scan_id]
        return ReasonerOutput(
            priority=out.priority,
            should_explain=out.should_explain,
            should_review=out.should_review,
            reasoning=out.reasoning,
            narrative=out.narrative,
            latency_ms=out.latency_ms,
        )
