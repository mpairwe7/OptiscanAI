"""Serving wrapper for the compact (vocabulary-pruned) clinical narrator.

The narrator emits the report as **plain prose**, not a JSON envelope. Triage is
served by :mod:`src.triage`, so the narrator has no structured contract left to
break — which is what removes the parse-failure mode that made the 4-bit 135M
narrator fall back to the template on 100% of cases
(``docs/29-narrator-verification-and-gaps.md`` §2.2).

Its embedding table is pruned to the tokens this task can reach. The original
tokenizer is kept untouched; ``keep_ids.json`` maps between the tokenizer's id
space and the model's compact one. Rewriting BPE merges would be easy to get
subtly wrong, whereas a remap is exact and reversible:

    text --tokenizer--> old ids --old2new--> model --new2old--> tokenizer --> text

Because the output head only has rows for kept tokens, the model *cannot* emit a
token outside the set — the pruning is self-enforcing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def narrative_prompt(findings_summary: str, n_findings: int, referral_priority: str) -> str:
    """The prose prompt. Must stay byte-identical to the training prompt in
    ``scripts/build_compact_narrator.py`` — a drifted prompt silently degrades
    generation quality with no error."""
    return (
        f"Retinal screening, {n_findings} finding(s). "
        f"Referral priority: {referral_priority}.\n"
        f"Findings:\n{findings_summary}\n\n"
        "Write a 3-4 sentence clinical screening report for the referring "
        "ophthalmologist. State the findings and their probabilities, then give a "
        "clear recommendation. Do not invent findings."
    )


def findings_summary(predictions: list[dict]) -> str:
    """Mirror of ``Case.disease_summary`` for the pipeline's prediction dicts."""
    return "\n".join(
        f"- {p['name']} ({p['code']}): {float(p['probability']):.1%} confidence"
        for p in predictions[:10]
    )


class CompactNarrator:
    """Vocabulary-pruned causal LM that returns a clinical narrative string."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        load_4bit: bool = False,
        load_8bit: bool = False,
        dtype: str | None = "float16",
        max_new_tokens: int = 160,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = Path(model_dir)
        self._torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens

        spec = json.loads((model_dir / "keep_ids.json").read_text())
        self.keep_ids: list[int] = spec["keep_ids"]
        self.old2new: dict[int, int] = {o: n for n, o in enumerate(self.keep_ids)}
        #: tokens that could not be represented even byte-wise — should stay 0.
        self._unmapped = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        kwargs: dict[str, Any] = {"attn_implementation": "eager"}
        if load_4bit or load_8bit:
            from transformers import BitsAndBytesConfig

            if not torch.cuda.is_available() or device == "cpu":
                raise RuntimeError("quantized loading needs CUDA (bitsandbytes kernels)")
            # 4-bit reaches ~54 MB but measurably degrades content fidelity on a
            # 135M model (omission 0.42 vs 0.00 at bf16); 8-bit trades ~50 MB back
            # for far better faithfulness. See docs/29 §3.7.
            kwargs["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                if load_4bit
                else BitsAndBytesConfig(load_in_8bit=True)
            )
            kwargs["device_map"] = {"": device}
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).eval()
        else:
            if dtype:
                kwargs["torch_dtype"] = getattr(torch, dtype)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).to(device).eval()

        if self.model.config.vocab_size != len(self.keep_ids):
            raise ValueError(
                f"pruned model expects {self.model.config.vocab_size} tokens but "
                f"keep_ids lists {len(self.keep_ids)} — artifact mismatch"
            )

    def size_mb(self) -> float:
        total = sum(p.numel() * p.element_size() for p in self.model.parameters())
        total += sum(b.numel() * b.element_size() for b in self.model.buffers())
        return total / 1e6

    def _to_compact(self, ids: list[int]) -> list[int]:
        """Map tokenizer ids into the pruned space.

        A token outside the kept set is **re-encoded byte-by-byte** rather than
        dropped. Every single-byte token is retained precisely so that any string
        stays representable: dropping instead would hand the model a silently
        corrupted prompt — a finding's name could vanish mid-sentence with no
        error anywhere. Byte fallback costs a few extra tokens and preserves the
        text exactly.
        """
        out: list[int] = []
        for t in ids:
            new = self.old2new.get(t)
            if new is not None:
                out.append(new)
                continue
            piece = self.tokenizer.decode([t])
            fallback = [
                self.old2new[b]
                for b in self.tokenizer(piece, add_special_tokens=False)["input_ids"]
                if b in self.old2new
            ]
            if fallback:
                out.extend(fallback)
            else:
                self._unmapped += 1
                logger.warning(
                    "token %d (%r) is unrepresentable in the pruned vocabulary", t, piece
                )
        return out

    def _to_original(self, ids: list[int]) -> list[int]:
        return [self.keep_ids[t] for t in ids if 0 <= t < len(self.keep_ids)]

    def generate(self, prompt: str) -> str:
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        old_ids = self.tokenizer(chat, add_special_tokens=False)["input_ids"]
        ids = self._to_compact(old_ids)
        input_ids = self._torch.tensor([ids], device=self.device)
        eos_new = self.old2new.get(self.tokenizer.eos_token_id)
        with self._torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=self._torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_new,
                pad_token_id=eos_new,
            )
        generated = out[0][input_ids.shape[1] :].tolist()
        return self.tokenizer.decode(self._to_original(generated), skip_special_tokens=True).strip()

    def narrate(self, predictions: list[dict], referral_priority: str) -> str:
        """Produce a clinical narrative for one case. Always returns a string."""
        return self.generate(
            narrative_prompt(findings_summary(predictions), len(predictions), referral_priority)
        )
