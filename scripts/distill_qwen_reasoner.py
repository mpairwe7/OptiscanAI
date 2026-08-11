#!/usr/bin/env python3
"""Distil the Qwen3-8B teacher into a small Qwen student via LoRA SFT.

Runs under the **system python** (`/usr/bin/python3`), which has a CUDA-capable
torch (2.6.0+cu124) plus transformers/peft/accelerate. Trains on the teacher
traces produced by ``build_sft_dataset.py`` and writes a merged HF model dir that
``DistilledLLMReasoner`` can load directly.

    PYTHONPATH=. /usr/bin/python3 scripts/distill_qwen_reasoner.py \
        --data outputs/reasoner_comparison_real/sft_data.jsonl \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --out outputs/distilled_qwen --device cuda:2 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("distill_qwen")


def load_rows(path: str, val_frac: float, split: str) -> list[dict]:
    """Split traces into train/val.

    Sorted by ``scan_id`` so the split is reproducible and — critically —
    identical to the one the evaluators derive (``load_test_cases`` in
    ``sweep_narrator_architectures.py`` / ``quantize_distilled_reasoner.py``
    sorts before cutting at the same fraction). Without the sort the two agree
    only as long as the trace file happens to be written in sorted order; if it
    is ever regenerated in another order the "held-out" split would silently
    overlap the training set.
    """
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["scan_id"])
    cut = int(len(rows) * (1 - val_frac))
    return rows[:cut] if split == "train" else rows[cut:]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default="outputs/distilled_qwen")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len", type=int, default=768)
    p.add_argument("--val-frac", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None, help="cap train rows (smoke)")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device=%s base=%s", device, args.base_model)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bf16 only pays off on GPU; on CPU it is emulated and slow, so use fp32
    # there. attn_implementation="eager" avoids the flash/SDPA path; we also run
    # CPU with CUDA hidden because this host's Triton can't JIT-compile its CUDA
    # helper (no python3.10-dev), which the GPU LM path would otherwise hit.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    rows = load_rows(args.data, args.val_frac, "train")
    if args.limit:
        rows = rows[: args.limit]
    logger.info("training on %d examples for %d epochs", len(rows), args.epochs)

    def encode(row: dict):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tok(chat, add_special_tokens=False)["input_ids"]
        full_ids = tok(chat + row["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        full_ids = full_ids[: args.max_len]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return full_ids, labels

    encoded = [encode(r) for r in rows]

    def collate(batch):
        maxlen = max(len(x[0]) for x in batch)
        pad = tok.pad_token_id
        input_ids, attn, labels = [], [], []
        for ids, lab in batch:
            n = maxlen - len(ids)
            input_ids.append(ids + [pad] * n)
            attn.append([1] * len(ids) + [0] * n)
            labels.append(lab + [-100] * n)
        return (
            torch.tensor(input_ids, device=device),
            torch.tensor(attn, device=device),
            torch.tensor(labels, device=device),
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(encoded))
        total = 0.0
        for start in range(0, len(encoded), args.batch_size):
            batch = [encoded[i] for i in perm[start : start + args.batch_size]]
            input_ids, attn, labels = collate(batch)
            opt.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(batch)
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, args.epochs, total / max(1, len(encoded)))

    logger.info("merging LoRA and saving to %s", args.out)
    merged = model.merge_and_unload()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    logger.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
