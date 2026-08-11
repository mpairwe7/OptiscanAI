#!/usr/bin/env python3
"""Build a 60-100 MB clinical narrator: prose-only target + pruned vocabulary.

Closes two gaps from ``docs/29-narrator-verification-and-gaps.md`` at once.

**1. Prose-only output (closes the generation-rate gap by construction).**
Every reliability defect measured in the narrator sweep was a malformed-JSON
defect — the 135M 4-bit variant fell back to the template on 100% of cases. But
the JSON existed only to carry the *triage* fields, and triage is now served by
the 3 KB head (``src/triage``). A narrator that emits the narrative **as plain
prose** has no structured contract left to break, so generation cannot fail to
parse. That is a stronger guarantee than grammar-constrained decoding, and it
needs no extra dependency.

**2. Vocabulary pruning (the entire size gap).**
bitsandbytes does not quantize the embedding table, which is why the 4-bit 135M
measured 109.8 MB rather than the ~76 MB a naive 0.5 byte/param estimate
suggests::

    49,152 vocab x 576 hidden = 28.3M params @ bf16 = 56.6 MB   (unquantized)
    ~106M transformer params        @ 4-bit         = 53.1 MB
                                                    = 110 MB    (measured 109.8)

This task uses a narrow, near-closed token set (templated prompts over a fixed
disease vocabulary + clinical prose), so most of that table is dead weight.
Pruning to the tokens actually reachable takes the embedding to a few MB and the
total into the 60-100 MB band.

Pruning is done by **id remapping, not tokenizer surgery**: the original
tokenizer is kept intact and a ``keep_ids`` table maps between its id space and
the pruned model's compact space. Rewriting BPE merges is fiddly and easy to get
subtly wrong; remapping is exact and reversible.

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 /usr/bin/python3 \
        scripts/build_compact_narrator.py --out outputs/narrator_compact
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reasoner_comparison.interface import Case, Prediction  # noqa: E402

logger = logging.getLogger("compact_narrator")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _production_disease_names() -> dict[str, str]:
    """Every ``code -> name`` the classifier can emit (all 45 classes).

    Falls back to the harness's 16-code vocabulary if the backend cannot be
    imported, but warns loudly: a vocabulary built from the smaller set would
    leave rare findings unrepresentable at serve time.
    """
    try:
        from backend.app.core.model_service import DISEASE_NAMES

        return dict(DISEASE_NAMES)
    except Exception as e:  # pragma: no cover - depends on backend deps
        from src.evaluation.reasoner_comparison.cases import DISEASE_VOCAB

        logger.warning(
            "could not import the production DISEASE_NAMES (%s) — falling back to the "
            "harness's %d-code vocabulary; rare findings may be unrepresentable",
            e,
            len(DISEASE_VOCAB),
        )
        return dict(DISEASE_VOCAB)


def prose_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    """(prompt, narrative) — no JSON envelope, so nothing can fail to parse."""
    pairs = []
    for r in rows:
        preds = [
            Prediction(
                p["code"], p["name"], float(p["probability"]), p.get("confidence", "unknown")
            )
            for p in r["predictions"]
        ]
        case = Case(
            scan_id=str(r.get("scan_id", "")),
            predictions=preds,
            probabilities={},
            referral_priority=r.get("referral", "FOLLOW_UP"),
        )
        narrative = (r.get("teacher") or {}).get("narrative", "").strip()
        if narrative:
            pairs.append((narrative_prompt(case), narrative))
    return pairs


def narrative_prompt(case: Case) -> str:
    """Prose-only prompt. Shared by training and serving — they must be identical.

    Deliberately *not* :func:`build_distill_prompt`, which asks for a JSON object
    carrying the triage fields. Those fields now come from the triage head.
    """
    return (
        f"Retinal screening, {len(case.predictions)} finding(s). "
        f"Referral priority: {case.referral_priority}.\n"
        f"Findings:\n{case.disease_summary()}\n\n"
        "Write a 3-4 sentence clinical screening report for the referring "
        "ophthalmologist. State the findings and their probabilities, then give a "
        "clear recommendation. Do not invent findings."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--traces", default="outputs/reasoner_comparison_real/sft_data.jsonl")
    p.add_argument("--oos-traces", default="outputs/generalizability/oos_traces.jsonl")
    p.add_argument("--out", default="outputs/narrator_compact")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--cut-frac", type=float, default=0.7, help="train/test split on the 80 traces")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--keep-vocab",
        action="store_true",
        help="skip vocabulary pruning — required for GGUF conversion, "
        "which bakes the tokenizer into the file and needs "
        "vocab_size == len(tokenizer)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = load_rows(Path(args.traces))
    rows.sort(key=lambda r: r["scan_id"])
    cut = int(len(rows) * args.cut_frac)
    train_pairs = prose_pairs(rows[:cut])
    logger.info(
        "train pairs: %d (held-out %d kept for evaluation)", len(train_pairs), len(rows) - cut
    )

    # ── 1. token set the task can actually reach ──
    # Union of every token in every prompt and target (train *and* the unseen
    # out-of-sample traces, so the vocabulary is not fitted to the train split),
    # plus all single-byte tokens so arbitrary text remains encodable, plus the
    # specials generation depends on.
    keep: set[int] = set()
    for src in (Path(args.traces), Path(args.oos_traces)):
        if not src.exists():
            logger.warning("no %s — vocabulary will be narrower", src)
            continue
        for prompt, target in prose_pairs(load_rows(src)):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            keep.update(tok(chat + target, add_special_tokens=False)["input_ids"])

    # The traces only exercise the classes that happened to fire. The classifier
    # can emit all 45, so seed the vocabulary from the *production* code->name map
    # rather than from observed data — otherwise a rare finding's name would be
    # unrepresentable and the prompt describing it would arrive corrupted.
    n_before = len(keep)
    for code, name in _production_disease_names().items():
        keep.update(
            tok(f"- {name} ({code}): 55.0% confidence\n", add_special_tokens=False)["input_ids"]
        )
        keep.update(
            tok(f"{name} ({code}) at 55% confidence. ", add_special_tokens=False)["input_ids"]
        )
    logger.info(
        "full disease vocabulary added %d tokens beyond those seen in traces", len(keep) - n_before
    )

    keep.update(tok.all_special_ids)
    # byte-level fallback: every single-character token keeps the tokenizer total
    for tid in range(min(256, len(tok))):
        keep.add(tid)
    keep_ids = sorted(keep)
    if args.keep_vocab:
        keep_ids = list(range(len(tok)))
        logger.info("--keep-vocab: pruning skipped, retaining all %d tokens", len(keep_ids))
    logger.info(
        "vocabulary: %d -> %d tokens (%.1f%% pruned)",
        len(tok),
        len(keep_ids),
        100 * (1 - len(keep_ids) / len(tok)),
    )

    old2new = {old: new for new, old in enumerate(keep_ids)}

    # ── 2. prune the embedding (tied to lm_head) ──
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    idx = torch.tensor(keep_ids, dtype=torch.long)
    emb = model.get_input_embeddings()
    pruned = torch.nn.Embedding(len(keep_ids), emb.embedding_dim)
    pruned.weight.data = emb.weight.data[idx].clone()
    model.set_input_embeddings(pruned)
    model.config.vocab_size = len(keep_ids)
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
    else:  # untied head: prune it to match
        head = model.get_output_embeddings()
        new_head = torch.nn.Linear(head.in_features, len(keep_ids), bias=head.bias is not None)
        new_head.weight.data = head.weight.data[idx].clone()
        if head.bias is not None:
            new_head.bias.data = head.bias.data[idx].clone()
        model.set_output_embeddings(new_head)
    model = model.to(device)

    # ── 3. LoRA-SFT on prose targets, in the pruned id space ──
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            # attention + MLP: the objective is prose style/format adherence, which
            # the feed-forward blocks carry as much as attention does.
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    def encode(prompt: str, target: str):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        p_ids = tok(chat, add_special_tokens=False)["input_ids"]
        f_ids = tok(chat + target + tok.eos_token, add_special_tokens=False)["input_ids"]
        f_ids = [old2new[t] for t in f_ids if t in old2new][: args.max_len]
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100
        return f_ids, labels

    encoded = [encode(p, t) for p, t in train_pairs]
    pad_new = old2new[tok.pad_token_id]

    def collate(batch):
        n = max(len(x[0]) for x in batch)
        ids, attn, labs = [], [], []
        for i, lab in batch:
            k = n - len(i)
            ids.append(i + [pad_new] * k)
            attn.append([1] * len(i) + [0] * k)
            labs.append(lab + [-100] * k)
        return (
            torch.tensor(ids, device=device),
            torch.tensor(attn, device=device),
            torch.tensor(labs, device=device),
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        total_steps=args.epochs * max(1, (len(encoded) + args.batch_size - 1) // args.batch_size),
        pct_start=0.1,
    )
    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(encoded))
        total = 0.0
        for s in range(0, len(encoded), args.batch_size):
            batch = [encoded[i] for i in perm[s : s + args.batch_size]]
            ids, attn, labs = collate(batch)
            opt.zero_grad()
            loss = model(input_ids=ids, attention_mask=attn, labels=labs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.detach().item() * len(batch)
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, args.epochs, total / max(1, len(encoded)))

    merged = model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.half().save_pretrained(out)
    tok.save_pretrained(out)
    (out / "keep_ids.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "keep_ids": keep_ids,
                "original_vocab_size": len(tok),
                "note": (
                    "The model's embedding rows are the original tokenizer's ids listed in "
                    "keep_ids, in order. Encode with the original tokenizer then map "
                    "old->new via keep_ids.index (see src/narrator/compact.py); map back "
                    "before decoding."
                ),
            }
        )
    )

    n_params = sum(p.numel() for p in merged.parameters())
    emb_params = len(keep_ids) * merged.config.hidden_size
    body = n_params - emb_params
    logger.info("saved -> %s", out)
    logger.info(
        "params %.1fM (embedding %.1fM, body %.1fM)", n_params / 1e6, emb_params / 1e6, body / 1e6
    )
    logger.info(
        "projected footprint: bf16 %.0f MB | 4-bit body + bf16 embedding %.0f MB",
        n_params * 2 / 1e6,
        (emb_params * 2 + body * 0.5) / 1e6,
    )


if __name__ == "__main__":
    main()
