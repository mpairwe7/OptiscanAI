#!/usr/bin/env python3
"""Quantitative faithfulness test for the visual explainability layer.

Reviewers asked whether the saliency maps are actually tied to the evidence the
model uses. We answer with the standard deletion / insertion protocol
(Petsiuk et al., BMVC 2018) run at ViT patch granularity, against a random
saliency control on the identical images and classes:

  deletion  -- progressively blank the most-salient patches; a faithful map
               makes the target probability collapse fast (lower AUC better)
  insertion -- progressively reveal the most-salient patches on a blank canvas;
               a faithful map recovers the probability fast (higher AUC better)

Reported per attribution method with a paired Wilcoxon test versus random.

Usage:
    python3 scripts/ncc2026_explain_eval.py --n-images 60 --device cuda:7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("USE_PRETRAINED", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("explain")

CACHE = REPO / "outputs/ncc2026/cache"
OUT_DIR = REPO / "outputs/ncc2026"
CKPT = REPO / "outputs/checkpoints/v2/final_with_thresholds.pth"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

GRID = 14  # ViT-L/16 at 224 -> 14x14 patch tokens
PATCH = 16
N_STEPS = 20


def normalise(arr: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN.to(device)) / STD.to(device)


def build_model(device):
    from src.models.retinal_foundation_hybrid_v2 import RetinalFoundationHybridV2
    from src.models.vignn import ClinicalKnowledgeGraph

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    classes = ckpt["disease_columns"]
    model = RetinalFoundationHybridV2(
        num_classes=len(classes),
        hidden_dim=512,
        clinical_knowledge_graph=ClinicalKnowledgeGraph(classes),
        backbone="vit_large_patch16_224",
        img_size=224,
        use_lora=True,
        lora_rank=16,
        freeze_backbone=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(device).eval(), classes


def gradcam(model, x: torch.Tensor, target: int) -> np.ndarray:
    """Grad-CAM over the projected patch tokens of the retinal backbone."""
    feats: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        out.retain_grad()
        feats["tokens"] = out
        return out

    handle = model.encoder.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, target].backward()
        tok = feats["tokens"]  # [1, N, D]
        grad = tok.grad  # [1, N, D]
        weights = grad.mean(dim=1, keepdim=True)  # [1, 1, D] channel importance
        cam = torch.relu((tok * weights).sum(-1))[0]  # [N]
    finally:
        handle.remove()
    cam = cam[-GRID * GRID :] if cam.numel() > GRID * GRID else cam
    return cam.detach().float().cpu().numpy()


def integrated_gradients(model, x: torch.Tensor, target: int, steps: int = 24) -> np.ndarray:
    """Aumann-Shapley (Integrated Gradients) attribution, pooled to patches."""
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (baseline + a * (x - baseline)).clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(xi)[0, target].backward()
        total = total + xi.grad
    attr = ((x - baseline) * total / steps)[0].sum(0).abs()  # [224, 224]
    pooled = attr.reshape(GRID, PATCH, GRID, PATCH).mean(dim=(1, 3))
    return pooled.detach().float().cpu().numpy().ravel()


@torch.no_grad()
def curve(model, x: torch.Tensor, target: int, order: np.ndarray, mode: str) -> np.ndarray:
    """Probability of `target` as patches are removed (deletion) or added (insertion)."""
    n = GRID * GRID
    steps = np.linspace(0, n, N_STEPS + 1).astype(int)
    batch = []
    for k in steps:
        mask = (
            torch.ones(n, device=x.device)
            if mode == "deletion"
            else torch.zeros(n, device=x.device)
        )
        sel = torch.from_numpy(order[:k].copy()).to(x.device).long()
        mask[sel] = 0.0 if mode == "deletion" else 1.0
        m = mask.reshape(1, 1, GRID, GRID).repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3)
        batch.append(x * m)  # blanked patches sit at the normalised dataset mean (0)
    probs = []
    stacked = torch.cat(batch, 0)
    for i in range(0, len(stacked), 8):
        probs.append(torch.sigmoid(model(stacked[i : i + 8]).float())[:, target].cpu().numpy())
    return np.concatenate(probs)


def auc_of(curve_vals: np.ndarray) -> float:
    return float(np.trapezoid(curve_vals, dx=1.0 / (len(curve_vals) - 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, classes = build_model(device)

    imgs = np.load(CACHE / "test_images_224.npy")
    df = pd.read_csv(CACHE / "test_labels.csv", encoding="utf-8-sig")
    labels = df[classes].to_numpy(dtype=np.float32)
    probs = np.load(OUT_DIR / "probs_test_fp32.npz")["probs"]

    # Evaluate on true positives the model is confident about: the cases where a
    # clinician would actually look at the heat map.
    cands = []
    for ci, cname in enumerate(classes):
        pos = np.where(labels[:, ci] == 1)[0]
        for i in pos:
            cands.append((float(probs[i, ci]), int(i), ci, cname))
    cands.sort(reverse=True)
    rng = np.random.default_rng(args.seed)
    picked, seen = [], set()
    for _, i, ci, cname in cands:
        if i in seen:
            continue
        picked.append((i, ci, cname))
        seen.add(i)
        if len(picked) >= args.n_images:
            break
    logger.info("evaluating %d image/class pairs", len(picked))

    rows = []
    for k, (i, ci, cname) in enumerate(picked):
        x = normalise(imgs[i : i + 1], device)
        sal = {
            "gradcam": gradcam(model, x, ci),
            "integrated_gradients": integrated_gradients(model, x, ci),
            "random": rng.random(GRID * GRID),
        }
        row = {"image_id": int(df["ID"].iloc[i]), "class": cname, "p": float(probs[i, ci])}
        for meth, s in sal.items():
            order = np.argsort(-s)
            row[f"{meth}_deletion_auc"] = auc_of(curve(model, x, ci, order, "deletion"))
            row[f"{meth}_insertion_auc"] = auc_of(curve(model, x, ci, order, "insertion"))
        rows.append(row)
        if k % 10 == 0:
            logger.info("  %d/%d", k, len(picked))

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "explainability_faithfulness.csv", index=False)

    summary = {"n_pairs": len(res), "methods": {}}
    for meth in ("gradcam", "integrated_gradients", "random"):
        summary["methods"][meth] = {
            "deletion_auc_mean": float(res[f"{meth}_deletion_auc"].mean()),
            "deletion_auc_std": float(res[f"{meth}_deletion_auc"].std()),
            "insertion_auc_mean": float(res[f"{meth}_insertion_auc"].mean()),
            "insertion_auc_std": float(res[f"{meth}_insertion_auc"].std()),
        }
    for meth in ("gradcam", "integrated_gradients"):
        for mode, better in (("deletion", "less"), ("insertion", "greater")):
            stat, p = wilcoxon(
                res[f"{meth}_{mode}_auc"], res[f"random_{mode}_auc"], alternative=better
            )
            summary["methods"][meth][f"{mode}_vs_random_p"] = float(p)
            summary["methods"][meth][f"{mode}_vs_random_stat"] = float(stat)
    (OUT_DIR / "explainability_faithfulness.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
