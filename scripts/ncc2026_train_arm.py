#!/usr/bin/env python3
"""Train one arm of the NCC 2026 baseline / ablation study on RFMiD.

Every arm shares the data, the loss (AsymmetricLossV2, the production setting),
the schedule, the model-selection criterion (validation mAP) and the post-hoc
per-class threshold optimisation, so differences between arms are attributable
to the backbone and the adaptation strategy alone.

Arms:
    resnet50, tf_efficientnet_b3, vit_base_patch16_224  -- ImageNet, full fine-tune
    retfound_head   -- RETFound ViT-L frozen, head only        (LoRA ablated out)
    retfound_lora   -- RETFound ViT-L + LoRA r=16 + head       (production recipe)

Usage:
    python3 scripts/ncc2026_train_arm.py --arm resnet50 --device cuda:7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("arm")

CACHE = REPO / "outputs/ncc2026/cache"
OUT_DIR = REPO / "outputs/ncc2026"
CKPT = REPO / "outputs/checkpoints/v2/final_with_thresholds.pth"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

RETFOUND_ARMS = {"retfound_head", "retfound_lora"}


def rfmid_statistics() -> tuple[torch.Tensor, torch.Tensor]:
    """Channel mean/std measured on the RFMiD training split itself."""
    x = np.load(CACHE / "train_images_256.npy").astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1, 2))
    std = x.std(axis=(0, 1, 2))
    return (
        torch.tensor(mean).view(1, 3, 1, 1),
        torch.tensor(std).view(1, 3, 1, 1),
    )


def apply_clahe(arr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB, the usual fundus contrast normalisation.

    Applied to the whole split up front so it costs nothing per epoch.
    """
    import cv2

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lab = cv2.cvtColor(arr[i], cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out


def load_split(split: str, classes: list[str], size: int):
    imgs = np.load(CACHE / f"{split}_images_{size}.npy")
    df = pd.read_csv(CACHE / f"{split}_labels.csv", encoding="utf-8-sig")
    return imgs, df[classes].to_numpy(dtype=np.float32), df["ID"].to_numpy()


def to_gpu_batch(arr: np.ndarray, device, mean=None, std=None) -> torch.Tensor:
    mean = MEAN if mean is None else mean
    std = STD if std is None else std
    x = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - mean.to(device)) / std.to(device)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Geometric + photometric augmentation matching configs/hybrid_precision_2026."""
    b = x.shape[0]
    flip_h = torch.rand(b, generator=gen, device=x.device) < 0.5
    flip_v = torch.rand(b, generator=gen, device=x.device) < 0.5
    x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(-1), x)
    x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(-2), x)
    # brightness / contrast jitter in normalised space
    scale = 1.0 + (torch.rand(b, 1, 1, 1, generator=gen, device=x.device) - 0.5) * 0.4
    shift = (torch.rand(b, 1, 1, 1, generator=gen, device=x.device) - 0.5) * 0.4
    return x * scale + shift


def random_crop_224(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """RandomResizedCrop(224, scale=(0.8, 1.0)) applied per batch."""
    s = float(0.8 + 0.2 * torch.rand(1, generator=gen, device=x.device).item())
    side = max(int(round(x.shape[-1] * (s**0.5))), 224)
    top = int(torch.randint(0, x.shape[-2] - side + 1, (1,), generator=gen, device=x.device))
    left = int(torch.randint(0, x.shape[-1] - side + 1, (1,), generator=gen, device=x.device))
    crop = x[:, :, top : top + side, left : left + side]
    return F.interpolate(crop, size=(224, 224), mode="bicubic", align_corners=False)


def build_arm(arm: str, num_classes: int, classes: list[str], device):
    if arm in RETFOUND_ARMS:
        os.environ["USE_PRETRAINED"] = "1"  # load real RETFound MAE weights
        from src.models.retinal_foundation_hybrid_v2 import RetinalFoundationHybridV2
        from src.models.vignn import ClinicalKnowledgeGraph

        model = RetinalFoundationHybridV2(
            num_classes=num_classes,
            hidden_dim=512,
            clinical_knowledge_graph=ClinicalKnowledgeGraph(classes),
            backbone="vit_large_patch16_224",
            img_size=224,
            use_lora=(arm == "retfound_lora"),
            lora_rank=16,
            freeze_backbone=True,
        )
    else:
        import timm

        model = timm.create_model(arm, pretrained=True, num_classes=num_classes)
    return model.to(device)


@torch.no_grad()
def predict(model, imgs: np.ndarray, device, batch: int = 32, mean=None, std=None) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(imgs), batch):
        x = to_gpu_batch(imgs[i : i + batch], device, mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def macro_ap(y: np.ndarray, p: np.ndarray) -> float:
    vals = [
        average_precision_score(y[:, c], p[:, c]) for c in range(y.shape[1]) if y[:, c].sum() > 0
    ]
    return float(np.mean(vals)) if vals else 0.0


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    vals = [
        roc_auc_score(y[:, c], p[:, c]) for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)
    ]
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:7")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--norm",
        default="imagenet",
        choices=["imagenet", "dataset"],
        help="normalisation statistics: ImageNet (matches RETFound pretraining) or RFMiD's own",
    )
    ap.add_argument(
        "--preproc",
        default="none",
        choices=["none", "clahe"],
        help="optional contrast normalisation applied before the transform",
    )
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    classes = torch.load(CKPT, map_location="cpu", weights_only=False)["disease_columns"]
    xtr, ytr, _ = load_split("train", classes, 256)
    xva, yva, _ = load_split("val", classes, 224)
    xte, yte, ids_te = load_split("test", classes, 224)
    logger.info("train %s  val %s  test %s", xtr.shape, xva.shape, xte.shape)

    if args.preproc == "clahe":
        xtr, xva, xte = apply_clahe(xtr), apply_clahe(xva), apply_clahe(xte)
        logger.info("CLAHE applied to all three splits")
    if args.norm == "dataset":
        mean, std = rfmid_statistics()
        logger.info("RFMiD statistics: mean=%s std=%s", mean.flatten(), std.flatten())
    else:
        mean, std = MEAN, STD

    model = build_arm(args.arm, len(classes), classes, device)
    from src.models.retinal_foundation_hybrid_v2 import AsymmetricLossV2

    criterion = AsymmetricLossV2(gamma_neg=4.0, gamma_pos=0.0, clip=0.05, label_smoothing=0.05)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    lr = 5e-4 if args.arm in RETFOUND_ARMS else 1e-4
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    logger.info(
        "%s | trainable %.2fM / %.1fM total | lr %.0e", args.arm, n_train / 1e6, n_total / 1e6, lr
    )

    gen = torch.Generator(device=device).manual_seed(args.seed)
    best = {"map": -1.0, "epoch": -1, "state": None}
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(xtr))
        losses = []
        for i in range(0, len(order), args.batch_size):
            idx = np.sort(order[i : i + args.batch_size])
            x = to_gpu_batch(xtr[idx], device, mean, std)
            x = augment(random_crop_224(x, gen), gen)
            y = torch.from_numpy(ytr[idx]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        pva = predict(model, xva, device, mean=mean, std=std)
        vmap, vauc = macro_ap(yva, pva), macro_auc(yva, pva)
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "val_map": vmap, "val_auc": vauc}
        )
        logger.info(
            "epoch %2d  loss %.4f  val mAP %.4f  val AUC %.4f  (%.1f min)",
            epoch,
            np.mean(losses),
            vmap,
            vauc,
            (time.time() - t0) / 60,
        )
        if vmap > best["map"]:
            best = {
                "map": vmap,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    logger.info("best epoch %d (val mAP %.4f)", best["epoch"], best["map"])
    model.load_state_dict(best["state"])

    pva = predict(model, xva, device, mean=mean, std=std)
    pte = predict(model, xte, device, mean=mean, std=std)

    name = args.arm + (f"_{args.tag}" if args.tag else "")

    from src.evaluation.precision_threshold_optimizer import (
        optimize_thresholds_with_precision_floor,
    )

    taus, thr_report = optimize_thresholds_with_precision_floor(
        pva, yva, min_precision=0.10, disease_names=classes, fallback_threshold=0.95
    )

    np.savez_compressed(
        OUT_DIR / f"probs_test_arm_{name}.npz",
        probs=pte,
        labels=yte,
        ids=ids_te,
        classes=np.array(classes),
        thresholds=np.asarray(taus, dtype=np.float32),
    )
    np.savez_compressed(
        OUT_DIR / f"probs_val_arm_{name}.npz",
        probs=pva,
        labels=yva,
        classes=np.array(classes),
        thresholds=np.asarray(taus, dtype=np.float32),
    )
    summary = {
        "arm": args.arm,
        "name": name,
        "norm": args.norm,
        "preproc": args.preproc,
        "epochs": args.epochs,
        "best_epoch": best["epoch"],
        "val_map_best": best["map"],
        "params_total_M": round(n_total / 1e6, 2),
        "params_trainable_M": round(n_train / 1e6, 3),
        "trainable_fraction": round(n_train / n_total, 5),
        "lr": lr,
        "minutes": round((time.time() - t0) / 60, 1),
        "test_macro_auc": macro_auc(yte, pte),
        "test_macro_ap": macro_ap(yte, pte),
        "history": history,
        "threshold_summary": thr_report.get("summary", {}),
    }
    (OUT_DIR / f"arm_{name}.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info(
        "DONE %s  test macro AUC %.4f  macro AP %.4f",
        args.arm,
        summary["test_macro_auc"],
        summary["test_macro_ap"],
    )


if __name__ == "__main__":
    main()
