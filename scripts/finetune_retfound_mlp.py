#!/usr/bin/env python3
"""
Fine-tune RETFound ViT-L with a lightweight MLP head for multi-label retinal disease classification.

Diagnosis showed the SGT head (12.7M params) destroys RETFound features on this small dataset.
This script uses a 2-layer MLP head (~0.3M params) with end-to-end fine-tuning.

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 python3 -u scripts/finetune_retfound_mlp.py 2>&1 | tee outputs/full_pipeline/finetune_retfound_mlp.log
"""
import logging
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.datamodule import RetinalDataModule
from src.training.early_stopping import AdvancedEarlyStopping
from src.training.ema import ModelEMA
from src.training.losses import build_loss
from src.training.metrics import MetricTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger("finetune-mlp")

# ── Config ─────────────────────────────────────────────────────────────
BACKBONE = "vit_large_patch16_224"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
BACKBONE_LR = 2e-6  # very gentle — preserve retinal features
HEAD_LR = 3e-4
WARMUP_EPOCHS = 3
PATIENCE = 7
WEIGHT_DECAY = 0.05  # higher WD for large backbone
LABEL_SMOOTH = 0.02
DROP_PATH = 0.2  # stochastic depth for ViT-L
HEAD_DROPOUT = 0.3
OUT = Path("outputs/full_pipeline")


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model ──────────────────────────────────────────────────────────────
class RETFoundMLP(nn.Module):
    """RETFound ViT-L backbone + lightweight 2-layer MLP head.

    Architecture:
        ViT-L/16 (304M, frozen drop_path) → CLS token [1024]
        → LayerNorm → Linear(1024, 512) → GELU → Dropout
        → Linear(512, num_classes)
    """

    def __init__(self, num_classes=45, drop_path=0.2, head_dropout=0.3):
        super().__init__()
        # Backbone with stochastic depth
        self.encoder = timm.create_model(
            BACKBONE,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=drop_path,
        )
        self.backbone_dim = self.encoder.num_features  # 1024

        # Load RETFound weights
        for lp in ["pretrained_weights/RETFound_cfp.pth"]:
            if os.path.exists(lp):
                ckpt = torch.load(lp, map_location="cpu", weights_only=False)
                state = {
                    k: v
                    for k, v in ckpt["model"].items()
                    if not k.startswith("decoder") and "mask_token" not in k
                }
                self.encoder.load_state_dict(state, strict=False)
                log.info(f"  Loaded RETFound weights from {lp}")
                break

        # Lightweight MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Linear(self.backbone_dim, 512),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, num_classes),
        )
        # Init head
        nn.init.trunc_normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)
        nn.init.trunc_normal_(self.head[4].weight, std=0.02)
        nn.init.zeros_(self.head[4].bias)

    def forward(self, x):
        # CLS token from ViT
        features = self.encoder.forward_features(x)  # [B, N+1, 1024]
        cls_token = features[:, 0]  # [B, 1024]
        return self.head(cls_token)  # [B, num_classes]


# ── Dataset ────────────────────────────────────────────────────────────
class PreCachedDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dirs, cols, transform=None, img_size=224):
        self.cols = cols
        self.tfm = transform
        self.labels = (
            df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        )
        ids = df["ID"].values
        dirs = [Path(d) for d in img_dirs]
        log.info(f"  Pre-caching {len(ids)} images at {img_size}px...")
        t0 = time.time()
        self.cache = []
        for img_id in ids:
            loaded = False
            for d in dirs:
                for ext in (".png", ".jpg", ".jpeg"):
                    p = d / f"{img_id}{ext}"
                    if p.exists():
                        try:
                            self.cache.append(
                                np.array(
                                    Image.open(p)
                                    .convert("RGB")
                                    .resize((img_size, img_size), Image.BILINEAR),
                                    dtype=np.uint8,
                                )
                            )
                            loaded = True
                        except Exception:
                            pass
                        break
                if loaded:
                    break
            if not loaded:
                self.cache.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
        log.info(f"  Cached {len(self.cache)} in {time.time()-t0:.1f}s")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.cache[i])
        return (self.tfm(img) if self.tfm else img), torch.from_numpy(self.labels[i])

    def get_pos_weights(self):
        pos = self.labels.sum(0).clip(min=1)
        neg = len(self) - pos
        return torch.from_numpy(np.clip(neg / pos, 0.5, 50).astype(np.float32))


# ── Main ───────────────────────────────────────────────────────────────
def main():
    set_seed(42)
    cfg = yaml.safe_load(open("configs/train.yaml"))
    cfg["training"]["loss"] = "focal"
    cfg["training"]["focal_alpha"] = 0.75
    cfg["training"]["label_smoothing"] = LABEL_SMOOTH
    cfg["data"]["img_size"] = IMG_SIZE

    device = torch.device("cuda:0")
    log.info(
        f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info(0)[0]/1e9:.1f}GB free)"
    )

    # ── Data ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  RETFOUND + MLP HEAD FINE-TUNING")
    log.info("=" * 60)

    dm = RetinalDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    nc = len(dm.disease_columns)
    cfg["model"]["num_classes"] = nc
    img_dirs = [str(dm.train_dataset.img_dir), str(dm.val_dataset.img_dir)]
    train_tfm = get_train_transforms(cfg)
    val_tfm = get_val_transforms(cfg)

    def to_df(ds):
        df = pd.DataFrame(ds.labels_array, columns=ds.disease_columns)
        df["ID"] = ds.image_ids
        return df

    train_ds = PreCachedDataset(
        to_df(dm.train_dataset), img_dirs, dm.disease_columns, train_tfm, IMG_SIZE
    )
    val_ds = PreCachedDataset(
        to_df(dm.val_dataset), img_dirs, dm.disease_columns, val_tfm, IMG_SIZE
    )
    train_ld = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )
    log.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {nc}")

    # ── Model ──────────────────────────────────────────────────────────
    model = RETFoundMLP(num_classes=nc, drop_path=DROP_PATH, head_dropout=HEAD_DROPOUT).to(device)
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    head_p = sum(p.numel() for p in model.head.parameters()) / 1e6
    log.info(f"  Model: RETFound+MLP | {total_p:.1f}M total | head={head_p:.2f}M")

    # Differential LR
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("encoder"):
            backbone_params.append(param)
        else:
            head_params.append(param)
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    log.info(
        f"  DiffLR: backbone={BACKBONE_LR:.0e}({len(backbone_params)}p) head={HEAD_LR:.0e}({len(head_params)}p)"
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")
    pw = train_ds.get_pos_weights().to(device)
    crit = build_loss(cfg, pos_weight=pw)
    ema = ModelEMA(model, decay=0.9998)
    es = AdvancedEarlyStopping(patience=PATIENCE, min_delta=0.001, min_epochs=8, mode="max")
    met = MetricTracker()
    best_f1 = 0
    best_state = None
    best_metrics = {}

    # ── Training loop ──────────────────────────────────────────────────
    log.info(f"  Epochs: {EPOCHS} | BS: {BATCH_SIZE} | Patience: {PATIENCE}")
    log.info(f"  Loss: focal(α=0.75,γ=2) | WD: {WEIGHT_DECAY} | DropPath: {DROP_PATH}")
    log.info("-" * 60)

    for ep in range(EPOCHS):
        # Warmup
        if ep < WARMUP_EPOCHS:
            s = (ep + 1) / WARMUP_EPOCHS
            opt.param_groups[0]["lr"] = BACKBONE_LR * s
            opt.param_groups[1]["lr"] = HEAD_LR * s

        model.train()
        tloss = 0
        steps = 0
        for imgs, tgts in train_ld:
            imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = crit(model(imgs), tgts)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            tloss += loss.item()
            steps += 1
        if ep >= WARMUP_EPOCHS:
            sched.step()

        # Validate with EMA
        orig = deepcopy(model.state_dict())
        ema.apply(model)
        model.eval()
        vloss = 0
        vs = 0
        with torch.no_grad():
            for imgs, tgts in val_ld:
                imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lo = model(imgs)
                vloss += crit(lo, tgts).item()
                vs += 1
                met.update(lo, tgts)
        model.load_state_dict(orig)
        vm = met.compute()
        met.reset()
        f1 = vm.get("f1_macro", 0)
        auc = vm.get("auc_roc", 0)
        prec = vm.get("precision_macro", 0)
        rec = vm.get("recall_macro", 0)
        vm.get("accuracy_sample", 0)
        mAP = vm.get("mAP", 0)
        lr_bb = opt.param_groups[0]["lr"]
        lr_hd = opt.param_groups[1]["lr"]
        log.info(
            f"  E{ep+1}/{EPOCHS} | L:{tloss/max(steps,1):.4f}/{vloss/max(vs,1):.4f} | "
            f"F1:{f1:.4f} AUC:{auc:.4f} mAP:{mAP:.4f} P:{prec:.4f} R:{rec:.4f} "
            f"lr:{lr_bb:.1e}/{lr_hd:.1e}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_state = deepcopy(ema.state_dict())
            best_metrics = vm
            log.info(f"    ** New best F1={f1:.4f} AUC={auc:.4f} mAP={mAP:.4f} **")
        stop, _ = es(ep, {"f1": f1, "auc": auc, "loss": vloss / max(vs, 1)})
        if stop:
            log.info(f"  Early stopped @ epoch {ep+1}")
            break

    # Load best EMA weights
    if best_state:
        model.load_state_dict(best_state)

    # ── Threshold Optimization ─────────────────────────────────────────
    log.info("\n  Optimizing per-class thresholds...")
    model.eval()
    met.reset()
    with torch.no_grad():
        for imgs, tgts in val_ld:
            imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                met.update(model(imgs), tgts)

    optimal_thresholds = met.optimize_thresholds()
    opt_metrics = met.compute(threshold=optimal_thresholds)
    fixed_metrics = met.compute(threshold=0.5)

    log.info(
        f"  Fixed (0.5):     F1={fixed_metrics['f1_macro']:.4f} AUC={fixed_metrics.get('auc_roc',0):.4f} "
        f"mAP={fixed_metrics.get('mAP',0):.4f} P={fixed_metrics.get('precision_macro',0):.4f} "
        f"R={fixed_metrics.get('recall_macro',0):.4f}"
    )
    log.info(
        f"  Optimized:       F1={opt_metrics['f1_macro']:.4f} AUC={opt_metrics.get('auc_roc',0):.4f} "
        f"mAP={opt_metrics.get('mAP',0):.4f} P={opt_metrics.get('precision_macro',0):.4f} "
        f"R={opt_metrics.get('recall_macro',0):.4f}"
    )
    log.info(
        f"  Thresholds:      [{optimal_thresholds.min():.3f}, {optimal_thresholds.max():.3f}] "
        f"mean={optimal_thresholds.mean():.3f}"
    )

    best_f1 = opt_metrics["f1_macro"]
    best_metrics = opt_metrics

    # ── Save ───────────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT / "best_retfound_mlp.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": nc,
            "model_name": "retfound_mlp",
            "backbone": BACKBONE,
            "best_f1": best_f1,
            "metrics": {k: float(v) for k, v in best_metrics.items()},
            "optimal_thresholds": optimal_thresholds.tolist(),
            "img_size": IMG_SIZE,
        },
        ckpt_path,
    )
    log.info(f"\n  Checkpoint saved: {ckpt_path}")

    # ── Summary ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  FINE-TUNING COMPLETE")
    log.info("=" * 60)
    print("\n  Model:        RETFound + MLP head")
    print(f"  Backbone:     {BACKBONE} ({total_p-head_p:.1f}M params)")
    print(f"  Head:         2-layer MLP ({head_p:.2f}M params)")
    print(f"  Best F1:      {best_f1:.4f}")
    print(f"  AUC:          {best_metrics.get('auc_roc', 0):.4f}")
    print(f"  mAP:          {best_metrics.get('mAP', 0):.4f}")
    print(f"  Precision:    {best_metrics.get('precision_macro', 0):.4f}")
    print(f"  Recall:       {best_metrics.get('recall_macro', 0):.4f}")
    print(f"  Thresholds:   [{optimal_thresholds.min():.3f}, {optimal_thresholds.max():.3f}]")
    print(f"  Checkpoint:   {ckpt_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()
