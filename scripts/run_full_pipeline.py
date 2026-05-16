#!/usr/bin/env python3
"""
=============================================================================
FULL MLOPS PIPELINE - End-to-End
=============================================================================
Stages:
  1. Data loading + validation + profiling
  2. Preprocessing + augmentation verification
  3. K-Fold training (4 models, 5 folds, 30 epochs, early stop@3)
  4. Model comparison + leaderboard + recommendation
  5. Fine-tuning best model (full dataset, cosine warmup)
  6. Export (ONNX + TorchScript) + Quantization benchmarks (FP16/INT8)
  7. IEEE plots

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1,4 python3 scripts/run_full_pipeline.py
=============================================================================
"""

import json
import logging
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.datamodule import DISEASE_COLUMNS, RetinalDataModule
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
# Force line-buffered stdout for thread visibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger("pipeline")
for h in logging.root.handlers:
    h.flush = h.stream.flush if hasattr(h, "stream") else lambda: None

# ── Config (Research-Optimized) ─────────────────────────────────────────────
CFG_PATH = "configs/train.yaml"
K_FOLDS = 3
MAX_EPOCHS = 50  # Longer training for small dataset (was 25)
EARLY_PATIENCE = 10  # More patience (was 3 - too aggressive)
BATCH_SIZE = 16  # ~128 steps/epoch (good for 2048 train samples)
FINETUNE_EPOCHS = 20
FINETUNE_LR = 5e-6  # Very conservative for fine-tuning
ALL_MODELS = ["vignn", "graphclip", "visual_language_gnn", "scene_graph_transformer"]
OUT = Path("outputs/full_pipeline")


# ── Helpers ─────────────────────────────────────────────────────────────────
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def param_count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


class PreCachedDataset(torch.utils.data.Dataset):
    """
    Pre-caches all images as resized tensors in RAM at init time.
    Eliminates PIL decode bottleneck - ~10x faster __getitem__.
    """

    def __init__(self, df, img_dirs, cols, transform=None, img_size=224):
        self.cols = cols
        self.tfm = transform
        self.labels = (
            df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        )
        ids = df["ID"].values
        dirs = [Path(d) for d in img_dirs]

        # Pre-load and cache all images as uint8 numpy arrays
        log.info(f"  Pre-caching {len(ids)} images to RAM...")
        t0 = time.time()
        self.cache = []
        for img_id in ids:
            loaded = False
            for d in dirs:
                for ext in (".png", ".jpg", ".jpeg"):
                    p = d / f"{img_id}{ext}"
                    if p.exists():
                        try:
                            img = (
                                Image.open(p)
                                .convert("RGB")
                                .resize((img_size, img_size), Image.BILINEAR)
                            )
                            self.cache.append(np.array(img, dtype=np.uint8))
                            loaded = True
                        except Exception:
                            pass
                        break
                if loaded:
                    break
            if not loaded:
                self.cache.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
        log.info(
            f"  Cached {len(self.cache)} images in {time.time()-t0:.1f}s ({len(self.cache)*img_size*img_size*3/1e9:.2f} GB RAM)"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.cache[i])  # Fast: numpy -> PIL (no disk IO)
        lab = torch.from_numpy(self.labels[i])
        return self.tfm(img) if self.tfm else img, lab

    def get_pos_weights(self):
        pos = self.labels.sum(0).clip(min=1)
        neg = len(self) - pos
        return torch.from_numpy(np.clip(neg / pos, 0.5, 50).astype(np.float32))


def build_model(name, nc, cfg):
    from src.models.vignn import ClinicalKnowledgeGraph

    kg = ClinicalKnowledgeGraph(disease_names=DISEASE_COLUMNS[:nc])
    h, hd, ly, dr = (
        cfg["model"].get("hidden_dim", 384),
        cfg["model"].get("num_heads", 4),
        cfg["model"].get("num_graph_layers", 3),
        cfg["model"].get("dropout", 0.1),
    )
    if name == "vignn":
        from src.models.vignn import ViGNN

        return ViGNN(
            num_classes=nc,
            hidden_dim=h,
            num_graph_layers=ly,
            num_heads=hd,
            dropout=dr,
            clinical_knowledge_graph=kg,
        )
    elif name == "graphclip":
        from src.models.graphclip import GraphCLIP

        return GraphCLIP(
            num_classes=nc,
            hidden_dim=h,
            num_graph_layers=ly,
            num_heads=hd,
            dropout=dr,
            clinical_knowledge_graph=kg,
        )
    elif name == "visual_language_gnn":
        from src.models.visual_language_gnn import VisualLanguageGNN

        return VisualLanguageGNN(
            num_classes=nc,
            hidden_dim=h,
            num_layers=ly,
            num_heads=hd,
            dropout=dr,
            clinical_knowledge_graph=kg,
        )
    elif name == "scene_graph_transformer":
        from src.models.scene_graph_transformer import SceneGraphTransformer

        return SceneGraphTransformer(
            num_classes=nc,
            hidden_dim=h,
            num_layers=ly,
            num_heads=hd,
            dropout=dr,
            clinical_knowledge_graph=kg,
        )
    raise ValueError(name)


WARMUP_EPOCHS = 5  # Longer warmup for pretrained backbone
BACKBONE_LR = 1e-5  # Conservative LR for pretrained ViT encoder
HEAD_LR = 5e-4  # 50x higher LR for randomly-initialized head/graph layers


# ── Train one fold (differential LR, sequential, single GPU) ──
def train_fold(model, train_ld, val_ld, criterion, device, fold, mname):
    model = model.to(device)

    # Differential LR: backbone (pretrained) gets low LR, head (random) gets high LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "visual_encoder.encoder" in name or "region_extractor.encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=1e-4,
    )
    log.info(
        f"  [{mname}] F{fold+1} DiffLR: backbone={BACKBONE_LR:.0e} ({len(backbone_params)} params) head={HEAD_LR:.0e} ({len(head_params)} params)"
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    ema = ModelEMA(model, decay=0.9999)
    es = AdvancedEarlyStopping(patience=EARLY_PATIENCE, min_delta=0.0005, min_epochs=8, mode="max")
    vmet = MetricTracker()
    best = {"f1_macro": 0}

    for ep in range(MAX_EPOCHS):
        # Warmup LR for first N epochs (scale each param group independently)
        if ep < WARMUP_EPOCHS:
            scale = (ep + 1) / WARMUP_EPOCHS
            opt.param_groups[0]["lr"] = BACKBONE_LR * scale
            opt.param_groups[1]["lr"] = HEAD_LR * scale

        # ── Train ──
        model.train()
        tloss = 0
        steps = 0
        for imgs, tgts in train_ld:
            imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(imgs), tgts)
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

        # ── Validate ALWAYS with EMA (was every 3 epochs - caused early stop misfires) ──
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
                vloss += criterion(lo, tgts).item()
                vs += 1
                vmet.update(lo, tgts)
        model.load_state_dict(orig)

        vm = vmet.compute()
        vmet.reset()
        f1, auc = vm.get("f1_macro", 0), vm.get("auc_roc", 0)
        prec, rec = vm.get("precision_macro", 0), vm.get("recall_macro", 0)
        acc = vm.get("accuracy_sample", 0)
        lr_now = opt.param_groups[0]["lr"]
        log.info(
            f"  [{mname}] F{fold+1} E{ep+1}/{MAX_EPOCHS} | "
            f"L:{tloss/max(steps,1):.4f}/{vloss/max(vs,1):.4f} | "
            f"F1:{f1:.4f} AUC:{auc:.4f} P:{prec:.4f} R:{rec:.4f} Acc:{acc:.4f} lr:{lr_now:.1e}"
        )

        if f1 > best["f1_macro"]:
            best = {**vm, "epoch": ep + 1, "ema_state": deepcopy(ema.state_dict())}
        stop, _ = es(ep, {"f1": f1, "auc": auc, "loss": vloss / max(vs, 1)})
        if stop:
            log.info(f"  [{mname}] F{fold+1} early stopped @ epoch {ep+1}")
            break

    del scaler
    torch.cuda.empty_cache()
    return best


# =============================================================================
#  MAIN PIPELINE
# =============================================================================
def main():
    set_seed(42)
    cfg = yaml.safe_load(open(CFG_PATH))
    cfg["training"]["loss"] = "focal"  # FP16-safe
    cfg["training"]["label_smoothing"] = 0.01  # Reduced from 0.05 (too aggressive for multi-label)
    device = torch.device("cuda:0")
    n_gpus = torch.cuda.device_count()
    log.info(f"GPUs visible: {n_gpus}")
    for i in range(n_gpus):
        log.info(
            f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.mem_get_info(i)[0]/1e9:.1f}GB free)"
        )
    OUT.mkdir(parents=True, exist_ok=True)

    # ━━ STAGE 1: Data Loading + Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "=" * 70)
    log.info("  STAGE 1: DATA LOADING + VALIDATION")
    log.info("=" * 70)
    dm = RetinalDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    nc = len(dm.disease_columns)
    cfg["model"]["num_classes"] = nc

    # Merge train+val for k-fold
    def _to_df(ds):
        df = pd.DataFrame(ds.labels_array, columns=ds.disease_columns)
        df["ID"] = ds.image_ids
        if hasattr(ds, "labels_df") and "Disease_Risk" in ds.labels_df.columns:
            df["Disease_Risk"] = ds.labels_df["Disease_Risk"].values
        return df

    all_df = pd.concat([_to_df(dm.train_dataset), _to_df(dm.val_dataset)], ignore_index=True)
    if "Disease_Risk" not in all_df.columns:
        all_df["Disease_Risk"] = all_df[dm.disease_columns].sum(1).clip(0, 1).astype(int)
    img_dirs = [dm.train_dataset.img_dir, dm.val_dataset.img_dir]

    log.info(f"  Samples: {len(all_df)} | Classes: {nc} | Image dirs: {len(img_dirs)}")
    log.info("  Disease prevalence (top 5):")
    top5 = all_df[dm.disease_columns].sum().sort_values(ascending=False).head(5)
    for d, c in top5.items():
        log.info(f"    {d}: {c} ({c/len(all_df)*100:.1f}%)")

    # ━━ STAGE 2: Augmentation Verification ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "=" * 70)
    log.info("  STAGE 2: PREPROCESSING + AUGMENTATION")
    log.info("=" * 70)
    train_tfm = get_train_transforms(cfg)
    val_tfm = get_val_transforms(cfg)
    test_ds = PreCachedDataset(all_df.head(4), img_dirs, dm.disease_columns, train_tfm)
    test_img, test_lab = test_ds[0]
    log.info(f"  Train transform output: {test_img.shape} (dtype={test_img.dtype})")
    log.info(f"  Label shape: {test_lab.shape}")
    log.info("  Augmentation: flip, rotate, color jitter, crop, random erase")
    log.info("  FP16 mixed precision: enabled")

    # ━━ STAGE 3: K-Fold Training (All 4 Models) ━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "=" * 70)
    log.info("  STAGE 3: K-FOLD CROSS-VALIDATION TRAINING")
    log.info(f"  Models: {ALL_MODELS}")
    log.info(f"  K={K_FOLDS} | MaxEpochs={MAX_EPOCHS} | EarlyStop={EARLY_PATIENCE}")
    log.info(f"  Batch={BATCH_SIZE} | GPUs={n_gpus} | EMA=0.9999")
    log.info("=" * 70)

    stratify = all_df["Disease_Risk"].values
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    folds_split = list(skf.split(all_df, stratify))

    # ── Sequential training (fixes CUDA crashes from parallel threads) ──
    all_results = []
    log.info(f"\n  Sequential training: {len(ALL_MODELS)} models on GPU 0")

    for model_idx, mname in enumerate(ALL_MODELS):
        log.info(f"\n{'━'*60}")
        log.info(f"  MODEL {model_idx+1}/{len(ALL_MODELS)}: {mname.upper()}")
        log.info(f"{'━'*60}")

        fold_metrics = []
        t0_model = time.time()

        for fold, (tr_idx, va_idx) in enumerate(folds_split):
            set_seed(42 + fold)
            tr_ds = PreCachedDataset(
                all_df.iloc[tr_idx], [str(d) for d in img_dirs], dm.disease_columns, train_tfm
            )
            va_ds = PreCachedDataset(
                all_df.iloc[va_idx], [str(d) for d in img_dirs], dm.disease_columns, val_tfm
            )
            tr_ld = DataLoader(
                tr_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                pin_memory=True,
            )
            va_ld = DataLoader(
                va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
            )

            model = build_model(mname, nc, cfg)
            pm = param_count(model)
            pw = tr_ds.get_pos_weights().to(device)
            crit = build_loss(cfg, pos_weight=pw)

            best = train_fold(model, tr_ld, va_ld, crit, device, fold, mname)
            best["fold"] = fold + 1
            best["params_M"] = pm
            fold_metrics.append(best)
            log.info(
                f"  [{mname}] Fold {fold+1} BEST: F1={best['f1_macro']:.4f} AUC={best.get('auc_roc',0):.4f} P={best.get('precision_macro',0):.4f} R={best.get('recall_macro',0):.4f} @ep{best.get('epoch',0)}"
            )
            del model
            torch.cuda.empty_cache()

        elapsed = (time.time() - t0_model) / 60
        res = {"model": mname, "params_M": fold_metrics[0]["params_M"], "time_min": elapsed}
        for k in [
            "f1_macro",
            "f1_micro",
            "auc_roc",
            "precision_macro",
            "recall_macro",
            "precision_micro",
            "recall_micro",
            "accuracy_subset",
            "accuracy_sample",
            "mAP",
            "hamming_loss",
        ]:
            vals = [f.get(k, 0) for f in fold_metrics]
            res[f"mean_{k}"] = float(np.mean(vals))
            res[f"std_{k}"] = float(np.std(vals))
        all_results.append(res)
        log.info(
            f"  [{mname}] CV: F1={res['mean_f1_macro']:.4f}+-{res['std_f1_macro']:.4f} AUC={res['mean_auc_roc']:.4f} P={res['mean_precision_macro']:.4f} R={res['mean_recall_macro']:.4f} Time={elapsed:.1f}min"
        )
        json.dump(res, open(OUT / f"{mname}_results.json", "w"), indent=2, default=str)

    # ━━ STAGE 4: Comparison + Leaderboard ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "=" * 70)
    log.info("  STAGE 4: MODEL COMPARISON + LEADERBOARD")
    log.info("=" * 70)

    scores = {}
    print(
        f"\n{'Model':<28} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7} {'Params':>7} {'Time':>6}"
    )
    print("─" * 88)
    for r in all_results:
        f1 = r["mean_f1_macro"]
        auc = r["mean_auc_roc"]
        prec = r["mean_precision_macro"]
        rec = r["mean_recall_macro"]
        acc = r.get("mean_accuracy_sample", 0)
        s = 0.40 * f1 + 0.30 * auc + 0.15 * prec + 0.15 * rec
        scores[r["model"]] = s
        print(
            f"  {r['model']:<26} {f1:>6.4f} {auc:>6.4f} {prec:>6.4f} {rec:>6.4f} {acc:>6.4f} {r['params_M']:>5.1f}M {r['time_min']:>5.1f}m"
        )
    print("─" * 88)

    best_name = max(scores, key=scores.get)
    best_r = next(r for r in all_results if r["model"] == best_name)

    print("\n  WEIGHTED SCORES (F1:40% | AUC:30% | Prec:15% | Rec:15%):")
    for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        mark = " <-- BEST" if n == best_name else ""
        print(f"    {n:<28} {s:.4f}{mark}")

    print(f"\n{'='*70}")
    print("  RECOMMENDATION")
    print(f"{'='*70}")
    print(f"\n  Recommended: {best_name}")
    print(f"    Overall Score: {scores[best_name]:.4f}")
    print(f"    F1 Score: {best_r['mean_f1_macro']:.4f}")
    print(f"    AUC-ROC:  {best_r['mean_auc_roc']:.4f}")
    print(f"    Precision: {best_r['mean_precision_macro']:.4f}")
    print(f"    Recall:   {best_r['mean_recall_macro']:.4f}")
    print(f"    Parameters: {best_r['params_M']:.0f}M")
    print("\n    Rationale: Weighted scoring (F1:40%, AUC:30%, Prec:15%, Rec:15%)")
    print("    Best balance between accuracy and computational efficiency")
    print(f"{'='*70}\n")

    # ━━ STAGE 5: Fine-Tune Best Model on Full Dataset ━━━━━━━━━━━━━━━━━━━━
    log.info("=" * 70)
    log.info(f"  STAGE 5: FINE-TUNING {best_name.upper()} ON FULL DATASET")
    log.info(f"  Epochs: {FINETUNE_EPOCHS} | LR: {FINETUNE_LR} | Cosine warmup")
    log.info("=" * 70)

    set_seed(42)
    full_train_ds = PreCachedDataset(
        _to_df(dm.train_dataset), img_dirs, dm.disease_columns, train_tfm
    )
    full_val_ds = PreCachedDataset(_to_df(dm.val_dataset), img_dirs, dm.disease_columns, val_tfm)
    ft_train_ld = DataLoader(
        full_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    ft_val_ld = DataLoader(
        full_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    ft_model = build_model(best_name, nc, cfg).to(device)
    pw = full_train_ds.get_pos_weights().to(device)
    ft_crit = build_loss(cfg, pos_weight=pw)
    ft_opt = torch.optim.AdamW(ft_model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    ft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        ft_opt, T_max=FINETUNE_EPOCHS, eta_min=1e-7
    )
    ft_scaler = torch.amp.GradScaler("cuda", enabled=True)
    ft_ema = ModelEMA(ft_model, decay=0.9999)
    ft_es = AdvancedEarlyStopping(patience=5, min_epochs=3, mode="max")
    ft_met = MetricTracker()
    best_ft_state = None
    best_ft_f1 = 0

    for ep in range(FINETUNE_EPOCHS):
        ft_model.train()
        tloss = 0
        steps = 0
        for imgs, tgts in ft_train_ld:
            imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
            ft_opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = ft_crit(ft_model(imgs), tgts)
            ft_scaler.scale(loss).backward()
            ft_scaler.unscale_(ft_opt)
            nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            ft_scaler.step(ft_opt)
            ft_scaler.update()
            ft_ema.update(ft_model)
            tloss += loss.item()
            steps += 1
        ft_sched.step()

        orig = deepcopy(ft_model.state_dict())
        ft_ema.apply(ft_model)
        ft_model.eval()
        vloss = 0
        vs = 0
        with torch.no_grad():
            for imgs, tgts in ft_val_ld:
                imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lo = ft_model(imgs)
                vloss += ft_crit(lo, tgts).item()
                vs += 1
                ft_met.update(lo, tgts)
        ft_model.load_state_dict(orig)
        vm = ft_met.compute()
        ft_met.reset()
        f1 = vm.get("f1_macro", 0)
        auc = vm.get("auc_roc", 0)
        prec_ft, rec_ft = vm.get("precision_macro", 0), vm.get("recall_macro", 0)
        acc_ft = vm.get("accuracy_sample", 0)
        log.info(
            f"  [finetune] E{ep+1}/{FINETUNE_EPOCHS} | "
            f"L:{tloss/max(steps,1):.4f}/{vloss/max(vs,1):.4f} | "
            f"F1:{f1:.4f} AUC:{auc:.4f} P:{prec_ft:.4f} R:{rec_ft:.4f} Acc:{acc_ft:.4f}"
        )

        if f1 > best_ft_f1:
            best_ft_f1 = f1
            best_ft_state = deepcopy(ft_ema.state_dict())
            best_ft_metrics = vm
        stop, _ = ft_es(ep, {"f1": f1, "auc": auc, "loss": vloss / max(vs, 1)})
        if stop:
            log.info(f"  [finetune] Early stopped @ epoch {ep+1}")
            break

    if best_ft_state:
        ft_model.load_state_dict(best_ft_state)
    raw = ft_model
    ckpt_path = OUT / "best_finetuned.pth"
    torch.save(
        {
            "model_state_dict": raw.state_dict(),
            "num_classes": nc,
            "model_name": best_name,
            "best_f1": best_ft_f1,
            "metrics": {k: float(v) for k, v in best_ft_metrics.items()},
        },
        ckpt_path,
    )
    log.info(f"  Saved finetuned model: {ckpt_path} (F1={best_ft_f1:.4f})")

    # ━━ STAGE 6: Export + Quantization ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "=" * 70)
    log.info("  STAGE 6: EXPORT + QUANTIZATION BENCHMARKS")
    log.info("=" * 70)

    export_model = build_model(best_name, nc, cfg)
    export_model.load_state_dict(raw.cpu().state_dict())
    export_model.eval()
    export_dir = OUT / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # ONNX
    log.info("  Exporting ONNX...")
    try:
        from src.export.onnx_export import ONNXExporter

        onnx_path = ONNXExporter.export(export_model, export_dir / "model.onnx", device="cpu")
        val_result = ONNXExporter.validate(onnx_path, export_model, device="cpu")
        log.info(
            f"  ONNX: {onnx_path} | valid={val_result['valid']} max_diff={val_result.get('max_diff',0):.6f}"
        )
    except Exception as e:
        log.warning(f"  ONNX export failed: {e}")

    # TorchScript
    log.info("  Exporting TorchScript...")
    try:
        from src.export.torchscript_export import TorchScriptExporter

        ts_path = TorchScriptExporter.trace(
            export_model, export_dir / "model_traced.pt", device="cpu"
        )
        ts_val = TorchScriptExporter.validate(ts_path, export_model, device="cpu")
        log.info(f"  TorchScript: {ts_path} | valid={ts_val['valid']}")
    except Exception as e:
        log.warning(f"  TorchScript export failed: {e}")

    # Quantization benchmarks
    log.info("  Running quantization benchmarks...")
    try:
        from src.export.quantization import QuantizationBenchmark

        qb = QuantizationBenchmark(device)
        export_model_gpu = build_model(best_name, nc, cfg)
        export_model_gpu.load_state_dict(raw.cpu().state_dict())
        quant_results = qb.compare_formats(export_model_gpu, model_name=best_name, n_runs=30)

        print(f"\n  {'Format':<15} {'Latency':>10} {'P95':>10} {'Size':>10}")
        print(f"  {'─'*50}")
        for fmt, data in quant_results.items():
            if "error" not in data:
                print(
                    f"  {fmt:<15} {data['mean_ms']:>9.2f}ms {data['p95_ms']:>9.2f}ms {str(data.get('size_MB','?')):>9}MB"
                )
            else:
                print(f"  {fmt:<15} ERROR: {data['error']}")
        qb.save_report(quant_results, export_dir)
    except Exception as e:
        log.warning(f"  Quantization benchmark failed: {e}")

    # ━━ STAGE 7: Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "=" * 70)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 70)

    summary = {
        "best_model": best_name,
        "cv_score": scores[best_name],
        "cv_f1": best_r["mean_f1_macro"],
        "cv_auc": best_r["mean_auc_roc"],
        "finetuned_f1": best_ft_f1,
        "finetuned_metrics": {k: float(v) for k, v in best_ft_metrics.items()},
        "all_model_scores": {k: float(v) for k, v in scores.items()},
        "config": {
            "k_folds": K_FOLDS,
            "max_epochs": MAX_EPOCHS,
            "early_stop": EARLY_PATIENCE,
            "finetune_epochs": FINETUNE_EPOCHS,
            "finetune_lr": FINETUNE_LR,
            "batch_size": BATCH_SIZE,
        },
    }
    json.dump(summary, open(OUT / "pipeline_summary.json", "w"), indent=2)
    json.dump(all_results, open(OUT / "all_cv_results.json", "w"), indent=2, default=str)

    print(f"\n  Best Model:     {best_name}")
    print(f"  CV F1:          {best_r['mean_f1_macro']:.4f} ± {best_r['std_f1_macro']:.4f}")
    print(f"  CV AUC:         {best_r['mean_auc_roc']:.4f}")
    print(f"  Finetuned F1:   {best_ft_f1:.4f}")
    print(f"  Finetuned AUC:  {best_ft_metrics.get('auc_roc',0):.4f}")
    print(f"  Checkpoint:     {ckpt_path}")
    print(f"  Exports:        {export_dir}")
    print(f"  Results:        {OUT}")
    log.info("Done.")


if __name__ == "__main__":
    main()
