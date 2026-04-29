#!/usr/bin/env python3
"""
Train a single model with K-fold CV on a single GPU.
Designed to be launched multiple times in parallel on different GPUs.

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 -u scripts/train_single_model.py --model vignn
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python3 -u scripts/train_single_model.py --model graphclip
"""
import argparse, json, logging, os, random, sys, time
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

from src.data.datamodule import RetinalDataModule, DISEASE_COLUMNS
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.losses import build_loss
from src.training.metrics import MetricTracker
from src.training.early_stopping import AdvancedEarlyStopping
from src.training.ema import ModelEMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger("train")

# ── Config ──
K_FOLDS        = 3
MAX_EPOCHS     = 25
EARLY_PATIENCE = 3
BATCH_SIZE     = 32
WARMUP_EPOCHS  = 5
BACKBONE_LR    = 1e-5
HEAD_LR        = 5e-4
OUT            = Path("outputs/full_pipeline")

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def param_count(m): return sum(p.numel() for p in m.parameters()) / 1e6

class PreCachedDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dirs, cols, transform=None, img_size=224):
        self.cols = cols; self.tfm = transform
        self.labels = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        ids = df["ID"].values; dirs = [Path(d) for d in img_dirs]
        log.info(f"  Pre-caching {len(ids)} images...")
        t0 = time.time(); self.cache = []
        for img_id in ids:
            loaded = False
            for d in dirs:
                for ext in (".png", ".jpg", ".jpeg"):
                    p = d / f"{img_id}{ext}"
                    if p.exists():
                        try:
                            self.cache.append(np.array(Image.open(p).convert("RGB").resize((img_size, img_size), Image.BILINEAR), dtype=np.uint8))
                            loaded = True
                        except: pass
                        break
                if loaded: break
            if not loaded:
                self.cache.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
        log.info(f"  Cached {len(self.cache)} in {time.time()-t0:.1f}s")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        img = Image.fromarray(self.cache[i])
        return (self.tfm(img) if self.tfm else img), torch.from_numpy(self.labels[i])
    def get_pos_weights(self):
        pos = self.labels.sum(0).clip(min=1); neg = len(self) - pos
        return torch.from_numpy(np.clip(neg/pos, 0.5, 50).astype(np.float32))

def build_model(name, nc, cfg):
    from src.models.vignn import ClinicalKnowledgeGraph
    kg = ClinicalKnowledgeGraph(disease_names=DISEASE_COLUMNS[:nc])
    h = cfg["model"].get("hidden_dim", 384)
    hd = cfg["model"].get("num_heads", 4)
    ly = cfg["model"].get("num_graph_layers", 3)
    dr = cfg["model"].get("dropout", 0.1)
    if name == "vignn":
        from src.models.vignn import ViGNN
        return ViGNN(num_classes=nc, hidden_dim=h, num_graph_layers=ly, num_heads=hd, dropout=dr, clinical_knowledge_graph=kg)
    elif name == "graphclip":
        from src.models.graphclip import GraphCLIP
        return GraphCLIP(num_classes=nc, hidden_dim=h, num_graph_layers=ly, num_heads=hd, dropout=dr, clinical_knowledge_graph=kg)
    elif name == "visual_language_gnn":
        from src.models.visual_language_gnn import VisualLanguageGNN
        return VisualLanguageGNN(num_classes=nc, hidden_dim=h, num_layers=ly, num_heads=hd, dropout=dr, clinical_knowledge_graph=kg)
    elif name == "scene_graph_transformer":
        from src.models.scene_graph_transformer import SceneGraphTransformer
        return SceneGraphTransformer(num_classes=nc, hidden_dim=h, num_layers=ly, num_heads=hd, dropout=dr, clinical_knowledge_graph=kg)
    raise ValueError(name)

def train_fold(model, train_ld, val_ld, criterion, device, fold, mname):
    model = model.to(device)
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if "visual_encoder.encoder" in name or "region_extractor.encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=1e-4)
    log.info(f"  [{mname}] F{fold+1} DiffLR backbone={BACKBONE_LR:.0e}({len(backbone_params)}p) head={HEAD_LR:.0e}({len(head_params)}p)")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")
    ema = ModelEMA(model, decay=0.9999)
    es = AdvancedEarlyStopping(patience=EARLY_PATIENCE, min_delta=0.0005, min_epochs=8, mode="max")
    vmet = MetricTracker(); best = {"f1_macro": 0}

    for ep in range(MAX_EPOCHS):
        if ep < WARMUP_EPOCHS:
            s = (ep + 1) / WARMUP_EPOCHS
            opt.param_groups[0]["lr"] = BACKBONE_LR * s
            opt.param_groups[1]["lr"] = HEAD_LR * s
        model.train(); tloss = 0; steps = 0
        for imgs, tgts in train_ld:
            imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(imgs), tgts)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); ema.update(model)
            tloss += loss.item(); steps += 1
        if ep >= WARMUP_EPOCHS: sched.step()

        orig = deepcopy(model.state_dict()); ema.apply(model); model.eval()
        vloss = 0; vs = 0
        with torch.no_grad():
            for imgs, tgts in val_ld:
                imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lo = model(imgs)
                vloss += criterion(lo, tgts).item(); vs += 1; vmet.update(lo, tgts)
        model.load_state_dict(orig)
        vm = vmet.compute(); vmet.reset()
        f1, auc = vm.get("f1_macro", 0), vm.get("auc_roc", 0)
        prec, rec, acc = vm.get("precision_macro", 0), vm.get("recall_macro", 0), vm.get("accuracy_sample", 0)
        lr_now = opt.param_groups[1]["lr"]
        log.info(f"  [{mname}] F{fold+1} E{ep+1}/{MAX_EPOCHS} | L:{tloss/max(steps,1):.4f}/{vloss/max(vs,1):.4f} | F1:{f1:.4f} AUC:{auc:.4f} P:{prec:.4f} R:{rec:.4f} Acc:{acc:.4f} lr:{lr_now:.1e}")
        if f1 > best["f1_macro"]:
            best = {**vm, "epoch": ep + 1, "ema_state": deepcopy(ema.state_dict())}
        stop, _ = es(ep, {"f1": f1, "auc": auc, "loss": vloss / max(vs, 1)})
        if stop:
            log.info(f"  [{mname}] F{fold+1} early stopped @ epoch {ep+1}"); break
    del scaler; torch.cuda.empty_cache()
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["vignn", "graphclip", "visual_language_gnn", "scene_graph_transformer"])
    args = parser.parse_args()
    mname = args.model

    set_seed(42)
    cfg = yaml.safe_load(open("configs/train.yaml"))
    cfg["training"]["loss"] = "focal"
    cfg["training"]["label_smoothing"] = 0.01
    device = torch.device("cuda:0")
    log.info(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info(0)[0]/1e9:.1f}GB free)")

    dm = RetinalDataModule(cfg); dm.prepare_data(); dm.setup("fit")
    nc = len(dm.disease_columns); cfg["model"]["num_classes"] = nc
    def to_df(ds):
        df = pd.DataFrame(ds.labels_array, columns=ds.disease_columns)
        df["ID"] = ds.image_ids
        df["Disease_Risk"] = df[ds.disease_columns].sum(1).clip(0,1).astype(int)
        return df
    all_df = pd.concat([to_df(dm.train_dataset), to_df(dm.val_dataset)], ignore_index=True)
    img_dirs = [str(dm.train_dataset.img_dir), str(dm.val_dataset.img_dir)]
    train_tfm = get_train_transforms(cfg); val_tfm = get_val_transforms(cfg)

    log.info(f"\n{'='*60}")
    log.info(f"  {mname.upper()} | K={K_FOLDS} | MaxEp={MAX_EPOCHS} | BS={BATCH_SIZE}")
    log.info(f"  DiffLR: backbone={BACKBONE_LR:.0e} head={HEAD_LR:.0e} | Warmup={WARMUP_EPOCHS}")
    log.info(f"{'='*60}")

    stratify = all_df["Disease_Risk"].values
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(all_df, stratify))
    fold_metrics = []; t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        set_seed(42 + fold)
        tr_ds = PreCachedDataset(all_df.iloc[tr_idx], img_dirs, dm.disease_columns, train_tfm)
        va_ds = PreCachedDataset(all_df.iloc[va_idx], img_dirs, dm.disease_columns, val_tfm)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        model = build_model(mname, nc, cfg)
        pm = param_count(model)
        pw = tr_ds.get_pos_weights().to(device)
        crit = build_loss(cfg, pos_weight=pw)
        best = train_fold(model, tr_ld, va_ld, crit, device, fold, mname)
        best["fold"] = fold + 1; best["params_M"] = pm
        fold_metrics.append(best)
        log.info(f"  [{mname}] Fold {fold+1} BEST: F1={best['f1_macro']:.4f} AUC={best.get('auc_roc',0):.4f} P={best.get('precision_macro',0):.4f} R={best.get('recall_macro',0):.4f} @ep{best.get('epoch',0)}")
        del model; torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 60
    res = {"model": mname, "params_M": fold_metrics[0]["params_M"], "time_min": elapsed}
    for k in ["f1_macro","f1_micro","auc_roc","precision_macro","recall_macro","accuracy_subset","accuracy_sample","mAP","hamming_loss"]:
        vals = [f.get(k, 0) for f in fold_metrics]
        res[f"mean_{k}"] = float(np.mean(vals)); res[f"std_{k}"] = float(np.std(vals))
    json.dump(res, open(OUT / f"{mname}_results.json", "w"), indent=2, default=str)
    log.info(f"\n{'='*60}")
    log.info(f"  {mname.upper()} CV COMPLETE")
    log.info(f"  F1:   {res['mean_f1_macro']:.4f} +/- {res['std_f1_macro']:.4f}")
    log.info(f"  AUC:  {res['mean_auc_roc']:.4f}")
    log.info(f"  Prec: {res['mean_precision_macro']:.4f}")
    log.info(f"  Rec:  {res['mean_recall_macro']:.4f}")
    log.info(f"  Acc:  {res.get('mean_accuracy_sample',0):.4f}")
    log.info(f"  Time: {elapsed:.1f} min")
    log.info(f"{'='*60}")

if __name__ == "__main__":
    main()
