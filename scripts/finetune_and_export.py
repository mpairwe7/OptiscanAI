#!/usr/bin/env python3
"""
Fine-tune the best model (SGT) on full dataset, then export to ONNX + TorchScript + quantization benchmarks.

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 python3 -u scripts/finetune_and_export.py
"""
import json, logging, os, random, sys, time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from src.data.datamodule import RetinalDataModule, DISEASE_COLUMNS
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.losses import build_loss
from src.training.metrics import MetricTracker
from src.training.early_stopping import AdvancedEarlyStopping
from src.training.ema import ModelEMA
from src.models.vignn import ClinicalKnowledgeGraph
from src.models.scene_graph_transformer import SceneGraphTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger("finetune")

# Config
BEST_MODEL     = "scene_graph_transformer"
BACKBONE       = "vit_large_patch16_224"  # RETFound retinal foundation model
FINETUNE_EPOCHS = 25
IMG_SIZE        = 224    # RETFound native resolution
BATCH_SIZE      = 16     # Reduced for ViT-L memory
BACKBONE_LR     = 5e-6   # Lower LR for large pretrained backbone
HEAD_LR         = 5e-4
WARMUP_EPOCHS   = 5
PATIENCE        = 8
OUT             = Path("outputs/full_pipeline")

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

def main():
    set_seed(42)
    cfg = yaml.safe_load(open("configs/train.yaml"))
    cfg["training"]["loss"] = "focal"
    cfg["training"]["focal_alpha"] = 0.75
    cfg["training"]["label_smoothing"] = 0.01
    cfg["data"]["img_size"] = IMG_SIZE
    device = torch.device("cuda:0")
    log.info(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info(0)[0]/1e9:.1f}GB free)")

    # ━━ STAGE 1: Load Data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "="*60)
    log.info("  FINE-TUNING SceneGraphTransformer ON FULL DATASET")
    log.info("="*60)

    dm = RetinalDataModule(cfg); dm.prepare_data(); dm.setup("fit")
    nc = len(dm.disease_columns); cfg["model"]["num_classes"] = nc
    img_dirs = [str(dm.train_dataset.img_dir), str(dm.val_dataset.img_dir)]
    train_tfm = get_train_transforms(cfg); val_tfm = get_val_transforms(cfg)

    def to_df(ds):
        df = pd.DataFrame(ds.labels_array, columns=ds.disease_columns)
        df["ID"] = ds.image_ids
        return df

    train_ds = PreCachedDataset(to_df(dm.train_dataset), img_dirs, dm.disease_columns, train_tfm, img_size=IMG_SIZE)
    val_ds = PreCachedDataset(to_df(dm.val_dataset), img_dirs, dm.disease_columns, val_tfm, img_size=IMG_SIZE)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    log.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {nc}")

    # ━━ STAGE 2: Build Model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    kg = ClinicalKnowledgeGraph(disease_names=DISEASE_COLUMNS[:nc])
    h = cfg["model"].get("hidden_dim", 384)
    model = SceneGraphTransformer(
        num_classes=nc, hidden_dim=h,
        num_layers=cfg["model"].get("num_graph_layers", 3),
        num_heads=cfg["model"].get("num_heads", 4),
        dropout=cfg["model"].get("dropout", 0.1),
        clinical_knowledge_graph=kg,
        backbone=BACKBONE,
        img_size=IMG_SIZE,
    ).to(device)
    log.info(f"  Model: {BEST_MODEL} | backbone={BACKBONE} | {param_count(model):.1f}M params")

    # Differential LR
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if "region_extractor.encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=1e-4)
    log.info(f"  DiffLR: backbone={BACKBONE_LR:.0e}({len(backbone_params)}p) head={HEAD_LR:.0e}({len(head_params)}p)")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")
    pw = train_ds.get_pos_weights().to(device)
    crit = build_loss(cfg, pos_weight=pw)
    ema = ModelEMA(model, decay=0.9999)
    es = AdvancedEarlyStopping(patience=PATIENCE, min_delta=0.0005, min_epochs=10, mode="max")
    met = MetricTracker()
    best_f1 = 0; best_state = None; best_metrics = {}

    # ━━ STAGE 3: Train ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info(f"  Epochs: {FINETUNE_EPOCHS} | BS: {BATCH_SIZE} | Patience: {PATIENCE}")
    log.info("-"*60)

    for ep in range(FINETUNE_EPOCHS):
        if ep < WARMUP_EPOCHS:
            s = (ep + 1) / WARMUP_EPOCHS
            opt.param_groups[0]["lr"] = BACKBONE_LR * s
            opt.param_groups[1]["lr"] = HEAD_LR * s

        model.train(); tloss = 0; steps = 0
        for imgs, tgts in train_ld:
            imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = crit(model(imgs), tgts)
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
                vloss += crit(lo, tgts).item(); vs += 1; met.update(lo, tgts)
        model.load_state_dict(orig)
        vm = met.compute(); met.reset()
        f1 = vm.get("f1_macro", 0); auc = vm.get("auc_roc", 0)
        prec = vm.get("precision_macro", 0); rec = vm.get("recall_macro", 0)
        acc = vm.get("accuracy_sample", 0)
        lr = opt.param_groups[1]["lr"]
        log.info(f"  E{ep+1}/{FINETUNE_EPOCHS} | L:{tloss/max(steps,1):.4f}/{vloss/max(vs,1):.4f} | F1:{f1:.4f} AUC:{auc:.4f} P:{prec:.4f} R:{rec:.4f} Acc:{acc:.4f} lr:{lr:.1e}")

        if f1 > best_f1:
            best_f1 = f1; best_state = deepcopy(ema.state_dict()); best_metrics = vm
            log.info(f"    ** New best F1={f1:.4f} **")
        stop, _ = es(ep, {"f1": f1, "auc": auc, "loss": vloss / max(vs, 1)})
        if stop:
            log.info(f"  Early stopped @ epoch {ep+1}"); break

    # Load best EMA weights
    if best_state: model.load_state_dict(best_state)

    # ━━ STAGE 3.5: Threshold Optimization ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n  Optimizing per-class thresholds on validation set...")
    model.eval(); met.reset()
    with torch.no_grad():
        for imgs, tgts in val_ld:
            imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(imgs)
            met.update(logits, tgts)

    optimal_thresholds = met.optimize_thresholds()
    opt_metrics = met.compute(threshold=optimal_thresholds)
    fixed_metrics = met.compute(threshold=0.5)

    log.info(f"  Fixed threshold (0.5):  F1={fixed_metrics['f1_macro']:.4f} AUC={fixed_metrics.get('auc_roc',0):.4f} P={fixed_metrics.get('precision_macro',0):.4f} R={fixed_metrics.get('recall_macro',0):.4f}")
    log.info(f"  Optimized thresholds:   F1={opt_metrics['f1_macro']:.4f} AUC={opt_metrics.get('auc_roc',0):.4f} P={opt_metrics.get('precision_macro',0):.4f} R={opt_metrics.get('recall_macro',0):.4f}")
    log.info(f"  Threshold range: [{optimal_thresholds.min():.3f}, {optimal_thresholds.max():.3f}] mean={optimal_thresholds.mean():.3f}")

    best_f1 = opt_metrics['f1_macro']
    best_metrics = opt_metrics

    # ━━ STAGE 4: Save Checkpoint ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT / "best_finetuned.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": nc,
        "model_name": BEST_MODEL,
        "best_f1": best_f1,
        "metrics": {k: float(v) for k, v in best_metrics.items()},
        "optimal_thresholds": optimal_thresholds.tolist(),
        "img_size": IMG_SIZE,
    }, ckpt_path)
    log.info(f"\n  Checkpoint saved: {ckpt_path} (F1={best_f1:.4f})")

    # ━━ STAGE 5: Export ONNX ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "="*60)
    log.info("  EXPORTING TO ONNX + TORCHSCRIPT")
    log.info("="*60)
    export_dir = OUT / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_model = SceneGraphTransformer(
        num_classes=nc, hidden_dim=h,
        num_layers=cfg["model"].get("num_graph_layers", 3),
        num_heads=cfg["model"].get("num_heads", 4),
        dropout=cfg["model"].get("dropout", 0.1),
        clinical_knowledge_graph=kg,
        backbone=BACKBONE,
        img_size=IMG_SIZE,
    )
    export_model.load_state_dict(model.cpu().state_dict())
    export_model.eval()

    # ONNX
    log.info("  Exporting ONNX...")
    try:
        from src.export.onnx_export import ONNXExporter
        onnx_path = ONNXExporter.export(export_model, export_dir / "sgt_model.onnx", device="cpu")
        val = ONNXExporter.validate(onnx_path, export_model, device="cpu")
        log.info(f"  ONNX: {onnx_path} | valid={val['valid']} | max_diff={val.get('max_diff',0):.6f}")
    except Exception as e:
        log.warning(f"  ONNX export failed: {e}")

    # TorchScript
    log.info("  Exporting TorchScript...")
    try:
        from src.export.torchscript_export import TorchScriptExporter
        ts_path = TorchScriptExporter.trace(export_model, export_dir / "sgt_model_traced.pt", device="cpu")
        ts_val = TorchScriptExporter.validate(ts_path, export_model, device="cpu")
        log.info(f"  TorchScript: {ts_path} | valid={ts_val['valid']}")
    except Exception as e:
        log.warning(f"  TorchScript export failed: {e}")

    # ━━ STAGE 6: Quantization Benchmarks ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "="*60)
    log.info("  QUANTIZATION BENCHMARKS")
    log.info("="*60)
    try:
        from src.export.quantization import QuantizationBenchmark
        qb = QuantizationBenchmark(device)
        quant_model = SceneGraphTransformer(
            num_classes=nc, hidden_dim=h,
            num_layers=cfg["model"].get("num_graph_layers", 3),
            num_heads=cfg["model"].get("num_heads", 4),
            dropout=cfg["model"].get("dropout", 0.1),
            clinical_knowledge_graph=kg,
            backbone=BACKBONE,
            img_size=IMG_SIZE,
        )
        quant_model.load_state_dict(model.state_dict())
        results = qb.compare_formats(quant_model, model_name=BEST_MODEL, n_runs=30)

        print(f"\n  {'Format':<15} {'Latency':>10} {'P95':>10} {'Size':>10}")
        print(f"  {'─'*50}")
        for fmt, data in results.items():
            if "error" not in data:
                print(f"  {fmt:<15} {data['mean_ms']:>9.2f}ms {data['p95_ms']:>9.2f}ms {str(data.get('size_MB','?')):>9}MB")
            else:
                print(f"  {fmt:<15} ERROR: {data['error'][:40]}")
        qb.save_report(results, export_dir)
    except Exception as e:
        log.warning(f"  Quantization benchmark failed: {e}")

    # ━━ SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("\n" + "="*60)
    log.info("  FINE-TUNING + EXPORT COMPLETE")
    log.info("="*60)
    print(f"\n  Model:        {BEST_MODEL}")
    print(f"  Image size:   {IMG_SIZE}")
    print(f"  Best F1:      {best_f1:.4f}")
    print(f"  AUC:          {best_metrics.get('auc_roc', 0):.4f}")
    print(f"  Precision:    {best_metrics.get('precision_macro', 0):.4f}")
    print(f"  Recall:       {best_metrics.get('recall_macro', 0):.4f}")
    print(f"  Accuracy:     {best_metrics.get('accuracy_sample', 0):.4f}")
    print(f"  Thresholds:   [{optimal_thresholds.min():.3f}, {optimal_thresholds.max():.3f}] mean={optimal_thresholds.mean():.3f}")
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  Exports:      {export_dir}")
    log.info("Done.")

if __name__ == "__main__":
    main()
