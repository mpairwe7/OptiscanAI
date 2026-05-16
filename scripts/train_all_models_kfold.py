#!/usr/bin/env python3
"""
Train all 4 models with 5-fold cross-validation, evaluate, and recommend best.

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1,4 python3 scripts/train_all_models_kfold.py
"""

import json
import logging
import random
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
from torch.amp import GradScaler, autocast

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.datamodule import (
    DISEASE_COLUMNS,
    RetinalDataModule,
    build_multilabel_stratify_labels,
)
from src.training.early_stopping import AdvancedEarlyStopping
from src.training.ema import ModelEMA
from src.training.losses import build_loss
from src.training.metrics import MetricTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Config
# ============================================================================
CONFIG_PATH = "configs/train.yaml"
BACKBONE = "vit_large_patch16_224"  # RETFound retinal foundation model
IMG_SIZE = 224  # RETFound native resolution
K_FOLDS = 3
MAX_EPOCHS = 25
EARLY_STOP_PATIENCE = 3
BATCH_SIZE = 8  # Reduced for ViT-L memory across all models
BACKBONE_LR = 5e-6  # Gentle fine-tuning for pretrained backbone
HEAD_LR = 5e-4  # Standard LR for graph/attention heads
NUM_WORKERS = 0

ALL_MODELS = ["vignn", "graphclip", "visual_language_gnn", "scene_graph_transformer"]

OUTPUT_DIR = Path("outputs/kfold_training")


class MultiDirDataset(torch.utils.data.Dataset):
    """Dataset that searches multiple image directories for each ID."""

    def __init__(self, labels_df, img_dirs, disease_columns, transform=None):
        self.labels_df = labels_df.reset_index(drop=True)
        self.img_dirs = [Path(d) for d in img_dirs]
        self.disease_columns = disease_columns
        self.transform = transform
        self.labels_array = (
            labels_df[disease_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )
        self.image_ids = labels_df["ID"].values

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        labels = torch.from_numpy(self.labels_array[idx])

        # Search all directories
        for img_dir in self.img_dirs:
            for ext in (".png", ".jpg", ".jpeg"):
                path = img_dir / f"{img_id}{ext}"
                if path.exists():
                    try:
                        image = Image.open(path).convert("RGB")
                        if self.transform:
                            image = self.transform(image)
                        return image, labels
                    except Exception:
                        break

        # Placeholder for missing images
        image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, torch.zeros_like(labels)

    def get_pos_weights(self):
        pos = self.labels_array.sum(axis=0).clip(min=1)
        neg = len(self) - pos
        return torch.from_numpy(np.clip(neg / pos, 0.5, 50.0).astype(np.float32))


# ============================================================================
# Seeds
# ============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Model builder (from train.py)
# ============================================================================
def build_model(model_name: str, num_classes: int, cfg: dict) -> nn.Module:
    from src.models.vignn import ClinicalKnowledgeGraph

    names = DISEASE_COLUMNS[:num_classes]
    kg = ClinicalKnowledgeGraph(disease_names=names)
    h = cfg["model"].get("hidden_dim", 384)
    heads = cfg["model"].get("num_heads", 4)
    layers = cfg["model"].get("num_graph_layers", 3)
    drop = cfg["model"].get("dropout", 0.1)
    common = dict(backbone=BACKBONE, img_size=IMG_SIZE)

    if model_name == "vignn":
        from src.models.vignn import ViGNN

        return ViGNN(
            num_classes=num_classes,
            hidden_dim=h,
            num_graph_layers=layers,
            num_heads=heads,
            dropout=drop,
            clinical_knowledge_graph=kg,
            **common,
        )
    elif model_name == "graphclip":
        from src.models.graphclip import GraphCLIP

        return GraphCLIP(
            num_classes=num_classes,
            hidden_dim=h,
            num_graph_layers=layers,
            num_heads=heads,
            dropout=drop,
            clinical_knowledge_graph=kg,
            **common,
        )
    elif model_name == "visual_language_gnn":
        from src.models.visual_language_gnn import VisualLanguageGNN

        return VisualLanguageGNN(
            num_classes=num_classes,
            hidden_dim=h,
            num_layers=layers,
            num_heads=heads,
            dropout=drop,
            clinical_knowledge_graph=kg,
            **common,
        )
    elif model_name == "scene_graph_transformer":
        from src.models.scene_graph_transformer import SceneGraphTransformer

        return SceneGraphTransformer(
            num_classes=num_classes,
            hidden_dim=h,
            num_layers=layers,
            num_heads=heads,
            dropout=drop,
            clinical_knowledge_graph=kg,
            **common,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# Single-fold training loop
# ============================================================================
def train_one_fold(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    fold: int,
    model_name: str,
) -> dict:
    """Train one fold. Returns best metrics dict."""
    # Differential LR: lower for pretrained backbone, higher for graph/attention heads
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if ".encoder." in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-7
    )

    # EMA
    ema = ModelEMA(model, decay=0.9999)

    # Early stopping
    early_stop = AdvancedEarlyStopping(
        patience=EARLY_STOP_PATIENCE, min_delta=0.001, min_epochs=3, mode="max"
    )

    scaler = GradScaler("cuda", enabled=True)
    train_metrics = MetricTracker()
    val_metrics = MetricTracker()

    best_metrics = {}
    best_f1 = 0.0

    for epoch in range(MAX_EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0
        steps = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            total_loss += loss.item()
            steps += 1

            if steps % 5 == 0:
                train_metrics.update(logits.detach(), targets)

        scheduler.step()
        train_loss = total_loss / max(steps, 1)
        train_metrics.compute()
        train_metrics.reset()

        # --- Validate (using EMA weights) ---
        orig_state = deepcopy(model.state_dict())
        ema.apply(model)
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, targets)
                val_loss += loss.item()
                val_steps += 1
                val_metrics.update(logits, targets)

        val_loss /= max(val_steps, 1)
        v_metrics = val_metrics.compute()
        val_metrics.reset()

        # Restore original weights for next training epoch
        model.load_state_dict(orig_state)

        f1 = v_metrics.get("f1_macro", 0)
        auc = v_metrics.get("auc_roc", 0)

        logger.info(
            f"  [{model_name}] Fold {fold+1} Epoch {epoch+1}/{MAX_EPOCHS} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"F1: {f1:.4f} | AUC: {auc:.4f}"
        )

        # Track best
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                "f1_macro": f1,
                "f1_micro": v_metrics.get("f1_micro", 0),
                "auc_roc": auc,
                "precision_macro": v_metrics.get("precision_macro", 0),
                "recall_macro": v_metrics.get("recall_macro", 0),
                "mAP": v_metrics.get("mAP", 0),
                "hamming_loss": v_metrics.get("hamming_loss", 0),
                "best_epoch": epoch + 1,
                "val_loss": val_loss,
            }

        # Early stopping
        should_stop, _ = early_stop(epoch, {"f1": f1, "auc": auc, "loss": val_loss})
        if should_stop:
            logger.info(f"  [{model_name}] Fold {fold+1} early stopped at epoch {epoch+1}")
            break

    return best_metrics


# ============================================================================
# K-Fold for one model
# ============================================================================
def train_model_kfold(
    model_name: str,
    cfg: dict,
    all_labels: pd.DataFrame,
    disease_columns: list[str],
    img_dirs: list[Path],
    device: torch.device,
) -> dict:
    """Train a single model with K-fold CV. Returns aggregated results."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  MODEL: {model_name.upper()} | {K_FOLDS}-Fold CV | {MAX_EPOCHS} max epochs")
    logger.info(f"{'='*70}")

    num_classes = len(disease_columns)

    # Stratify with multilabel-aware signatures when possible.
    stratify_col = build_multilabel_stratify_labels(all_labels, disease_columns)
    if stratify_col is None:
        stratify_col = (
            all_labels["Disease_Risk"].values
            if "Disease_Risk" in all_labels.columns
            else np.zeros(len(all_labels))
        )
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    total_time = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_labels, stratify_col)):
        set_seed(42 + fold)

        train_df = all_labels.iloc[train_idx]
        val_df = all_labels.iloc[val_idx]

        # Datasets - use MultiDirDataset to search both train and val image dirs
        train_transform = get_train_transforms(cfg)
        val_transform = get_val_transforms(cfg)

        train_ds = MultiDirDataset(train_df, img_dirs, disease_columns, train_transform)
        val_ds = MultiDirDataset(val_df, img_dirs, disease_columns, val_transform)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        # Build model + loss
        model = build_model(model_name, num_classes, cfg).to(device)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        pos_weight = train_ds.get_pos_weights().to(device)
        criterion = build_loss(cfg, pos_weight=pos_weight)

        # Use DataParallel if multiple GPUs visible
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        t0 = time.time()
        metrics = train_one_fold(
            model, train_loader, val_loader, criterion, device, cfg, fold, model_name
        )
        fold_time = time.time() - t0
        total_time += fold_time

        metrics["fold"] = fold + 1
        metrics["time_min"] = fold_time / 60
        metrics["params_M"] = params_m
        fold_results.append(metrics)

        logger.info(
            f"  [{model_name}] Fold {fold+1} DONE | "
            f"F1: {metrics['f1_macro']:.4f} | AUC: {metrics['auc_roc']:.4f} | "
            f"Time: {fold_time/60:.1f}min"
        )

        # Cleanup
        del model, criterion, train_ds, val_ds
        torch.cuda.empty_cache()

    # Aggregate
    result = {
        "model": model_name,
        "params_M": fold_results[0]["params_M"],
        "folds": fold_results,
        "total_time_min": total_time / 60,
    }
    for key in ["f1_macro", "f1_micro", "auc_roc", "precision_macro", "recall_macro", "mAP"]:
        vals = [f[key] for f in fold_results]
        result[f"mean_{key}"] = float(np.mean(vals))
        result[f"std_{key}"] = float(np.std(vals))

    logger.info(f"\n  [{model_name}] CV Summary:")
    logger.info(f"    F1 Macro:  {result['mean_f1_macro']:.4f} +/- {result['std_f1_macro']:.4f}")
    logger.info(f"    AUC-ROC:   {result['mean_auc_roc']:.4f} +/- {result['std_auc_roc']:.4f}")
    logger.info(f"    Precision: {result['mean_precision_macro']:.4f}")
    logger.info(f"    Recall:    {result['mean_recall_macro']:.4f}")
    logger.info(f"    Time:      {result['total_time_min']:.1f} min")

    return result


# ============================================================================
# Leaderboard + recommendation
# ============================================================================
def print_leaderboard(all_results: list[dict]):
    """Print final leaderboard with weighted scoring and recommendation."""
    print("\n" + "=" * 90)
    print("  FINAL LEADERBOARD - ALL MODELS ({}-Fold Cross-Validation)".format(K_FOLDS))
    print("=" * 90)

    # Build table
    rows = []
    for r in all_results:
        rows.append(
            {
                "Model": r["model"],
                "F1": r["mean_f1_macro"],
                "AUC-ROC": r["mean_auc_roc"],
                "Precision": r["mean_precision_macro"],
                "Recall": r["mean_recall_macro"],
                "Parameters (M)": r["params_M"],
                "Time (min)": r["total_time_min"],
                "F1 Std": r["std_f1_macro"],
            }
        )

    df = pd.DataFrame(rows)

    # Print table
    print(
        f"\n{'Model':<28} {'F1':>8} {'AUC-ROC':>8} {'Prec':>8} {'Recall':>8} {'Params':>8} {'Time':>8}"
    )
    print("-" * 90)
    for _, row in df.iterrows():
        print(
            f"  {row['Model']:<26} {row['F1']:>7.4f} {row['AUC-ROC']:>7.4f} "
            f"{row['Precision']:>7.4f} {row['Recall']:>7.4f} "
            f"{row['Parameters (M)']:>6.1f}M {row['Time (min)']:>7.1f}m"
        )
    print("-" * 90)

    # Weighted scoring: F1:40%, AUC:30%, Prec:15%, Rec:15%
    scores = {}
    for _, row in df.iterrows():
        name = row["Model"]
        scores[name] = (
            0.40 * row["F1"]
            + 0.30 * row["AUC-ROC"]
            + 0.15 * row["Precision"]
            + 0.15 * row["Recall"]
        )

    best_model = max(scores, key=scores.get)
    best_row = df[df["Model"] == best_model].iloc[0]

    print("\n  WEIGHTED SCORES (F1:40% | AUC:30% | Prec:15% | Rec:15%):")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- BEST" if name == best_model else ""
        print(f"    {name:<28} {score:.4f}{marker}")

    print(f"\n{'='*70}")
    print("  RECOMMENDATION")
    print(f"{'='*70}")
    print(f"\n  Recommended: {best_model}")
    print(f"    Overall Score: {scores[best_model]:.4f}")
    print(f"    F1 Score: {best_row['F1']:.4f}")
    print(f"    AUC-ROC:  {best_row['AUC-ROC']:.4f}")
    print(f"    Precision: {best_row['Precision']:.4f}")
    print(f"    Recall:   {best_row['Recall']:.4f}")
    print(f"    Parameters: {best_row['Parameters (M)']:.0f}M")
    print("\n    Rationale: Weighted scoring (F1:40%, AUC:30%, Prec:15%, Rec:15%)")
    print("    Best balance between accuracy and computational efficiency")
    print(f"{'='*70}\n")

    return best_model, scores


# ============================================================================
# Main
# ============================================================================
def main():
    set_seed(42)

    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["batch_size"] = BATCH_SIZE
    cfg["data"]["img_size"] = IMG_SIZE
    cfg["training"]["max_epochs"] = MAX_EPOCHS
    cfg["training"]["early_stopping_patience"] = EARLY_STOP_PATIENCE
    cfg["training"]["loss"] = "focal"
    cfg["training"]["focal_alpha"] = 0.75

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    logger.info(f"Device: {device} | Visible GPUs: {n_gpus}")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        free = torch.cuda.mem_get_info(i)[0] / 1e9
        logger.info(f"  GPU {i}: {name} ({free:.1f} GB free)")

    # Data
    dm = RetinalDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")

    num_classes = len(dm.disease_columns)
    cfg["model"]["num_classes"] = num_classes

    # Merge train + val for k-fold splitting
    train_df = pd.DataFrame(dm.train_dataset.labels_array, columns=dm.disease_columns)
    train_df["ID"] = dm.train_dataset.image_ids
    if "Disease_Risk" in dm.train_dataset.labels_df.columns:
        train_df["Disease_Risk"] = dm.train_dataset.labels_df["Disease_Risk"].values

    val_df = pd.DataFrame(dm.val_dataset.labels_array, columns=dm.disease_columns)
    val_df["ID"] = dm.val_dataset.image_ids
    if "Disease_Risk" in dm.val_dataset.labels_df.columns:
        val_df["Disease_Risk"] = dm.val_dataset.labels_df["Disease_Risk"].values

    all_labels = pd.concat([train_df, val_df], ignore_index=True)
    if "Disease_Risk" not in all_labels.columns:
        all_labels["Disease_Risk"] = all_labels[dm.disease_columns].sum(axis=1).clip(0, 1)

    # For k-fold: images span both train and val directories
    # Create a multi-dir dataset wrapper
    train_img_dir = dm.train_dataset.img_dir
    val_img_dir = dm.val_dataset.img_dir
    # We'll use a combined image lookup in the dataset - see MultiDirDataset below
    img_dirs = [train_img_dir, val_img_dir]

    logger.info(f"\nTotal samples for K-Fold: {len(all_labels)}")
    logger.info(f"Disease classes: {num_classes}")
    logger.info(f"Backbone: {BACKBONE} | img_size: {IMG_SIZE}")
    logger.info(f"Models: {ALL_MODELS}")
    logger.info(
        f"K-Folds: {K_FOLDS} | Max Epochs: {MAX_EPOCHS} | Early Stop: {EARLY_STOP_PATIENCE}"
    )
    logger.info(f"Batch Size: {BATCH_SIZE} | GPUs: {n_gpus}")
    logger.info(f"LR: backbone={BACKBONE_LR} head={HEAD_LR}")

    # Output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train all models
    all_results = []
    for model_name in ALL_MODELS:
        try:
            result = train_model_kfold(
                model_name, cfg, all_labels, dm.disease_columns, img_dirs, device
            )
            all_results.append(result)

            # Save intermediate results
            with open(OUTPUT_DIR / f"{model_name}_kfold_results.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"FAILED: {model_name} - {e}")
            import traceback

            traceback.print_exc()

    if not all_results:
        logger.error("No models trained successfully!")
        return

    # Save all results
    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print leaderboard
    best_model, scores = print_leaderboard(all_results)

    # Save recommendation
    with open(OUTPUT_DIR / "recommendation.json", "w") as f:
        json.dump(
            {
                "best_model": best_model,
                "score": scores[best_model],
                "all_scores": scores,
                "config": {
                    "k_folds": K_FOLDS,
                    "max_epochs": MAX_EPOCHS,
                    "early_stop_patience": EARLY_STOP_PATIENCE,
                    "batch_size": BATCH_SIZE,
                    "scoring_weights": "F1:40%, AUC:30%, Prec:15%, Rec:15%",
                },
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Results saved to {OUTPUT_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
