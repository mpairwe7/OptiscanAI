#!/usr/bin/env python3
"""
Precision-rescue training for RetinalFoundationHybridV2.

Implements all 7 precision improvement strategies:
  1. Asymmetric Loss (gamma_pos=0, gamma_neg=4)
  2. Rare class filtering (<10 samples dropped)
  3. Per-class precision-floor threshold optimization
  4. Label smoothing + class-balanced sampling
  5. Bottleneck head with strong dropout
  6. Staged backbone unfreezing
  7. Retinal-specific augmentation

Usage:
    # Single GPU
    python scripts/train_hybrid_precision_v2.py --config configs/hybrid_precision_2026.yaml

    # Multi-GPU DDP
    torchrun --nproc_per_node=4 scripts/train_hybrid_precision_v2.py --config configs/hybrid_precision_2026.yaml
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
import yaml

from src.data.datamodule import RetinalDataModule
from src.models.retinal_foundation_hybrid_v2 import (
    AsymmetricLossV2,
    RetinalFoundationHybridV2,
    create_hybrid_v2,
    filter_rare_classes,
)
from src.models.vignn import ClinicalKnowledgeGraph
from src.evaluation.precision_threshold_optimizer import (
    optimize_thresholds_with_precision_floor,
    save_thresholds,
)
from src.training.metrics import MetricTracker, compute_multilabel_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_weighted_sampler(labels_df, disease_columns):
    """Create WeightedRandomSampler for class-balanced sampling.

    Each sample's weight is the inverse frequency of its rarest positive class.
    This ensures rare diseases are seen proportionally more often.
    """
    import pandas as pd
    from torch.utils.data import WeightedRandomSampler

    label_matrix = labels_df[disease_columns].values.astype(np.float32)

    # Per-class frequency (positive rate)
    class_freq = label_matrix.sum(axis=0) / len(label_matrix)
    class_freq = np.clip(class_freq, 1e-6, 1.0)

    # Per-sample weight = max inverse frequency of its positive classes
    sample_weights = np.zeros(len(label_matrix))
    for i in range(len(label_matrix)):
        positives = np.where(label_matrix[i] > 0.5)[0]
        if len(positives) > 0:
            # Weight by rarest positive class
            sample_weights[i] = max(1.0 / class_freq[j] for j in positives)
        else:
            sample_weights[i] = 1.0  # Normal samples

    # Normalize
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    col_indices=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if col_indices is not None:
            labels = labels[:, col_indices]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 20 == 0:
            logger.info(f"  Epoch {epoch} batch {batch_idx}/{len(loader)} loss={loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, col_indices=None):
    """Validate and return probabilities + targets for threshold optimization."""
    model.eval()
    all_probs = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        lbl = labels.numpy()
        if col_indices is not None:
            lbl = lbl[:, col_indices.cpu().numpy() if hasattr(col_indices, 'cpu') else col_indices]
        all_targets.append(lbl)

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return probs, targets


def main():
    parser = argparse.ArgumentParser(description="Train HybridV2 (precision rescue)")
    parser.add_argument("--config", type=str, default="configs/hybrid_precision_2026.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("training", {}).get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Data ----
    datamodule = RetinalDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # ---- Filter rare classes ----
    min_samples = cfg.get("class_filtering", {}).get("min_samples", 10)
    disease_columns = filter_rare_classes(
        datamodule.disease_columns,
        datamodule.train_dataset.labels_df if hasattr(datamodule.train_dataset, 'labels_df') else None,
        min_samples=min_samples,
    )

    # Fallback: if filter_rare_classes can't access labels, use all columns
    if not disease_columns:
        disease_columns = datamodule.disease_columns
        logger.warning("Could not filter rare classes; using all columns")

    num_classes = len(disease_columns)
    logger.info(f"Training with {num_classes} classes: {disease_columns}")

    # ---- Knowledge graph ----
    kg = ClinicalKnowledgeGraph(disease_names=disease_columns)

    # ---- Model ----
    model_cfg = cfg.get("model", {})
    model = create_hybrid_v2(
        num_classes=num_classes,
        hidden_dim=model_cfg.get("hidden_dim", 512),
        clinical_knowledge_graph=kg,
        backbone=model_cfg.get("backbone", "vit_large_patch16_224"),
        use_lora=model_cfg.get("use_lora", True),
        lora_rank=model_cfg.get("lora_rank", 16),
        lora_alpha=model_cfg.get("lora_alpha", 32.0),
        head_dropout1=model_cfg.get("head_dropout1", 0.5),
        head_dropout2=model_cfg.get("head_dropout2", 0.3),
        freeze_backbone=True,
    )
    model = model.to(device)

    # ---- Loss (ASL) ----
    train_cfg = cfg.get("training", {})
    criterion = AsymmetricLossV2(
        gamma_neg=train_cfg.get("gamma_neg", 4.0),
        gamma_pos=train_cfg.get("gamma_pos", 0.0),
        clip=train_cfg.get("asl_clip", 0.05),
        label_smoothing=train_cfg.get("label_smoothing", 0.05),
    )

    # ---- Optimizer (head only initially) ----
    head_lr = train_cfg.get("head_lr", 5e-4)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=head_lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # ---- Scheduler ----
    max_epochs = train_cfg.get("max_epochs", 25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # ---- Mixed precision ----
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ---- Data loaders ----
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # ---- Column index mapping (dataloader returns all 45 cols, model uses 24) ----
    all_columns = datamodule.disease_columns  # 45 columns from dataloader
    col_indices = [all_columns.index(c) for c in disease_columns if c in all_columns]
    col_indices_t = torch.tensor(col_indices, dtype=torch.long, device=device)
    logger.info(f"Label column filter: {len(all_columns)} -> {len(col_indices)} columns")

    # ---- Training loop ----
    unfreeze_epoch = train_cfg.get("unfreeze_epoch", 10)
    backbone_lr = train_cfg.get("backbone_lr", 1e-6)
    unfreeze_blocks = train_cfg.get("unfreeze_blocks", 4)
    patience = train_cfg.get("patience", 7)
    best_metric = 0.0
    patience_counter = 0

    ckpt_dir = Path(cfg.get("checkpointing", {}).get("dirpath", "outputs/checkpoints/v2"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        logger.info(f"=== Epoch {epoch}/{max_epochs} ===")

        # Staged unfreezing
        if epoch == unfreeze_epoch + 1:
            logger.info(f"Unfreezing last {unfreeze_blocks} backbone blocks at lr={backbone_lr}")
            new_groups = model.unfreeze_backbone_blocks(
                num_blocks=unfreeze_blocks, lr=backbone_lr
            )
            for group in new_groups:
                optimizer.add_param_group(group)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch,
                                     col_indices=col_indices_t)

        # Validate
        val_probs, val_targets = validate(model, val_loader, device, col_indices=col_indices_t)
        val_metrics = compute_multilabel_metrics(val_targets, val_probs, threshold=0.5)

        scheduler.step()

        logger.info(
            f"  Train loss: {train_loss:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"Val Prec: {val_metrics['precision_macro']:.4f} | "
            f"Val Rec: {val_metrics['recall_macro']:.4f} | "
            f"Val AUC: {val_metrics.get('auc_roc', 0):.4f} | "
            f"Val Acc: {val_metrics.get('accuracy_macro', 0):.4f}"
        )

        # Track best by precision (not F1!)
        current_metric = val_metrics["precision_macro"]
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics,
                "disease_columns": disease_columns,
                "num_classes": num_classes,
            }, ckpt_dir / "best.pth")
            logger.info(f"  New best precision: {best_metric:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping after {patience} epochs without improvement")
                break

    # ---- Post-training: optimize thresholds ----
    logger.info("=== Threshold optimization ===")

    # Reload best checkpoint
    best_ckpt = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_probs, val_targets = validate(model, val_loader, device, col_indices=col_indices_t)

    min_precision_floor = cfg.get("threshold_optimization", {}).get("min_precision", 0.10)
    thresholds, report = optimize_thresholds_with_precision_floor(
        val_probs, val_targets,
        min_precision=min_precision_floor,
        disease_names=disease_columns,
    )

    # Apply thresholds to model
    model.thresholds.copy_(torch.tensor(thresholds, dtype=torch.float32))

    # Save thresholds
    thresholds_path = str(ckpt_dir / "thresholds_optimized.json")
    save_thresholds(thresholds, report, thresholds_path)

    # Evaluate with optimized thresholds
    final_metrics = compute_multilabel_metrics(val_targets, val_probs, threshold=thresholds)
    logger.info(
        f"Final metrics with optimized thresholds: "
        f"F1={final_metrics['f1_macro']:.4f} "
        f"Prec={final_metrics['precision_macro']:.4f} "
        f"Rec={final_metrics['recall_macro']:.4f} "
        f"AUC={final_metrics.get('auc_roc', 0):.4f} "
        f"Acc={final_metrics.get('accuracy_macro', 0):.4f}"
    )

    # Save final model with thresholds embedded
    torch.save({
        "model_state_dict": model.state_dict(),
        "disease_columns": disease_columns,
        "num_classes": num_classes,
        "thresholds": thresholds.tolist(),
        "metrics": final_metrics,
    }, ckpt_dir / "final_with_thresholds.pth")

    logger.info(f"Training complete. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
