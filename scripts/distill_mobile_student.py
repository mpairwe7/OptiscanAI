#!/usr/bin/env python3
"""Knowledge distillation: RetinalFoundationHybridV2 -> MobileStudentV1.

Usage:
    PYTHONPATH=. python scripts/distill_mobile_student.py \
        --config configs/distillation_mobile_2026.yaml

Produces:
    outputs/distillation/
        student_best.pth           # Best student checkpoint
        student_last.pth           # Last epoch checkpoint
        thresholds_student.json    # Re-optimized per-class thresholds
        distillation_metrics.json  # Training metrics + parity report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mobile_student import MobileStudentV1
from src.training.distillation_loss import PrecisionAwareDistillationLoss

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def build_teacher(cfg: dict, device: torch.device) -> nn.Module:
    """Load the pre-trained teacher model in eval mode."""
    teacher_cfg = cfg["teacher"]
    model_cfg_path = teacher_cfg.get("config", "configs/hybrid_precision_2026.yaml")

    import yaml

    with open(model_cfg_path) as f:
        teacher_model_cfg = yaml.safe_load(f)

    from src.models.retinal_foundation_hybrid_v2 import RetinalFoundationHybridV2
    from src.models.vignn import ClinicalKnowledgeGraph

    # Get num_classes — prefer from teacher checkpoint, then config
    ckpt_path = teacher_cfg.get("checkpoint")
    num_classes = teacher_model_cfg["model"].get("num_classes", 28)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "num_classes" in ckpt_peek:
            num_classes = ckpt_peek["num_classes"]
            logger.info("Using num_classes=%d from teacher checkpoint", num_classes)
        if "disease_columns" in ckpt_peek:
            disease_names = ckpt_peek["disease_columns"]
        else:
            disease_names = [f"class_{i}" for i in range(num_classes)]
        del ckpt_peek
    else:
        disease_names = [f"class_{i}" for i in range(num_classes)]

    # Override from distillation config if set
    cfg_nc = cfg.get("class_filtering", {}).get("num_classes")
    if cfg_nc:
        num_classes = cfg_nc

    kg = ClinicalKnowledgeGraph(disease_names=disease_names)
    mc = teacher_model_cfg["model"]

    teacher = RetinalFoundationHybridV2(
        num_classes=num_classes,
        hidden_dim=mc.get("hidden_dim", 512),
        head_dropout1=mc.get("head_dropout1", 0.5),
        head_dropout2=mc.get("head_dropout2", 0.3),
        clinical_knowledge_graph=kg,
        backbone=mc.get("backbone", "vit_large_patch16_224"),
        use_lora=mc.get("use_lora", True),
        lora_rank=mc.get("lora_rank", 16),
        lora_alpha=mc.get("lora_alpha", 32.0),
    )

    ckpt_path = teacher_cfg.get("checkpoint")
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        # Strip common prefixes
        cleaned = {}
        for k, v in state.items():
            k = k.replace("model.", "")
            cleaned[k] = v

        teacher.load_state_dict(cleaned, strict=False)
        logger.info("Loaded teacher checkpoint: %s (%d layers)", ckpt_path, len(cleaned))

        # Load thresholds from checkpoint if embedded
        if "thresholds" in ckpt and ckpt["thresholds"]:
            thresh = ckpt["thresholds"]
            if isinstance(thresh, list):
                thresh = torch.tensor(thresh, dtype=torch.float32)
            if len(thresh) == num_classes:
                teacher.thresholds.copy_(thresh)
                logger.info("Loaded thresholds from checkpoint (n=%d)", len(thresh))

        # Override num_classes from checkpoint if available
        if "num_classes" in ckpt and ckpt["num_classes"] != num_classes:
            logger.warning(
                "Checkpoint num_classes=%d differs from config=%d",
                ckpt["num_classes"], num_classes,
            )

    # Load thresholds from separate file if specified
    thresh_path = teacher_cfg.get("thresholds")
    if thresh_path and Path(thresh_path).exists():
        teacher.load_thresholds(thresh_path)
        logger.info("Loaded teacher thresholds from file: %s", thresh_path)

    # Merge LoRA for inference
    teacher.prepare_for_export()
    teacher = teacher.to(device).eval()

    for p in teacher.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in teacher.parameters())
    logger.info("Teacher loaded: %.1fM params", total_params / 1e6)
    return teacher


def build_student(cfg: dict, device: torch.device) -> MobileStudentV1:
    """Create the student model."""
    sc = cfg["student"]
    num_classes = cfg.get("class_filtering", {}).get("num_classes")
    if num_classes is None:
        teacher_cfg_path = cfg["teacher"].get("config", "configs/hybrid_precision_2026.yaml")
        import yaml

        with open(teacher_cfg_path) as f:
            tc = yaml.safe_load(f)
        num_classes = tc["model"].get("num_classes", 28)

    student = MobileStudentV1(
        num_classes=num_classes,
        hidden_dim=sc.get("hidden_dim", 512),
        dropout1=sc.get("dropout1", 0.4),
        dropout2=sc.get("dropout2", 0.25),
        pretrained=sc.get("pretrained", True),
    )
    return student.to(device)


def build_dataloader(cfg: dict, split: str = "train") -> DataLoader:
    """Build DataLoader using the existing RetinalDataModule."""
    from src.data.datamodule import RetinalDataModule

    dm = RetinalDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    if split == "train":
        return dm.train_dataloader()
    elif split == "val":
        return dm.val_dataloader()
    else:
        return dm.test_dataloader()


def extract_teacher_features(
    teacher: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """Extract teacher's global pool features (512-dim) via forward hook."""
    features = {}

    def hook_fn(module, input, output):
        features["global_pool"] = output

    handle = teacher.global_pool.register_forward_hook(hook_fn)
    with torch.no_grad():
        teacher(x)
    handle.remove()
    return features["global_pool"]


def validate_epoch(
    student: MobileStudentV1,
    val_loader: DataLoader,
    device: torch.device,
    thresholds: torch.Tensor,
    class_indices: list[int] | None = None,
) -> dict:
    """Compute validation metrics."""
    student.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            # Dataset returns [images, labels] list or {"image":..., "labels":...} dict
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            else:
                images, targets = batch["image"], batch["labels"]
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if class_indices is not None and targets.shape[1] > len(class_indices):
                targets = targets[:, class_indices]
            logits = student(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    preds = (all_probs >= thresholds.cpu().unsqueeze(0)).float()

    # Per-class metrics
    eps = 1e-8
    tp = (preds * all_targets).sum(dim=0)
    fp = (preds * (1 - all_targets)).sum(dim=0)
    fn = ((1 - preds) * all_targets).sum(dim=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    metrics = {
        "precision_macro": precision.mean().item(),
        "recall_macro": recall.mean().item(),
        "f1_macro": f1.mean().item(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "min_precision": precision.min().item(),
        "classes_below_floor": int((precision < 0.10).sum().item()),
    }

    student.train()
    return metrics


def train(cfg: dict) -> dict:
    """Run the full distillation training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Build components ----
    teacher = build_teacher(cfg, device)
    student = build_student(cfg, device)
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    # ---- Resolve class index mapping ----
    # The dataset has 45 columns but the teacher uses 24 classes.
    # We need to select the teacher's columns from the full label tensor.
    teacher_ckpt_path = cfg["teacher"].get("checkpoint")
    class_indices = None
    if teacher_ckpt_path and Path(teacher_ckpt_path).exists():
        ckpt_peek = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
        if "disease_columns" in ckpt_peek:
            teacher_cols = ckpt_peek["disease_columns"]
            # Get all 45 disease column names from the dataset
            all_cols = list(train_loader.dataset.disease_columns) if hasattr(train_loader.dataset, "disease_columns") else None
            if all_cols:
                class_indices = [all_cols.index(c) for c in teacher_cols if c in all_cols]
                logger.info(
                    "Class mapping: %d teacher classes from %d dataset columns",
                    len(class_indices), len(all_cols),
                )
        del ckpt_peek

    dist_cfg = cfg["distillation"]
    train_cfg = cfg["training"]
    cfg.get("precision_floor", {})

    criterion = PrecisionAwareDistillationLoss(
        alpha=dist_cfg.get("alpha", 0.6),
        beta=dist_cfg.get("beta", 0.15),
        gamma=dist_cfg.get("gamma", 0.05),
        initial_temperature=dist_cfg.get("initial_temperature", 6.0),
        final_temperature=dist_cfg.get("final_temperature", 2.0),
        total_epochs=train_cfg.get("max_epochs", 40),
        asl_gamma_neg=train_cfg.get("gamma_neg", 4.0),
        asl_gamma_pos=train_cfg.get("gamma_pos", 0.0),
        asl_clip=train_cfg.get("asl_clip", 0.05),
        label_smoothing=train_cfg.get("label_smoothing", 0.05),
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=train_cfg.get("lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    max_epochs = train_cfg.get("max_epochs", 40)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    total_steps = max_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )

    use_amp = "cuda" in str(device) and train_cfg.get("precision") == "bf16-mixed"
    scaler = GradScaler(enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # ---- Output dirs ----
    ckpt_dir = Path(cfg["checkpointing"]["dirpath"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    best_metric = 0.0
    patience_counter = 0
    patience = train_cfg.get("patience", 10)
    history = []

    teacher_thresholds = teacher.thresholds.clone().to(device)
    student.thresholds.copy_(teacher_thresholds)

    for epoch in range(max_epochs):
        criterion.set_epoch(epoch)
        student.train()

        epoch_losses = {"total": 0, "kd": 0, "task": 0, "feature": 0, "threshold": 0}
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{max_epochs}",
            disable=False,
        )

        for batch in pbar:
            # Dataset returns [images, labels] list or {"image":..., "labels":...} dict
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            else:
                images, targets = batch["image"], batch["labels"]
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Slice targets to teacher's class subset if dataset has more columns
            if class_indices is not None and targets.shape[1] > len(class_indices):
                targets = targets[:, class_indices]

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(images)
                teacher_features = extract_teacher_features(teacher, images)

            # Student forward (autocast for model, loss in fp32 for BCE safety)
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                student_logits = student(images)
                student_features = student.get_features(images)

            # Loss computation in fp32 (BCE is unsafe under autocast)
            losses = criterion(
                student_logits=student_logits.float(),
                teacher_logits=teacher_logits.float(),
                targets=targets.float(),
                student_features=student_features.float(),
                teacher_features=teacher_features.float(),
                thresholds=teacher_thresholds,
            )
            loss = losses["total"]

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if train_cfg.get("gradient_clip_val"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    student.parameters(), train_cfg["gradient_clip_val"]
                )
            scaler.step(optimizer)
            scaler.update()

            # Warmup + cosine schedule
            step = epoch * len(train_loader) + n_batches
            if step < warmup_steps:
                lr_scale = (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = train_cfg.get("lr", 3e-4) * lr_scale
            else:
                scheduler.step()

            # Accumulate losses
            epoch_losses["total"] += losses["total"].item()
            epoch_losses["kd"] += losses["kd_loss"].item()
            epoch_losses["task"] += losses["task_loss"].item()
            epoch_losses["feature"] += losses["feature_loss"].item()
            epoch_losses["threshold"] += losses["threshold_loss"].item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{losses['total'].item():.4f}",
                T=f"{losses['temperature'].item():.1f}",
            )

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Validation
        val_metrics = validate_epoch(student, val_loader, device, teacher_thresholds, class_indices)

        epoch_info = {
            "epoch": epoch + 1,
            "losses": epoch_losses,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "temperature": criterion.temperature,
        }
        history.append(epoch_info)

        logger.info(
            "Epoch %d/%d — loss=%.4f (kd=%.4f task=%.4f feat=%.4f thresh=%.4f) "
            "val_P=%.3f val_R=%.3f val_F1=%.3f T=%.1f min_P=%.3f below_floor=%d",
            epoch + 1, max_epochs,
            epoch_losses["total"], epoch_losses["kd"], epoch_losses["task"],
            epoch_losses["feature"], epoch_losses["threshold"],
            val_metrics["precision_macro"], val_metrics["recall_macro"],
            val_metrics["f1_macro"], criterion.temperature,
            val_metrics["min_precision"], val_metrics["classes_below_floor"],
        )

        # Checkpointing
        monitor = val_metrics.get(
            cfg["checkpointing"].get("monitor", "precision_macro"), 0
        )
        if monitor > best_metric:
            best_metric = monitor
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": student.state_dict(),
                    "thresholds": student.thresholds.cpu(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                ckpt_dir / "student_best.pth",
            )
            logger.info("Saved best checkpoint (metric=%.4f)", best_metric)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "thresholds": student.thresholds.cpu(),
                "val_metrics": val_metrics,
            },
            ckpt_dir / "student_last.pth",
        )

    # ---- Save final artifacts ----
    # Re-optimize thresholds for the student
    student.save_thresholds(ckpt_dir / "thresholds_student.json")

    # Save training history
    metrics_path = ckpt_dir / "distillation_metrics.json"
    summary = {
        "best_metric": best_metric,
        "total_epochs": len(history),
        "student_params": student.get_param_summary(),
        "final_val_metrics": history[-1]["val_metrics"] if history else {},
        "history": history,
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved metrics to %s", metrics_path)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Distill MobileStudentV1")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/distillation_mobile_2026.yaml",
        help="Path to distillation config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config)
    summary = train(cfg)

    print("\nDistillation complete!")
    print(f"  Best metric: {summary['best_metric']:.4f}")
    print(f"  Total epochs: {summary['total_epochs']}")
    if summary.get("student_params"):
        sp = summary["student_params"]
        print(f"  Student params: {sp['total_params'] / 1e6:.2f}M")
        print(f"  Est. INT8 ONNX: {sp['estimated_onnx_int8_mb']:.1f} MB")


if __name__ == "__main__":
    main()
