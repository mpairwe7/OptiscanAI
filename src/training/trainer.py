"""
Multi-GPU DDP training engine for retinal disease classification.
Uses PyTorch native DDP with torchrun, mixed precision, and W&B logging.
"""

import json
import logging
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    StepLR,
)
from tqdm import tqdm

from src.data.mixup import build_mixup
from src.training.ema import ModelEMA
from src.training.lr_finder import LRFinder
from src.training.metrics import MetricTracker

logger = logging.getLogger(__name__)


class DDPTrainer:
    """
    Distributed Data Parallel trainer with mixed precision, gradient
    accumulation, early stopping, and experiment tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        cfg: dict,
        datamodule,
    ):
        self.cfg = cfg
        self.train_cfg = cfg["training"]
        self.dist_cfg = cfg["distributed"]
        self.ckpt_cfg = cfg["checkpointing"]
        self.eval_cfg = cfg.get("evaluation", {})

        # DDP setup
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.global_rank == 0
        self.distributed = self.world_size > 1

        # Device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        # Move model to device
        model = model.to(self.device)

        # Sync batchnorm
        if self.distributed and self.dist_cfg.get("sync_batchnorm", True):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Wrap in DDP
        if self.distributed:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.dist_cfg.get(
                    "find_unused_parameters", False
                ),
            )
        else:
            self.model = model

        self.criterion = criterion.to(self.device)
        self.datamodule = datamodule

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        # Scheduler
        self.scheduler = None  # Built after dataloader is ready

        # Mixed precision
        precision = self.train_cfg.get("precision", "16-mixed")
        self.use_amp = self.device.type == "cuda" and (
            "16" in str(precision) or "bf16" in str(precision)
        )
        self.autocast_device = self.device.type
        self.amp_dtype = torch.bfloat16 if "bf16" in str(precision) else torch.float16
        self.scaler = GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        # Gradient accumulation
        self.grad_accum_steps = self.train_cfg.get("gradient_accumulation_steps", 1)
        self.grad_clip_val = self.train_cfg.get("gradient_clip_val", 1.0)

        # Metrics
        self.default_threshold = self.eval_cfg.get("threshold", 0.5)
        self.optimal_threshold_search = self.eval_cfg.get(
            "optimal_threshold_search", False
        )
        self.threshold_search_space = self.eval_cfg.get("threshold_search_space")
        self.monitor_metric_name = (
            "f1_macro_opt" if self.optimal_threshold_search else "f1_macro"
        )
        self.train_metrics = MetricTracker(threshold=self.default_threshold)
        self.val_metrics = MetricTracker(threshold=self.default_threshold)

        # Augmentation and evaluation helpers
        self.mixup_fn = build_mixup(cfg)
        ema_cfg = self.train_cfg.get("ema", {})
        self.use_ema = ema_cfg.get("enabled", False)
        self.ema = (
            ModelEMA(self._raw_model(), decay=ema_cfg.get("decay", 0.9999))
            if self.use_ema
            else None
        )

        # Early stopping
        self.patience = self.train_cfg.get("early_stopping_patience", 5)
        self.best_metric = 0.0
        self.epochs_no_improve = 0
        self.best_model_state = None
        self.best_thresholds = None

        # Checkpointing
        self.ckpt_dir = Path(self.ckpt_cfg.get("dirpath", "outputs/checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = self.ckpt_cfg.get("save_top_k", 3)

        # Output dir
        self.output_dir = Path(cfg.get("logging", {}).get("save_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        self.wandb_run = None
        if self.is_main:
            self._init_logger()

    def _init_logger(self):
        """Initialize experiment tracking (W&B or fallback to CSV)."""
        log_cfg = self.cfg.get("logging", {})
        logger_type = log_cfg.get("logger", "wandb")

        if logger_type == "wandb":
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=log_cfg.get("project_name", "retinal-disease-mlops"),
                    name=log_cfg.get("experiment_name", "run"),
                    config=self.cfg,
                    dir=str(self.output_dir),
                    resume="allow",
                )
                logger.info("W&B logging initialized")
            except Exception as e:
                logger.warning(f"W&B init failed ({e}), falling back to CSV")
                self.wandb_run = None

    def _raw_model(self) -> nn.Module:
        """Return the underlying model without the DDP wrapper."""
        return self.model.module if self.distributed else self.model

    def _eval_model(self) -> nn.Module:
        """Return the model used for validation and export."""
        if self.use_ema and self.ema is not None:
            return self.ema.module()
        return self._raw_model()

    def _run_lr_finder(self, train_loader):
        """Run an optional learning-rate sweep before scheduler creation."""
        lr_cfg = self.train_cfg.get("lr_finder", {})
        if not lr_cfg.get("enabled", False):
            return

        suggested_lr = self.train_cfg["learning_rate"]

        if self.is_main:
            finder = LRFinder()
            suggested_lr = finder.find(
                model=self._raw_model(),
                criterion=self.criterion,
                train_loader=train_loader,
                device=self.device,
                min_lr=lr_cfg.get("min_lr", 1e-7),
                max_lr=lr_cfg.get("max_lr", 10.0),
                num_steps=lr_cfg.get("num_steps", 100),
            )
            try:
                finder.plot(self.output_dir / "lr_finder")
            except Exception as e:
                logger.warning(f"Could not save LR finder plot: {e}")

        if self.distributed:
            lr_tensor = torch.tensor([suggested_lr], device=self.device, dtype=torch.float32)
            dist.broadcast(lr_tensor, src=0)
            suggested_lr = float(lr_tensor.item())

        self.train_cfg["learning_rate"] = suggested_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = suggested_lr

        if self.is_main:
            logger.info(f"LR finder applied suggested LR: {suggested_lr:.2e}")

    def _build_scheduler(self, train_loader):
        """Build LR scheduler after dataloader is available."""
        sched_name = self.train_cfg.get("scheduler", "cosine")
        max_epochs = self.train_cfg["max_epochs"]
        steps_per_epoch = max(1, math.ceil(len(train_loader) / self.grad_accum_steps))

        if sched_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs * steps_per_epoch,
                eta_min=1e-7,
            )
        elif sched_name == "step":
            self.scheduler = StepLR(
                self.optimizer, step_size=max_epochs // 3, gamma=0.1
            )
        elif sched_name == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.train_cfg["learning_rate"],
                total_steps=max_epochs * steps_per_epoch,
                pct_start=0.1,
            )

    def train(self):
        """Main training loop."""
        max_epochs = self.train_cfg["max_epochs"]
        warmup_epochs = self.train_cfg.get("warmup_epochs", 2)

        # Create dataloaders
        train_loader = self.datamodule.train_dataloader(distributed=self.distributed)
        val_loader = (
            self.datamodule.val_dataloader(distributed=False)
            if (not self.distributed or self.is_main)
            else None
        )
        self._run_lr_finder(train_loader)
        self._build_scheduler(train_loader)

        if self.is_main:
            eff_batch = self.datamodule.batch_size * self.world_size * self.grad_accum_steps
            logger.info(f"{'='*70}")
            logger.info("  TRAINING START")
            logger.info(f"  GPUs: {self.world_size} | Batch/GPU: {self.datamodule.batch_size}")
            logger.info(f"  Effective batch size: {eff_batch}")
            logger.info(f"  Epochs: {max_epochs} | LR: {self.train_cfg['learning_rate']}")
            logger.info(f"  Mixed precision: {self.use_amp} ({self.amp_dtype})")
            logger.info(f"  Gradient accumulation steps: {self.grad_accum_steps}")
            logger.info(f"  EMA enabled: {self.use_ema}")
            logger.info(f"  MixUp/CutMix enabled: {self.mixup_fn is not None}")
            logger.info(f"{'='*70}")

        history = []
        start_time = time.time()

        for epoch in range(max_epochs):
            # Set epoch for distributed sampler
            if self.distributed:
                train_loader.sampler.set_epoch(epoch)

            # Warmup LR
            if epoch < warmup_epochs:
                warmup_lr = self.train_cfg["learning_rate"] * (epoch + 1) / warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            # Train epoch
            train_loss = self._train_epoch(train_loader, epoch, max_epochs)
            train_metrics = self.train_metrics.compute()
            self.train_metrics.reset()

            # Validate
            val_loss = 0.0
            val_metrics = {}
            epoch_thresholds = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, epoch, max_epochs)
                val_metrics = self.val_metrics.compute()
                if self.optimal_threshold_search:
                    epoch_thresholds = self.val_metrics.optimize_thresholds(
                        self.threshold_search_space
                    )
                    opt_metrics = self.val_metrics.compute(epoch_thresholds)
                    val_metrics.update(
                        {f"{k}_opt": v for k, v in opt_metrics.items()}
                    )
                self.val_metrics.reset()

            # Current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            stop_requested = False
            if self.is_main:
                epoch_data = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": current_lr,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }
                history.append(epoch_data)

                f1 = val_metrics.get(self.monitor_metric_name, val_metrics.get("f1_macro", 0))
                auc = val_metrics.get("auc_roc", 0)
                if self.optimal_threshold_search:
                    logger.info(
                        f"Epoch {epoch+1}/{max_epochs} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Val F1(opt): {f1:.4f} | Val F1@0.5: {val_metrics.get('f1_macro', 0):.4f} | "
                        f"Val AUC: {auc:.4f} | LR: {current_lr:.2e}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch+1}/{max_epochs} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Val F1: {f1:.4f} | Val AUC: {auc:.4f} | LR: {current_lr:.2e}"
                    )

                if self.wandb_run:
                    self.wandb_run.log(epoch_data)

                monitor_metric = val_metrics.get(
                    self.monitor_metric_name, val_metrics.get("f1_macro", 0)
                )
                is_best = monitor_metric > self.best_metric
                if monitor_metric > self.best_metric:
                    self.best_metric = monitor_metric
                    self.epochs_no_improve = 0
                    self.best_model_state = deepcopy(self._eval_model().state_dict())
                    self.best_thresholds = (
                        epoch_thresholds.tolist()
                        if epoch_thresholds is not None
                        else None
                    )
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        logger.info(
                            f"Early stopping triggered after {self.patience} epochs "
                            f"without improvement. Best {self.monitor_metric_name}: "
                            f"{self.best_metric:.4f}"
                        )
                        stop_requested = True

                self._checkpoint(
                    epoch=epoch,
                    metric=monitor_metric,
                    all_metrics=val_metrics,
                    is_best=is_best,
                    decision_thresholds=epoch_thresholds,
                )

            # Sync early stopping decision across ranks
            if self.distributed:
                stop_tensor = torch.tensor(
                    [1 if (self.is_main and stop_requested) else 0],
                    device=self.device,
                )
                dist.broadcast(stop_tensor, src=0)
                stop_requested = bool(stop_tensor.item())

            if stop_requested:
                break

        elapsed = time.time() - start_time
        if self.is_main:
            logger.info(f"Training complete in {elapsed/60:.1f} minutes")
            logger.info(
                f"Best validation {self.monitor_metric_name}: {self.best_metric:.4f}"
            )

            # Save final artifacts
            self._save_final(history)

        return history

    def _train_epoch(self, loader, epoch, max_epochs) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        step_count = 0

        pbar = (
            tqdm(loader, desc=f"Train {epoch+1}/{max_epochs}", leave=False)
            if self.is_main
            else loader
        )

        self.optimizer.zero_grad()
        num_batches = len(loader)

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.mixup_fn is not None:
                images, targets = self.mixup_fn(images, targets)

            accum_divisor = self.grad_accum_steps
            is_last_batch = batch_idx == num_batches - 1
            if is_last_batch and (batch_idx + 1) % self.grad_accum_steps != 0:
                accum_divisor = (batch_idx % self.grad_accum_steps) + 1

            with autocast(
                device_type=self.autocast_device,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, targets) / accum_divisor

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_val
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.use_ema and self.ema is not None:
                    self.ema.update(self._raw_model())

                if self.scheduler is not None and epoch >= self.cfg["training"].get("warmup_epochs", 2):
                    self.scheduler.step()

            total_loss += loss.item() * accum_divisor
            step_count += 1

            if bool(torch.all((targets == 0) | (targets == 1)).item()):
                self.train_metrics.update(logits.detach(), targets)

            if self.is_main and isinstance(pbar, tqdm):
                pbar.set_postfix(loss=f"{loss.item() * accum_divisor:.4f}")

        if step_count % self.grad_accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_val
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.use_ema and self.ema is not None:
                self.ema.update(self._raw_model())
            if self.scheduler is not None and epoch >= self.cfg["training"].get("warmup_epochs", 2):
                self.scheduler.step()

        return total_loss / max(step_count, 1)

    @torch.no_grad()
    def _validate_epoch(self, loader, epoch, max_epochs) -> float:
        """Validate one epoch."""
        model = self._eval_model()
        model.eval()
        total_loss = 0.0
        step_count = 0

        pbar = (
            tqdm(loader, desc=f"Val {epoch+1}/{max_epochs}", leave=False)
            if self.is_main
            else loader
        )

        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(
                device_type=self.autocast_device,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = model(images)
                loss = self.criterion(logits, targets)

            total_loss += loss.item()
            step_count += 1
            self.val_metrics.update(logits, targets)

        return total_loss / max(step_count, 1)

    def _checkpoint(
        self,
        epoch: int,
        metric: float,
        all_metrics: dict,
        is_best: bool,
        decision_thresholds=None,
    ):
        """Save model checkpoint if metric improved."""
        raw_model = self._eval_model()

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_f1": self.best_metric,
            "monitor_metric_name": self.monitor_metric_name,
            "metrics": all_metrics,
            "config": self.cfg,
            "decision_thresholds": (
                decision_thresholds.tolist()
                if decision_thresholds is not None
                else self.best_thresholds
            ),
        }

        # Always save last
        if self.ckpt_cfg.get("save_last", True):
            path = self.ckpt_dir / "last.pt"
            torch.save(ckpt, path)

        # Save best
        if is_best:
            path = self.ckpt_dir / "best.pt"
            torch.save(ckpt, path)
            logger.info(f"  Saved best checkpoint: F1={metric:.4f}")

        # Save top-k
        path = self.ckpt_dir / f"epoch_{epoch+1:02d}_f1_{metric:.4f}.pt"
        torch.save(ckpt, path)

        # Cleanup old checkpoints
        ckpts = sorted(
            self.ckpt_dir.glob("epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in ckpts[self.save_top_k :]:
            old.unlink()

    def _save_final(self, history: list):
        """Save training history and final model artifacts."""
        # Save history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save best model in deployment format
        if self.best_model_state is not None:
            deploy_path = self.output_dir / "best_model.pth"
            torch.save(
                {
                    "model_state_dict": self.best_model_state,
                    "num_classes": self.cfg["model"]["num_classes"],
                    "best_f1": self.best_metric,
                    "config": self.cfg["model"],
                    "decision_thresholds": self.best_thresholds,
                },
                deploy_path,
            )
            logger.info(f"Saved deployment model: {deploy_path}")

        # Save metadata
        meta = {
            "best_f1": self.best_metric,
            "num_classes": self.cfg["model"]["num_classes"],
            "model_name": self.cfg["model"]["name"],
            "total_epochs": len(history),
            "gpus_used": self.world_size,
            "precision": self.train_cfg.get("precision", "32"),
            "monitor_metric_name": self.monitor_metric_name,
            "decision_thresholds": self.best_thresholds,
        }
        meta_path = self.output_dir / "training_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if self.wandb_run:
            self.wandb_run.finish()
