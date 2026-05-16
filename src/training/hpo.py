"""Hyperparameter optimization with Optuna for retinal disease models."""

import logging
from copy import deepcopy
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def run_hpo(
    base_cfg: dict,
    n_trials: int = 20,
    metric: str = "val/f1_macro",
    direction: str = "maximize",
    timeout_seconds: Optional[int] = None,
) -> dict:
    """Run Optuna HPO study, return best config.

    Args:
        base_cfg: Base training config dict
        n_trials: Number of optimization trials
        metric: Metric to optimize
        direction: 'maximize' or 'minimize'
        timeout_seconds: Optional timeout

    Returns:
        Best config dict with optimized hyperparameters
    """
    try:
        import optuna
        from optuna.trial import Trial
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        return base_cfg

    def objective(trial: Trial) -> float:
        cfg = deepcopy(base_cfg)

        # Hyperparameter search space
        cfg["training"]["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        cfg["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        cfg["training"]["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.15)
        cfg["training"]["gradient_clip_val"] = trial.suggest_float("grad_clip", 0.5, 5.0)

        # Loss function
        loss_type = trial.suggest_categorical("loss", ["focal", "asymmetric"])
        cfg["training"]["loss"] = loss_type
        if loss_type == "focal":
            cfg["training"]["focal_alpha"] = trial.suggest_float("focal_alpha", 0.1, 0.5)
            cfg["training"]["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 4.0)

        # Model architecture (if applicable)
        cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        cfg["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [256, 384, 512])

        # Scheduler
        cfg["training"]["scheduler"] = trial.suggest_categorical(
            "scheduler", ["cosine", "step", "onecycle"]
        )
        cfg["training"]["warmup_epochs"] = trial.suggest_int("warmup_epochs", 1, 5)

        # Reduce epochs for HPO speed
        cfg["training"]["max_epochs"] = min(cfg["training"].get("max_epochs", 30), 10)

        # Run abbreviated training
        from src.data.datamodule import RetinalDataModule
        from src.training.losses import build_loss
        from src.training.metrics import MetricTracker

        try:
            from train import build_model

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            dm = RetinalDataModule(cfg)
            dm.prepare_data()
            dm.setup(stage="fit")
            cfg["model"]["num_classes"] = len(dm.disease_columns)
            cfg["model"]["disease_names"] = dm.disease_columns

            model = build_model(cfg).to(device)
            pos_weight = dm.pos_weights.to(device)
            criterion = build_loss(cfg, pos_weight=pos_weight)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["learning_rate"],
                weight_decay=cfg["training"]["weight_decay"],
            )

            train_loader = dm.train_dataloader()
            val_loader = dm.val_dataloader()

            # Quick training loop (no DDP for HPO)
            best_metric = 0.0
            for epoch in range(cfg["training"]["max_epochs"]):
                model.train()
                for batch_idx, (images, targets) in enumerate(train_loader):
                    if batch_idx > 50:  # Limit batches per epoch for speed
                        break
                    images, targets = images.to(device), targets.to(device)
                    output = model(images)
                    loss = criterion(output, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["training"]["gradient_clip_val"]
                    )
                    optimizer.step()

                # Validate
                model.eval()
                tracker = MetricTracker(threshold=0.5)
                with torch.no_grad():
                    for batch_idx, (images, targets) in enumerate(val_loader):
                        if batch_idx > 20:
                            break
                        images, targets = images.to(device), targets.to(device)
                        output = model(images)
                        tracker.update(output, targets)

                metrics = tracker.compute()
                if cfg.get("evaluation", {}).get("optimal_threshold_search", False):
                    thresholds = tracker.optimize_thresholds(
                        cfg.get("evaluation", {}).get("threshold_search_space")
                    )
                    val_f1 = tracker.compute(thresholds).get("f1_macro", 0.0)
                else:
                    val_f1 = metrics.get("f1_macro", 0.0)
                best_metric = max(best_metric, val_f1)

                trial.report(val_f1, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return best_metric

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    study = optuna.create_study(
        direction=direction,
        study_name="retinal_disease_hpo",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    # Apply best params to config
    best_cfg = deepcopy(base_cfg)
    best_params = study.best_params
    logger.info(f"Best trial: {study.best_trial.number} with {metric}={study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    best_cfg["training"]["learning_rate"] = best_params.get(
        "lr", best_cfg["training"]["learning_rate"]
    )
    best_cfg["training"]["weight_decay"] = best_params.get(
        "weight_decay", best_cfg["training"]["weight_decay"]
    )
    best_cfg["training"]["label_smoothing"] = best_params.get(
        "label_smoothing", best_cfg["training"]["label_smoothing"]
    )
    best_cfg["training"]["gradient_clip_val"] = best_params.get(
        "grad_clip", best_cfg["training"]["gradient_clip_val"]
    )
    best_cfg["training"]["loss"] = best_params.get("loss", best_cfg["training"]["loss"])
    best_cfg["model"]["dropout"] = best_params.get("dropout", best_cfg["model"]["dropout"])
    best_cfg["model"]["hidden_dim"] = best_params.get("hidden_dim", best_cfg["model"]["hidden_dim"])
    best_cfg["training"]["scheduler"] = best_params.get(
        "scheduler", best_cfg["training"]["scheduler"]
    )
    best_cfg["training"]["warmup_epochs"] = best_params.get(
        "warmup_epochs", best_cfg["training"]["warmup_epochs"]
    )

    return best_cfg
