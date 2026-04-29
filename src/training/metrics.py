"""
Evaluation metrics for multi-label retinal disease classification.
Extracted from notebook cells 30-31.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    hamming_loss,
    average_precision_score,
)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def _normalize_thresholds(
    threshold: float | list[float] | np.ndarray, num_classes: int
) -> np.ndarray:
    """Expand a scalar threshold or validate per-class thresholds."""
    if np.isscalar(threshold):
        return np.full(num_classes, float(threshold), dtype=np.float32)

    thresholds = np.asarray(threshold, dtype=np.float32).reshape(-1)
    if thresholds.size != num_classes:
        raise ValueError(
            f"Expected {num_classes} thresholds, got {thresholds.size}"
        )
    return thresholds


def compute_multilabel_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float | list[float] | np.ndarray = 0.5,
) -> dict[str, float]:
    """Compute multilabel metrics for scalar or per-class thresholds."""
    thresholds = _normalize_thresholds(threshold, probs.shape[1])
    preds = (probs > thresholds.reshape(1, -1)).astype(np.float32)

    metrics = {}

    metrics["f1_macro"] = f1_score(
        targets, preds, average="macro", zero_division=0
    )
    metrics["f1_micro"] = f1_score(
        targets, preds, average="micro", zero_division=0
    )
    metrics["f1_samples"] = f1_score(
        targets, preds, average="samples", zero_division=0
    )

    metrics["precision_macro"] = precision_score(
        targets, preds, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        targets, preds, average="macro", zero_division=0
    )

    metrics["precision_micro"] = precision_score(
        targets, preds, average="micro", zero_division=0
    )
    metrics["recall_micro"] = recall_score(
        targets, preds, average="micro", zero_division=0
    )

    # Subset accuracy: fraction of samples where ALL labels match exactly
    metrics["accuracy_subset"] = float((preds == targets).all(axis=1).mean())
    # Sample accuracy: average per-sample label match rate
    metrics["accuracy_sample"] = float(
        ((preds == targets).sum(axis=1) / targets.shape[1]).mean()
    )

    # Per-class accuracy averaged (macro): mean of per-class (TP+TN)/(TP+TN+FP+FN)
    per_class_acc = []
    for c in range(targets.shape[1]):
        per_class_acc.append(accuracy_score(targets[:, c], preds[:, c]))
    metrics["accuracy_macro"] = float(np.mean(per_class_acc))

    # Micro accuracy: total correct label predictions / total label predictions
    metrics["accuracy_micro"] = float(
        (preds == targets).sum() / max(targets.size, 1)
    )

    # Overall accuracy (Jaccard-style): mean IoU across samples
    intersection = (preds * targets).sum(axis=1)
    union = ((preds + targets) > 0).sum(axis=1).astype(float)
    union = np.where(union == 0, 1, union)
    metrics["accuracy_jaccard"] = float((intersection / union).mean())

    metrics["hamming_loss"] = hamming_loss(targets, preds)
    metrics["threshold_mean"] = float(thresholds.mean())
    metrics["threshold_min"] = float(thresholds.min())
    metrics["threshold_max"] = float(thresholds.max())

    try:
        valid_cols = (targets.sum(axis=0) > 0) & (targets.sum(axis=0) < len(targets))
        if valid_cols.sum() >= 2:
            metrics["auc_roc"] = roc_auc_score(
                targets[:, valid_cols],
                probs[:, valid_cols],
                average="macro",
            )
            metrics["mAP"] = average_precision_score(
                targets[:, valid_cols],
                probs[:, valid_cols],
                average="macro",
            )
        else:
            metrics["auc_roc"] = 0.0
            metrics["mAP"] = 0.0
    except ValueError:
        metrics["auc_roc"] = 0.0
        metrics["mAP"] = 0.0

    return metrics


def find_optimal_thresholds(
    targets: np.ndarray,
    probs: np.ndarray,
    search_space: list[float] | np.ndarray | None = None,
    default_threshold: float = 0.5,
) -> np.ndarray:
    """Find per-class thresholds that maximize classwise F1 on a validation set."""
    if search_space is None:
        search_space = np.arange(0.05, 0.96, 0.05, dtype=np.float32)
    else:
        search_space = np.asarray(search_space, dtype=np.float32)

    thresholds = np.full(probs.shape[1], float(default_threshold), dtype=np.float32)

    for class_idx in range(probs.shape[1]):
        y_true = targets[:, class_idx]
        y_prob = probs[:, class_idx]

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        best_threshold = float(default_threshold)
        best_score = -1.0

        for candidate in search_space:
            y_pred = (y_prob > candidate).astype(np.float32)
            score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score + 1e-12:
                best_score = score
                best_threshold = float(candidate)
            elif np.isclose(score, best_score, atol=1e-12):
                current_gap = abs(float(candidate) - float(default_threshold))
                best_gap = abs(best_threshold - float(default_threshold))
                if current_gap < best_gap:
                    best_threshold = float(candidate)

        thresholds[class_idx] = best_threshold

    return thresholds


class MetricTracker:
    """Accumulates predictions across batches and computes epoch-level metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.all_logits = []
        self.all_targets = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        self.all_logits.append(logits.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return concatenated logits, targets, and probabilities."""
        if not self.all_logits:
            empty = np.empty((0, 0), dtype=np.float32)
            return empty, empty, empty

        logits = torch.cat(self.all_logits, dim=0).numpy()
        targets = torch.cat(self.all_targets, dim=0).numpy()
        probs = _sigmoid(logits)
        return logits, targets, probs

    def compute(
        self, threshold: float | list[float] | np.ndarray | None = None
    ) -> dict[str, float]:
        """Compute all metrics from accumulated predictions."""
        _, targets, probs = self.get_arrays()
        if targets.size == 0:
            return {}
        return compute_multilabel_metrics(
            targets, probs, threshold=self.threshold if threshold is None else threshold
        )

    def optimize_thresholds(
        self, search_space: list[float] | np.ndarray | None = None
    ) -> np.ndarray:
        """Find per-class thresholds from accumulated validation outputs."""
        _, targets, probs = self.get_arrays()
        if targets.size == 0:
            return np.array([], dtype=np.float32)
        return find_optimal_thresholds(
            targets, probs, search_space=search_space, default_threshold=self.threshold
        )
