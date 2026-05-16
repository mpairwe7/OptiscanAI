"""
Temperature scaling calibration and ECE computation.
Post-hoc calibration (Guo et al. 2017) for reliable prediction confidence.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Learns a single temperature parameter to calibrate model outputs."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-4)

    @torch.no_grad()
    def calibrate(
        self, model: nn.Module, val_loader: DataLoader, device: torch.device, max_iter: int = 50
    ) -> float:
        """Optimize temperature on validation set using NLL."""
        model.eval()
        all_logits, all_targets = [], []
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

        logits_cat = torch.cat(all_logits)
        targets_cat = torch.cat(all_targets)
        self.to(logits_cat.device)

        # Optimize temperature with LBFGS
        self.temperature.data.fill_(1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits_cat / self.temperature
            loss = nn.functional.binary_cross_entropy_with_logits(scaled, targets_cat)
            loss.backward()
            return loss

        optimizer.step(closure)
        temp = self.temperature.item()
        logger.info(f"Calibrated temperature: {temp:.4f}")
        return temp


def compute_ece(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> tuple[float, dict]:
    """
    Expected Calibration Error for multi-label classification.
    Returns (ece_value, reliability_data_for_plotting).
    """
    # Flatten to binary predictions
    probs_flat = probs.flatten()
    targets_flat = targets.flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs_flat >= lo) & (probs_flat < hi)
        count = mask.sum()
        if count > 0:
            bin_accuracies[i] = targets_flat[mask].mean()
            bin_confidences[i] = probs_flat[mask].mean()
            bin_counts[i] = count

    # Weighted average of |accuracy - confidence|
    total = bin_counts.sum()
    ece = np.sum(bin_counts / max(total, 1) * np.abs(bin_accuracies - bin_confidences))

    return float(ece), {
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_confidences": bin_confidences.tolist(),
        "bin_counts": bin_counts.tolist(),
        "bin_boundaries": bin_boundaries.tolist(),
    }


def bootstrap_confidence_interval(
    metric_fn,
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for any metric function.
    Returns (mean, lower, upper)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred_or_prob[idx])
            scores.append(s)
        except (ValueError, IndexError):
            continue
    if not scores:
        return 0.0, 0.0, 0.0
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return float(scores.mean()), float(np.percentile(scores, alpha * 100)), float(np.percentile(scores, (1 - alpha) * 100))
