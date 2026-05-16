"""
AdvancedEarlyStopping - Extracted from notebook cell 29.
Multi-metric monitoring, adaptive patience, overfitting detection.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Optional

import torch.nn as nn


class AdvancedEarlyStopping:
    """
    Advanced early stopping with performance analysis.
    - Monitors multiple metrics (F1, AUC, Loss)
    - Adaptive patience
    - Performance degradation & overfitting detection
    - Best-weight restoration
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        min_epochs: int = 3,
        mode: str = "max",
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score: Optional[float] = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.history: dict[str, list] = defaultdict(list)
        self.analysis_results: dict = {}

    def __call__(self, epoch: int, metrics: dict, model: nn.Module = None):
        """Returns (should_stop, is_best)."""
        primary = "f1" if "f1" in metrics else list(metrics.keys())[0]
        score = metrics.get(primary, 0)

        for k, v in metrics.items():
            self.history[k].append(v)
        self.history["epoch"].append(epoch)

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if model and self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
            return False, True

        improved = (
            score > (self.best_score + self.min_delta)
            if self.mode == "max"
            else score < (self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if model and self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
            return False, True

        self.counter += 1
        if epoch >= self.min_epochs and self.counter >= self.patience:
            self.early_stop = True
            self._analyze()
        return self.early_stop, False

    def restore_best(self, model: nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

    def _analyze(self):
        self.analysis_results = {
            "stopped_early": True,
            "best_epoch": self.best_epoch,
            "total_epochs": len(self.history["epoch"]),
            "patience_exhausted": self.counter,
            "insights": [],
        }
        if "loss" in self.history and len(self.history["loss"]) >= 3:
            recent = self.history["loss"][-3:]
            if all(recent[i] > recent[i - 1] for i in range(1, 3)):
                self.analysis_results["insights"].append("Loss increasing - model diverging")
        if "f1" in self.history and len(self.history["f1"]) >= 3:
            recent = self.history["f1"][-3:]
            if all(recent[i] < recent[i - 1] for i in range(1, 3)):
                self.analysis_results["insights"].append("F1 declining - overfitting")

    def get_analysis(self) -> dict:
        return self.analysis_results
