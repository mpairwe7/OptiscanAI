"""
BiasVarianceMonitor - Extracted from notebook cell 42.
Monitors train/val gap for overfitting/underfitting diagnosis.
"""
from __future__ import annotations

import numpy as np


class BiasVarianceMonitor:
    """
    Monitor bias-variance trade-off during training.
    Diagnoses overfitting/underfitting and provides recommendations.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.train_scores: list[float] = []
        self.val_scores: list[float] = []
        self.test_score: float | None = None

    def update(self, train_score: float, val_score: float):
        self.train_scores.append(train_score)
        self.val_scores.append(val_score)

    def set_test_score(self, test_score: float):
        self.test_score = test_score

    def analyze(self) -> dict:
        if len(self.train_scores) < 3:
            return {"status": "insufficient_data"}

        final_train = self.train_scores[-1]
        final_val = self.val_scores[-1]
        best_val = max(self.val_scores)
        gap = final_train - final_val
        recent_std = float(np.std(self.val_scores[-5:]) if len(self.val_scores) >= 5
                           else np.std(self.val_scores))

        diagnosis = self._diagnose(final_train, final_val, gap, recent_std)
        health = self._health_score(gap, recent_std)

        return {
            "model": self.model_name,
            "final_train_f1": final_train,
            "final_val_f1": final_val,
            "best_val_f1": best_val,
            "train_val_gap": gap,
            "val_std": recent_std,
            "test_f1": self.test_score,
            "diagnosis": diagnosis,
            "recommendations": self._recommend(diagnosis),
            "health_score": health,
        }

    def _diagnose(self, train: float, val: float, gap: float, std: float) -> str:
        if gap > 0.15:
            return "SEVERE_OVERFITTING"
        if gap > 0.10:
            return "MODERATE_OVERFITTING"
        if train < 0.70 and val < 0.70:
            return "UNDERFITTING"
        if std > 0.05:
            return "HIGH_VARIANCE"
        if gap < 0.08 and val > 0.73:
            return "EXCELLENT"
        if 0.05 <= gap <= 0.10 and val > 0.70:
            return "OPTIMAL"
        return "NEEDS_MONITORING"

    def _recommend(self, diagnosis: str) -> list[str]:
        recs = {
            "SEVERE_OVERFITTING": [
                "Increase dropout to 0.3", "Add more augmentation",
                "Reduce model complexity", "Stronger L2 regularization",
            ],
            "MODERATE_OVERFITTING": [
                "Increase dropout to 0.2", "Reduce LR by 50%", "More training data",
            ],
            "UNDERFITTING": [
                "Increase model capacity", "Train longer", "Lower dropout",
            ],
            "HIGH_VARIANCE": [
                "Reduce LR", "Increase batch size", "Add batch normalization",
            ],
            "OPTIMAL": ["Model is well-regularized", "Ready for deployment"],
            "EXCELLENT": ["Outstanding performance", "Deploy with confidence"],
            "NEEDS_MONITORING": ["Continue monitoring", "Check learning curves"],
        }
        return recs.get(diagnosis, ["Unknown"])

    def _health_score(self, gap: float, std: float) -> float:
        penalty = max(0, (gap - 0.10) * 300) + max(0, (std - 0.03) * 500)
        return max(0, min(100, 100 - penalty))
