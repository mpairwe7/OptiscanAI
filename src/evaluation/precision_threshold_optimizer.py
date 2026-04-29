"""
Precision-floor threshold optimization for multi-label medical classification.

Standard F1-maximizing threshold search drives thresholds down, tanking precision.
This module finds per-class thresholds that satisfy a minimum precision floor,
then maximizes recall subject to that constraint.

Usage:
    from src.evaluation.precision_threshold_optimizer import optimize_thresholds_with_precision_floor

    thresholds, report = optimize_thresholds_with_precision_floor(
        probs, labels, min_precision=0.10
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def optimize_thresholds_with_precision_floor(
    probs: np.ndarray,
    labels: np.ndarray,
    min_precision: float = 0.10,
    search_space: Optional[np.ndarray] = None,
    disease_names: Optional[list[str]] = None,
    fallback_threshold: float = 0.95,
) -> Tuple[np.ndarray, Dict]:
    """Find per-class thresholds with a minimum precision guarantee.

    Algorithm per class:
      1. Sweep thresholds from high (0.95) to low (0.05).
      2. At each threshold, compute precision.
      3. Find the LOWEST threshold where precision >= min_precision.
      4. If no threshold meets the floor, set to fallback_threshold (effectively disable).

    This is the inverse of F1 optimization: we start conservative (high threshold)
    and only lower it as far as precision allows.

    Parameters
    ----------
    probs : np.ndarray
        Predicted probabilities [N, C].
    labels : np.ndarray
        Ground truth binary labels [N, C].
    min_precision : float
        Minimum acceptable precision per class. Default 0.10.
    search_space : np.ndarray | None
        Thresholds to search. Default: 0.05 to 0.95 in steps of 0.02.
    disease_names : list[str] | None
        Class names for reporting.
    fallback_threshold : float
        Threshold for classes that cannot meet precision floor.

    Returns
    -------
    thresholds : np.ndarray
        Per-class optimized thresholds [C].
    report : dict
        Detailed per-class report with precision, recall, F1 at chosen threshold.
    """
    if search_space is None:
        search_space = np.arange(0.05, 0.96, 0.02, dtype=np.float32)

    num_classes = probs.shape[1]
    thresholds = np.full(num_classes, fallback_threshold, dtype=np.float32)

    # Sort search space descending (start conservative)
    search_desc = np.sort(search_space)[::-1]

    per_class_report = {}

    for c in range(num_classes):
        y_true = labels[:, c]
        y_prob = probs[:, c]
        n_positive = int(y_true.sum())
        class_name = disease_names[c] if disease_names and c < len(disease_names) else f"class_{c}"

        # Skip classes with no positive samples
        if n_positive == 0:
            per_class_report[class_name] = {
                "threshold": fallback_threshold,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "n_positive": 0,
                "status": "no_positives",
            }
            continue

        # Find lowest threshold meeting precision floor
        best_threshold = fallback_threshold
        best_recall = 0.0
        best_precision = 0.0
        best_f1 = 0.0
        found_valid = False

        for t in search_desc:
            y_pred = (y_prob >= t).astype(np.float32)
            n_pred_positive = y_pred.sum()

            if n_pred_positive == 0:
                continue

            tp = ((y_pred == 1) & (y_true == 1)).sum()
            prec = tp / n_pred_positive
            rec = tp / n_positive if n_positive > 0 else 0.0

            if prec >= min_precision:
                # This threshold meets precision floor
                # Keep going lower to maximize recall while maintaining precision
                if rec >= best_recall:
                    best_threshold = float(t)
                    best_precision = float(prec)
                    best_recall = float(rec)
                    best_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    found_valid = True

        thresholds[c] = best_threshold

        per_class_report[class_name] = {
            "threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
            "f1": float(best_f1),
            "n_positive": n_positive,
            "status": "optimized" if found_valid else "fallback",
        }

    # Summary stats
    valid_thresholds = [r for r in per_class_report.values() if r["status"] == "optimized"]
    report = {
        "per_class": per_class_report,
        "summary": {
            "min_precision_floor": min_precision,
            "classes_optimized": len(valid_thresholds),
            "classes_fallback": num_classes - len(valid_thresholds),
            "mean_threshold": float(thresholds.mean()),
            "mean_precision": float(np.mean([r["precision"] for r in valid_thresholds])) if valid_thresholds else 0.0,
            "mean_recall": float(np.mean([r["recall"] for r in valid_thresholds])) if valid_thresholds else 0.0,
            "mean_f1": float(np.mean([r["f1"] for r in valid_thresholds])) if valid_thresholds else 0.0,
        },
    }

    logger.info(
        f"Threshold optimization: {len(valid_thresholds)}/{num_classes} classes met "
        f"precision floor {min_precision}. Mean threshold: {thresholds.mean():.3f}"
    )

    return thresholds, report


def save_thresholds(
    thresholds: np.ndarray,
    report: Dict,
    output_path: str = "outputs/thresholds_optimized.json",
):
    """Save optimized thresholds and report to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    data = {
        "thresholds": thresholds.tolist(),
        **report,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved optimized thresholds to {output_path}")
    return output_path
