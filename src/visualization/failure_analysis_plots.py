"""
Failure analysis visualizations - error breakdown, confusion patterns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.visualization.ieee_style import (
    IEEE_COLORS,
    add_watermark,
    ieee_figure,
    ieee_style,
    save_ieee,
)


def plot_per_class_error_breakdown(
    y_true: np.ndarray, y_pred: np.ndarray, disease_names: list[str],
    save_dir: Path, top_k: int = 20,
):
    """Stacked bar: FP + FN per class, sorted by total errors."""
    with ieee_style():
        fig, ax = ieee_figure(1, 1, width="double", height_ratio=0.55)
        fp = ((y_pred == 1) & (y_true == 0)).sum(axis=0)
        fn = ((y_pred == 0) & (y_true == 1)).sum(axis=0)
        total_errors = fp + fn
        top_idx = np.argsort(total_errors)[::-1][:top_k]

        names = [disease_names[i] for i in top_idx]
        x = np.arange(len(names))
        ax.bar(x, fp[top_idx], label="False Positive", color=IEEE_COLORS["orange"], edgecolor="black", lw=0.3)
        ax.bar(x, fn[top_idx], bottom=fp[top_idx], label="False Negative", color=IEEE_COLORS["red"], edgecolor="black", lw=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Error Count")
        ax.set_title(f"Per-Class Error Breakdown (Top {top_k})")
        ax.legend(fontsize=7)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_error_breakdown")


def plot_error_by_label_cardinality(
    y_true: np.ndarray, y_pred: np.ndarray, save_dir: Path,
):
    """Error rate vs number of positive labels per sample."""
    with ieee_style():
        fig, ax = ieee_figure(1, 1, width="single", height_ratio=0.8)
        cardinalities = y_true.sum(axis=1).astype(int)
        unique_cards = sorted(set(cardinalities))

        error_rates, sample_counts = [], []
        for c in unique_cards:
            mask = cardinalities == c
            if mask.sum() == 0:
                continue
            f1 = f1_score(y_true[mask], y_pred[mask], average="samples", zero_division=0)
            error_rates.append(1 - f1)
            sample_counts.append(mask.sum())

        ax.bar(range(len(unique_cards)), error_rates, color=IEEE_COLORS["blue"],
               edgecolor="black", lw=0.3, alpha=0.8)
        ax.set_xticks(range(len(unique_cards)))
        ax.set_xticklabels(unique_cards, fontsize=7)
        ax.set_xlabel("Number of Positive Labels")
        ax.set_ylabel("Error Rate (1 - F1 Samples)")
        ax.set_title("Error Rate by Label Cardinality")

        # Annotate sample counts
        for i, (er, sc) in enumerate(zip(error_rates, sample_counts)):
            ax.text(i, er + 0.01, f"n={sc}", ha="center", fontsize=6, color="gray")

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_error_by_cardinality")


def generate_all_failure_analysis(
    y_true: np.ndarray, y_pred: np.ndarray, disease_names: list[str],
    save_dir: str | Path = "outputs/plots/evaluation",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating failure analysis plots -> {save_dir}")
    plot_per_class_error_breakdown(y_true, y_pred, disease_names, save_dir)
    print("  [1/2] Error breakdown")
    plot_error_by_label_cardinality(y_true, y_pred, save_dir)
    print("  [2/2] Error by cardinality")
