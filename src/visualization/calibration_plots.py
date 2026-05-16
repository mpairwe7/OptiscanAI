"""
Calibration visualizations - reliability diagrams, ECE, confidence histograms.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.visualization.ieee_style import (
    IEEE_COLORS,
    add_watermark,
    ieee_figure,
    ieee_style,
    save_ieee,
)


def plot_reliability_diagram(
    reliability_data: dict,
    ece: float,
    save_dir: Path,
    title_suffix: str = "",
):
    """Reliability diagram with gap shading and ECE annotation."""
    with ieee_style():
        fig, ax = ieee_figure(1, 1, width="single", height_ratio=1.0)
        bins = np.array(reliability_data["bin_confidences"])
        accs = np.array(reliability_data["bin_accuracies"])
        counts = np.array(reliability_data["bin_counts"])
        mask = counts > 0

        ax.bar(bins[mask], accs[mask], width=0.06, alpha=0.6, color=IEEE_COLORS["blue"],
               edgecolor="black", lw=0.3, label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
        # Gap shading
        for b, a in zip(bins[mask], accs[mask]):
            ax.plot([b, b], [b, a], color=IEEE_COLORS["red"], lw=1.5, alpha=0.6)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Reliability Diagram{title_suffix}")
        ax.text(0.05, 0.92, f"ECE = {ece:.4f}", transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        add_watermark(fig)
        save_ieee(fig, save_dir / f"fig_reliability_diagram{title_suffix.replace(' ', '_').lower()}")


def plot_calibration_before_after(
    before_data: dict, before_ece: float,
    after_data: dict, after_ece: float,
    save_dir: Path,
):
    """Side-by-side reliability diagrams: before and after temperature scaling."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.6)
        for ax, data, ece, title in [
            (axes[0], before_data, before_ece, "(a) Before Calibration"),
            (axes[1], after_data, after_ece, "(b) After Temperature Scaling"),
        ]:
            bins = np.array(data["bin_confidences"])
            accs = np.array(data["bin_accuracies"])
            counts = np.array(data["bin_counts"])
            mask = counts > 0
            ax.bar(bins[mask], accs[mask], width=0.06, alpha=0.6, color=IEEE_COLORS["blue"],
                   edgecolor="black", lw=0.3)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Actual Accuracy")
            ax.set_title(title)
            ax.text(0.05, 0.92, f"ECE = {ece:.4f}", transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_calibration_before_after")


def plot_confidence_histogram(
    y_prob: np.ndarray, y_true: np.ndarray, save_dir: Path,
):
    """Positive vs negative confidence distribution."""
    with ieee_style():
        fig, ax = ieee_figure(1, 1, width="single", height_ratio=0.8)
        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]
        bins = np.linspace(0, 1, 40)
        ax.hist(neg_probs, bins, alpha=0.6, color=IEEE_COLORS["blue"], density=True, label=f"Neg (n={len(neg_probs):,})")
        ax.hist(pos_probs, bins, alpha=0.6, color=IEEE_COLORS["red"], density=True, label=f"Pos (n={len(pos_probs):,})")
        ax.axvline(0.5, ls="--", color="black", lw=0.8, alpha=0.5)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title("Confidence Distribution: Positive vs Negative")
        ax.legend(fontsize=7)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_confidence_histogram")
