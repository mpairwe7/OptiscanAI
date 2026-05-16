"""
Gradient and weight monitoring visualizations.
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


def plot_gradient_norms(
    grad_norms: list[float], save_dir: Path,
):
    """Gradient norm over training steps."""
    with ieee_style():
        fig, ax = ieee_figure(1, 1, width="single", height_ratio=0.7)
        steps = range(len(grad_norms))
        ax.plot(steps, grad_norms, lw=0.8, color=IEEE_COLORS["blue"], alpha=0.7)
        # Smoothed
        if len(grad_norms) > 20:
            kernel = max(len(grad_norms) // 50, 5)
            smoothed = np.convolve(grad_norms, np.ones(kernel) / kernel, mode="valid")
            ax.plot(range(len(smoothed)), smoothed, lw=1.5, color=IEEE_COLORS["red"], label="Smoothed")

        ax.axhline(100, ls="--", color="red", lw=0.8, alpha=0.5, label="Explosion threshold")
        ax.axhline(1e-5, ls="--", color="orange", lw=0.8, alpha=0.5, label="Vanishing threshold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm History")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_gradient_norms")


def plot_lr_vs_loss(
    history: list[dict], save_dir: Path,
):
    """Learning rate overlaid with loss curve."""
    with ieee_style():
        fig, ax1 = ieee_figure(1, 1, width="single", height_ratio=0.8)
        epochs = [h["epoch"] for h in history]
        loss = [h["train_loss"] for h in history]
        lrs = [h.get("lr", 0) for h in history]

        color_loss = IEEE_COLORS["blue"]
        color_lr = IEEE_COLORS["red"]

        ax1.plot(epochs, loss, "o-", ms=3, color=color_loss, label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=color_loss)
        ax1.tick_params(axis="y", labelcolor=color_loss)

        ax2 = ax1.twinx()
        ax2.plot(epochs, lrs, "s--", ms=3, color=color_lr, alpha=0.7, label="Learning Rate")
        ax2.set_ylabel("Learning Rate", color=color_lr)
        ax2.tick_params(axis="y", labelcolor=color_lr)
        ax2.set_yscale("log")

        ax1.set_title("Loss vs Learning Rate Schedule")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_lr_vs_loss")
