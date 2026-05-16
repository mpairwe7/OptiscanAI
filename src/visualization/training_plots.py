"""
Training Visualization - IEEE Publication Quality.
Loss curves, LR schedules, bias-variance, convergence analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.visualization.ieee_style import (
    IEEE_COLORS,
    METRIC_COLORS,
    add_watermark,
    ieee_figure,
    ieee_style,
    save_ieee,
)


def plot_training_curves(
    history: list[dict],
    save_dir: Path,
    model_name: str = "ViGNN",
):
    """Training & validation loss/metric curves (notebook cell 37 enhanced)."""
    with ieee_style():
        fig, axes = ieee_figure(2, 2, width="double", height_ratio=0.5)
        epochs = [h["epoch"] for h in history]

        # (a) Loss curves
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        axes[0, 0].plot(epochs, train_loss, "o-", color=IEEE_COLORS["blue"], ms=3, label="Train")
        axes[0, 0].plot(epochs, val_loss, "s-", color=IEEE_COLORS["red"], ms=3, label="Val")
        axes[0, 0].fill_between(epochs, train_loss, val_loss, alpha=0.1, color="gray")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("(a) Training & Validation Loss")
        axes[0, 0].legend()

        # (b) F1 scores
        for key, label, color in [
            ("val/f1_macro", "F1 Macro", METRIC_COLORS["f1_macro"]),
            ("val/f1_micro", "F1 Micro", METRIC_COLORS["f1_micro"]),
        ]:
            vals = [h.get(key, 0) for h in history]
            if any(v > 0 for v in vals):
                axes[0, 1].plot(epochs, vals, "o-", ms=3, color=color, label=label)
        best_f1_idx = np.argmax([h.get("val/f1_macro", 0) for h in history])
        best_f1 = history[best_f1_idx].get("val/f1_macro", 0)
        axes[0, 1].axvline(epochs[best_f1_idx], ls="--", color="gray", lw=0.8, alpha=0.5)
        axes[0, 1].annotate(
            f"Best: {best_f1:.4f}",
            xy=(epochs[best_f1_idx], best_f1),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=7,
            color=IEEE_COLORS["red"],
            arrowprops=dict(arrowstyle="->", color=IEEE_COLORS["red"], lw=0.8),
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("F1 Score")
        axes[0, 1].set_title("(b) Validation F1 Scores")
        axes[0, 1].legend()

        # (c) AUC-ROC + mAP
        for key, label, color in [
            ("val/auc_roc", "AUC-ROC", METRIC_COLORS["auc_roc"]),
            ("val/mAP", "mAP", METRIC_COLORS["mAP"]),
        ]:
            vals = [h.get(key, 0) for h in history]
            if any(v > 0 for v in vals):
                axes[1, 0].plot(epochs, vals, "o-", ms=3, color=color, label=label)
        axes[1, 0].axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.3, label="Random baseline")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_title("(c) AUC-ROC & mAP")
        axes[1, 0].legend(fontsize=7)

        # (d) Learning rate
        lrs = [h.get("lr", 0) for h in history]
        axes[1, 1].plot(epochs, lrs, "o-", ms=3, color=IEEE_COLORS["teal"])
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("(d) Learning Rate Schedule")

        fig.suptitle(f"{model_name} - Training Curves", fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_training_curves")


def plot_bias_variance(
    history: list[dict],
    save_dir: Path,
    model_name: str = "ViGNN",
):
    """Bias-variance trade-off analysis (notebook cell 42 enhanced)."""
    with ieee_style():
        fig, axes = ieee_figure(1, 3, width="double", height_ratio=0.5)

        epochs = [h["epoch"] for h in history]
        train_f1 = [h.get("train/f1_macro", 0) for h in history]
        val_f1 = [h.get("val/f1_macro", 0) for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]

        # (a) Train vs Val F1 gap
        axes[0].plot(epochs, train_f1, "o-", ms=3, color=IEEE_COLORS["blue"], label="Train F1")
        axes[0].plot(epochs, val_f1, "s-", ms=3, color=IEEE_COLORS["red"], label="Val F1")
        gap = [t - v for t, v in zip(train_f1, val_f1)]
        axes[0].fill_between(
            epochs, val_f1, train_f1, alpha=0.15, color=IEEE_COLORS["orange"], label="Gap (overfit)"
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("F1 Macro")
        axes[0].set_title("(a) Generalization Gap")
        axes[0].legend(fontsize=7)

        # (b) Loss gap
        axes[1].plot(epochs, train_loss, "o-", ms=3, color=IEEE_COLORS["blue"], label="Train Loss")
        axes[1].plot(epochs, val_loss, "s-", ms=3, color=IEEE_COLORS["red"], label="Val Loss")
        axes[1].fill_between(epochs, train_loss, val_loss, alpha=0.15, color=IEEE_COLORS["red"])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("(b) Loss Divergence")
        axes[1].legend(fontsize=7)

        # (c) Diagnostic summary
        if len(history) > 1:
            final_gap = gap[-1]
            trend = "increasing" if len(gap) > 2 and gap[-1] > gap[-2] else "stable"
            status = (
                "Overfitting"
                if final_gap > 0.1
                else "Underfitting" if val_f1[-1] < 0.15 else "Balanced"
            )
            color_map = {"Overfitting": "red", "Underfitting": "orange", "Balanced": "green"}
            text = (
                f"Diagnostic: {status}\n\n"
                f"Train F1:  {train_f1[-1]:.4f}\n"
                f"Val F1:    {val_f1[-1]:.4f}\n"
                f"Gap:       {final_gap:.4f}\n"
                f"Trend:     {trend}\n\n"
                f"Recommendation:\n"
            )
            if status == "Overfitting":
                text += "  - Increase dropout\n  - Add data augmentation\n  - Reduce model size"
            elif status == "Underfitting":
                text += "  - Increase model capacity\n  - Train longer\n  - Lower learning rate"
            else:
                text += "  - Continue training\n  - Consider ensembling"

            axes[2].text(
                0.05,
                0.95,
                text,
                transform=axes[2].transAxes,
                fontsize=7,
                va="top",
                family="monospace",
                bbox=dict(
                    boxstyle="round", fc=color_map.get(status, "#f0f0f0"), ec="#cccccc", alpha=0.2
                ),
            )
        axes[2].set_axis_off()
        axes[2].set_title("(c) Bias-Variance Diagnostic")

        fig.suptitle(
            f"{model_name} - Bias-Variance Analysis", fontsize=11, fontweight="bold", y=1.02
        )
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_bias_variance")


def plot_convergence_analysis(
    history: list[dict],
    save_dir: Path,
    model_name: str = "ViGNN",
):
    """Convergence speed and efficiency analysis."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.5)

        epochs = [h["epoch"] for h in history]
        val_f1 = [h.get("val/f1_macro", 0) for h in history]
        [h["val_loss"] for h in history]

        # (a) Rate of improvement
        improvements = [0] + [val_f1[i] - val_f1[i - 1] for i in range(1, len(val_f1))]
        colors_bar = [IEEE_COLORS["green"] if v > 0 else IEEE_COLORS["red"] for v in improvements]
        axes[0].bar(epochs, improvements, color=colors_bar, edgecolor="black", lw=0.3, alpha=0.8)
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("F1 Improvement")
        axes[0].set_title("(a) Per-Epoch F1 Improvement")

        # (b) Normalized convergence
        if max(val_f1) > 0:
            norm_f1 = [v / max(val_f1) for v in val_f1]
        else:
            norm_f1 = val_f1
        axes[1].plot(epochs, norm_f1, "o-", ms=3, color=IEEE_COLORS["blue"])
        axes[1].axhline(0.95, ls="--", color="gray", lw=0.8, alpha=0.5, label="95% convergence")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Normalized F1")
        axes[1].set_title("(b) Convergence Progress")
        axes[1].legend(fontsize=7)

        fig.suptitle(f"{model_name} - Convergence Analysis", fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_convergence_analysis")


def generate_all_training_plots(
    history: list[dict],
    save_dir: str | Path = "outputs/plots/training",
    model_name: str = "ViGNN",
):
    """Generate all training visualization plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating training plots -> {save_dir}")

    plot_training_curves(history, save_dir, model_name)
    print("  [1/3] Training curves")

    plot_bias_variance(history, save_dir, model_name)
    print("  [2/3] Bias-variance analysis")

    plot_convergence_analysis(history, save_dir, model_name)
    print("  [3/3] Convergence analysis")

    print(f"  Training plots saved to {save_dir}")
