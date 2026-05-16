"""
Precision Rescue Visualizations — IEEE Publication Quality.

Plots for the v2 precision-rescue pipeline:
  - Before/after precision comparison
  - Per-class threshold heatmap (precision-floor optimized)
  - ASL loss landscape vs Focal Loss
  - Class filtering impact (45 -> 25-28 classes)
  - Staged unfreezing learning dynamics
  - Precision-Recall trade-off surface
  - TTA improvement analysis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.ieee_style import (
    CB_PALETTE,
    IEEE_COLORS,
    METRIC_COLORS,
    add_watermark,
    ieee_figure,
    ieee_style,
    save_ieee,
)

# ---------------------------------------------------------------------------
# 1. Before/After Precision Comparison
# ---------------------------------------------------------------------------


def plot_before_after_comparison(
    before_metrics: dict[str, dict],
    after_metrics: dict[str, dict],
    save_dir: Path,
):
    """Side-by-side bar chart: v1 (45 classes) vs v2 (precision-rescue).

    Parameters
    ----------
    before_metrics : dict
        Model name -> {f1_macro, precision_macro, recall_macro, auc_roc}
    after_metrics : dict
        Same structure for v2 models.
    save_dir : Path
        Output directory.
    """
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.55)
        metrics_to_plot = [
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "auc_roc",
            "accuracy_macro",
        ]
        labels = ["Precision", "Recall", "F1", "AUC", "Accuracy"]

        # (a) Before (v1)
        ax = axes[0]
        x = np.arange(len(labels))
        w = 0.8 / max(len(before_metrics), 1)
        for i, (model, vals) in enumerate(before_metrics.items()):
            bar_vals = [vals.get(m, 0) for m in metrics_to_plot]
            ax.bar(
                x + i * w,
                bar_vals,
                w,
                label=model,
                color=CB_PALETTE[i % len(CB_PALETTE)],
                edgecolor="black",
                lw=0.3,
            )
        ax.set_xticks(x + w * len(before_metrics) / 2)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("(a) Before: 45 Classes (v1)")
        ax.legend(fontsize=6, loc="upper right")
        ax.axhline(0.10, ls=":", color="red", lw=0.8, alpha=0.6, label="Precision floor")

        # (b) After (v2)
        ax = axes[1]
        for i, (model, vals) in enumerate(after_metrics.items()):
            bar_vals = [vals.get(m, 0) for m in metrics_to_plot]
            ax.bar(
                x + i * w,
                bar_vals,
                w,
                label=model,
                color=CB_PALETTE[i % len(CB_PALETTE)],
                edgecolor="black",
                lw=0.3,
            )
        ax.set_xticks(x + w * max(len(after_metrics), 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("(b) After: Precision Rescue (v2)")
        ax.legend(fontsize=6, loc="upper right")
        ax.axhline(0.10, ls=":", color="red", lw=0.8, alpha=0.6)

        fig.suptitle("Precision Rescue: Before vs After", fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_before_after_comparison")


# ---------------------------------------------------------------------------
# 2. Per-Class Threshold Heatmap
# ---------------------------------------------------------------------------


def plot_threshold_heatmap(
    threshold_report: dict,
    save_dir: Path,
):
    """Heatmap showing optimized thresholds + achieved precision/recall per class.

    Parameters
    ----------
    threshold_report : dict
        Output from precision_threshold_optimizer with 'per_class' key.
    save_dir : Path
        Output directory.
    """
    per_class = threshold_report.get("per_class", {})
    if not per_class:
        return

    with ieee_style():
        names = list(per_class.keys())
        thresholds = [per_class[n]["threshold"] for n in names]
        precisions = [per_class[n]["precision"] for n in names]
        recalls = [per_class[n]["recall"] for n in names]
        statuses = [per_class[n]["status"] for n in names]
        n_positive = [per_class[n]["n_positive"] for n in names]

        data = np.array([thresholds, precisions, recalls]).T
        col_labels = ["Threshold", "Precision", "Recall"]

        fig, axes = ieee_figure(1, 2, width="double", height_ratio=max(0.4, len(names) * 0.06))

        # (a) Heatmap
        ax = axes[0]
        im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(col_labels, fontsize=7)
        for i in range(len(names)):
            for j in range(3):
                color = "white" if data[i, j] > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=color,
                    fontweight="bold",
                )
        ax.set_title("(a) Per-Class Optimized Thresholds")
        fig.colorbar(im, ax=ax, fraction=0.03)

        # (b) Samples + status bar chart
        ax = axes[1]
        colors = [
            IEEE_COLORS["green"] if s == "optimized" else IEEE_COLORS["red"] for s in statuses
        ]
        ax.barh(range(len(names)), n_positive, color=colors, edgecolor="black", lw=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Positive Samples")
        ax.set_title("(b) Training Samples per Class")

        legend_elements = [
            mpatches.Patch(color=IEEE_COLORS["green"], label="Optimized"),
            mpatches.Patch(color=IEEE_COLORS["red"], label="Fallback (precision floor unmet)"),
        ]
        ax.legend(handles=legend_elements, fontsize=6, loc="lower right")

        fig.suptitle(
            "Precision-Floor Threshold Optimization", fontsize=11, fontweight="bold", y=1.02
        )
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_threshold_heatmap")


# ---------------------------------------------------------------------------
# 3. ASL vs Focal Loss Comparison
# ---------------------------------------------------------------------------


def plot_asl_vs_focal_loss(
    save_dir: Path,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    focal_gamma: float = 2.0,
):
    """Compare ASL and Focal Loss weighting curves.

    Shows how ASL with gamma_pos=0, gamma_neg=4 penalizes false positives
    much harder than standard Focal Loss.
    """
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.55)

        p = np.linspace(0.01, 0.99, 200)

        # (a) Loss weight for NEGATIVE examples (false positives)
        ax = axes[0]
        focal_weight_neg = p**focal_gamma  # (1 - (1-p))^gamma = p^gamma
        asl_weight_neg = p**gamma_neg
        ax.plot(
            p,
            focal_weight_neg,
            "-",
            color=IEEE_COLORS["blue"],
            lw=1.5,
            label=f"Focal (gamma={focal_gamma})",
        )
        ax.plot(
            p,
            asl_weight_neg,
            "-",
            color=IEEE_COLORS["red"],
            lw=1.5,
            label=f"ASL (gamma_neg={gamma_neg})",
        )
        ax.set_xlabel("Model Confidence (for negative class)")
        ax.set_ylabel("Loss Weight")
        ax.set_title("(a) False Positive Penalty")
        ax.legend(fontsize=7)
        ax.annotate(
            "ASL suppresses FP\nmuch harder",
            xy=(0.7, 0.15),
            fontsize=7,
            color=IEEE_COLORS["red"],
            fontweight="bold",
        )

        # (b) Loss weight for POSITIVE examples (true positives)
        ax = axes[1]
        focal_weight_pos = (1 - p) ** focal_gamma
        asl_weight_pos = (1 - p) ** gamma_pos  # gamma_pos=0 means weight=1 always
        ax.plot(
            p,
            focal_weight_pos,
            "-",
            color=IEEE_COLORS["blue"],
            lw=1.5,
            label=f"Focal (gamma={focal_gamma})",
        )
        ax.plot(
            p,
            asl_weight_pos,
            "-",
            color=IEEE_COLORS["green"],
            lw=1.5,
            label=f"ASL (gamma_pos={gamma_pos})",
        )
        ax.set_xlabel("Model Confidence (for positive class)")
        ax.set_ylabel("Loss Weight")
        ax.set_title("(b) True Positive Weighting")
        ax.legend(fontsize=7)
        ax.annotate(
            "ASL keeps weight=1\nfor ALL positives",
            xy=(0.5, 0.8),
            fontsize=7,
            color=IEEE_COLORS["green"],
            fontweight="bold",
        )

        fig.suptitle("Asymmetric Loss vs Focal Loss", fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_asl_vs_focal")


# ---------------------------------------------------------------------------
# 4. Class Filtering Impact
# ---------------------------------------------------------------------------


def plot_class_filtering_impact(
    all_classes: list[str],
    class_counts: dict[str, int],
    min_samples: int,
    save_dir: Path,
):
    """Show which classes are kept vs dropped and the long-tail distribution.

    Parameters
    ----------
    all_classes : list[str]
        All 45 disease names.
    class_counts : dict[str, int]
        Disease name -> positive sample count in training set.
    min_samples : int
        Minimum samples threshold.
    save_dir : Path
        Output directory.
    """
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.7)

        # Sort by count descending
        sorted_classes = sorted(all_classes, key=lambda c: class_counts.get(c, 0), reverse=True)
        counts = [class_counts.get(c, 0) for c in sorted_classes]
        colors = [
            IEEE_COLORS["green"] if class_counts.get(c, 0) >= min_samples else IEEE_COLORS["red"]
            for c in sorted_classes
        ]

        # (a) Bar chart with threshold line
        ax = axes[0]
        ax.barh(range(len(sorted_classes)), counts, color=colors, edgecolor="black", lw=0.2)
        ax.axvline(min_samples, ls="--", color="red", lw=1.2, label=f"Threshold = {min_samples}")
        ax.set_yticks(range(len(sorted_classes)))
        ax.set_yticklabels(sorted_classes, fontsize=5)
        ax.invert_yaxis()
        ax.set_xlabel("Positive Training Samples")
        ax.set_title(f"(a) Class Distribution (n={len(all_classes)})")
        ax.legend(fontsize=7)

        # (b) Summary pie chart
        ax = axes[1]
        n_kept = sum(1 for c in counts if c >= min_samples)
        n_dropped = len(all_classes) - n_kept
        ax.pie(
            [n_kept, n_dropped],
            labels=[f"Retained ({n_kept})", f"Dropped ({n_dropped})"],
            colors=[IEEE_COLORS["green"], IEEE_COLORS["red"]],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 8},
        )
        ax.set_title(f"(b) Class Filtering (min={min_samples})")

        fig.suptitle("Ultra-Rare Class Filtering", fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_class_filtering")


# ---------------------------------------------------------------------------
# 5. Staged Unfreezing Learning Dynamics
# ---------------------------------------------------------------------------


def plot_staged_unfreezing(
    history: list[dict],
    unfreeze_epoch: int,
    save_dir: Path,
):
    """Show training dynamics with staged backbone unfreezing.

    Highlights the unfreeze transition point and separate backbone/head LR curves.

    Parameters
    ----------
    history : list[dict]
        Training history with keys: epoch, train_loss, val_loss, val/precision_macro,
        val/f1_macro, lr (and optionally backbone_lr).
    unfreeze_epoch : int
        Epoch at which backbone was unfrozen.
    save_dir : Path
        Output directory.
    """
    with ieee_style():
        fig, axes = ieee_figure(2, 2, width="double", height_ratio=0.5)
        epochs = [h["epoch"] for h in history]

        # (a) Loss with unfreeze marker
        ax = axes[0, 0]
        ax.plot(
            epochs,
            [h["train_loss"] for h in history],
            "o-",
            ms=2,
            color=IEEE_COLORS["blue"],
            label="Train",
        )
        ax.plot(
            epochs,
            [h["val_loss"] for h in history],
            "s-",
            ms=2,
            color=IEEE_COLORS["red"],
            label="Val",
        )
        ax.axvline(unfreeze_epoch, ls="--", color="green", lw=1.2, label="Backbone unfreeze")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("(a) Loss (ASL)")
        ax.legend(fontsize=6)

        # (b) Precision tracking
        ax = axes[0, 1]
        if any("val/precision_macro" in h for h in history):
            ax.plot(
                epochs,
                [h.get("val/precision_macro", 0) for h in history],
                "o-",
                ms=2,
                color=IEEE_COLORS["green"],
                label="Val Precision",
            )
        if any("val/recall_macro" in h for h in history):
            ax.plot(
                epochs,
                [h.get("val/recall_macro", 0) for h in history],
                "s-",
                ms=2,
                color=IEEE_COLORS["purple"],
                label="Val Recall",
            )
        ax.axvline(unfreeze_epoch, ls="--", color="green", lw=1.2)
        ax.axhline(0.10, ls=":", color="red", lw=0.8, alpha=0.5, label="Precision floor")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("(b) Precision & Recall")
        ax.legend(fontsize=6)

        # (c) F1 + AUC + Accuracy
        ax = axes[1, 0]
        if any("val/f1_macro" in h for h in history):
            ax.plot(
                epochs,
                [h.get("val/f1_macro", 0) for h in history],
                "o-",
                ms=2,
                color=METRIC_COLORS["f1_macro"],
                label="F1 Macro",
            )
        if any("val/auc_roc" in h for h in history):
            ax.plot(
                epochs,
                [h.get("val/auc_roc", 0) for h in history],
                "s-",
                ms=2,
                color=METRIC_COLORS["auc_roc"],
                label="AUC-ROC",
            )
        if any("val/accuracy_macro" in h for h in history):
            ax.plot(
                epochs,
                [h.get("val/accuracy_macro", 0) for h in history],
                "^-",
                ms=2,
                color=METRIC_COLORS["accuracy_macro"],
                label="Accuracy",
            )
        ax.axvline(unfreeze_epoch, ls="--", color="green", lw=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("(c) F1 & AUC-ROC")
        ax.legend(fontsize=6)

        # (d) Learning rate schedule (dual axis)
        ax = axes[1, 1]
        if any("lr" in h for h in history):
            ax.plot(
                epochs,
                [h.get("lr", 0) for h in history],
                "-",
                color=IEEE_COLORS["blue"],
                lw=1.5,
                label="Head LR",
            )
        if any("backbone_lr" in h for h in history):
            ax2 = ax.twinx()
            ax2.plot(
                epochs,
                [h.get("backbone_lr", 0) for h in history],
                "-",
                color=IEEE_COLORS["orange"],
                lw=1.5,
                label="Backbone LR",
            )
            ax2.set_ylabel("Backbone LR", color=IEEE_COLORS["orange"], fontsize=7)
            ax2.tick_params(axis="y", labelcolor=IEEE_COLORS["orange"])
        ax.axvline(unfreeze_epoch, ls="--", color="green", lw=1.2, label="Unfreeze")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Head LR")
        ax.set_title("(d) Learning Rate Schedule")
        ax.legend(fontsize=6, loc="upper right")

        fig.suptitle("Staged Backbone Unfreezing Dynamics", fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_staged_unfreezing")


# ---------------------------------------------------------------------------
# 6. Precision-Recall Trade-off Surface
# ---------------------------------------------------------------------------


def plot_precision_recall_tradeoff(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_dir: Path,
    model_name: str = "HybridV2",
):
    """2D surface showing precision/recall at different global thresholds,
    with the precision floor marked.
    """
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.55)
        from sklearn.metrics import f1_score, precision_score, recall_score

        thresholds = np.arange(0.05, 0.96, 0.02)
        precisions, recalls, f1s = [], [], []

        for t in thresholds:
            preds = (y_prob >= t).astype(float)
            precisions.append(precision_score(y_true, preds, average="macro", zero_division=0))
            recalls.append(recall_score(y_true, preds, average="macro", zero_division=0))
            f1s.append(f1_score(y_true, preds, average="macro", zero_division=0))

        # (a) P/R/F1 vs threshold
        ax = axes[0]
        ax.plot(thresholds, precisions, "o-", ms=2, color=IEEE_COLORS["green"], label="Precision")
        ax.plot(thresholds, recalls, "s-", ms=2, color=IEEE_COLORS["purple"], label="Recall")
        ax.plot(thresholds, f1s, "^-", ms=2, color=IEEE_COLORS["blue"], label="F1")
        ax.axhline(0.10, ls=":", color="red", lw=1.0, alpha=0.7, label="Precision floor (0.10)")
        ax.set_xlabel("Global Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"(a) {model_name}: Precision-Recall Trade-off")
        ax.legend(fontsize=6)
        ax.set_ylim(0, 1)

        # (b) Precision vs Recall parametric curve
        ax = axes[1]
        ax.plot(recalls, precisions, "o-", ms=2, color=IEEE_COLORS["blue"])
        ax.axhline(0.10, ls=":", color="red", lw=1.0, alpha=0.7, label="Precision floor")
        ax.set_xlabel("Recall (macro)")
        ax.set_ylabel("Precision (macro)")
        ax.set_title("(b) Precision-Recall Curve (macro)")
        ax.legend(fontsize=7)

        # Annotate optimal F1 point
        best_idx = np.argmax(f1s)
        ax.annotate(
            f"Best F1={f1s[best_idx]:.3f}\nP={precisions[best_idx]:.3f}\nR={recalls[best_idx]:.3f}",
            xy=(recalls[best_idx], precisions[best_idx]),
            xytext=(20, 20),
            textcoords="offset points",
            fontsize=6,
            arrowprops=dict(arrowstyle="->", color="red"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange"),
        )

        fig.suptitle(
            "Precision-Recall Operating Point Analysis", fontsize=11, fontweight="bold", y=1.02
        )
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_precision_recall_tradeoff")


# ---------------------------------------------------------------------------
# 7. TTA Improvement Analysis
# ---------------------------------------------------------------------------


def plot_tta_improvement(
    metrics_no_tta: dict,
    metrics_with_tta: dict,
    save_dir: Path,
):
    """Compare metrics with and without Test-Time Augmentation.

    Parameters
    ----------
    metrics_no_tta : dict
        Metrics without TTA.
    metrics_with_tta : dict
        Metrics with TTA (6 augmented views).
    """
    with ieee_style():
        fig, ax = ieee_figure(1, 1, width="single", height_ratio=0.8)

        metrics_list = ["precision_macro", "recall_macro", "f1_macro", "auc_roc", "accuracy_macro"]
        labels = ["Precision", "Recall", "F1", "AUC", "Accuracy"]
        x = np.arange(len(labels))

        no_tta = [metrics_no_tta.get(m, 0) for m in metrics_list]
        with_tta = [metrics_with_tta.get(m, 0) for m in metrics_list]

        w = 0.35
        ax.bar(
            x - w / 2,
            no_tta,
            w,
            label="No TTA",
            color=IEEE_COLORS["gray"],
            edgecolor="black",
            lw=0.3,
        )
        ax.bar(
            x + w / 2,
            with_tta,
            w,
            label="TTA (6 views)",
            color=IEEE_COLORS["green"],
            edgecolor="black",
            lw=0.3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_title("Test-Time Augmentation Impact")
        ax.legend()
        ax.set_ylim(0, max(max(no_tta), max(with_tta)) * 1.3 + 0.05)

        # Delta annotations
        for i, (v1, v2) in enumerate(zip(no_tta, with_tta)):
            delta = v2 - v1
            sign = "+" if delta >= 0 else ""
            ax.text(
                i + w / 2,
                v2 + 0.01,
                f"{sign}{delta:.3f}",
                ha="center",
                fontsize=6,
                color=IEEE_COLORS["green"] if delta > 0 else IEEE_COLORS["red"],
                fontweight="bold",
            )

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_tta_improvement")


# ---------------------------------------------------------------------------
# 8. HybridV2 Architecture Diagram
# ---------------------------------------------------------------------------


def plot_hybrid_v2_architecture(save_dir: Path):
    """Architecture diagram for RetinalFoundationHybridV2."""
    with ieee_style():
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_axis_off()

        colors = {
            "input": "#E8F4F8",
            "backbone": "#B8E0F6",
            "graph": "#D4F1D4",
            "classifier": "#FFE5B4",
            "output": "#FADADD",
            "gate": "#F5F5DC",
        }

        def box(x, y, w, h, text, color, fontsize=7):
            rect = plt.Rectangle(
                (x, y), w, h, facecolor=color, edgecolor="black", lw=1.0, zorder=2, clip_on=False
            )
            ax.add_patch(rect)
            ax.text(
                x + w / 2,
                y + h / 2,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                zorder=3,
            )

        def arrow(x1, y1, x2, y2):
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"),
            )

        # Fundus Gate
        box(0.5, 8.5, 2.5, 0.8, "Fundus Gate\n(MobileNetV3)", colors["gate"], 6)
        arrow(1.75, 8.5, 1.75, 7.8)

        # Input
        box(0.5, 7.0, 2.5, 0.7, "Input Image\n3 x 224 x 224", colors["input"], 6)
        arrow(1.75, 7.0, 1.75, 6.3)

        # RETFound backbone
        box(
            0.2,
            5.0,
            3.2,
            1.2,
            "RETFound ViT-Large\n304M params (frozen)\n+ LoRA rank=16",
            colors["backbone"],
            6,
        )
        arrow(1.75, 5.0, 1.75, 4.3)

        # Graph reasoning
        box(0.5, 3.5, 2.5, 0.7, "SparseTopK\nGraph Attention", colors["graph"], 6)
        arrow(1.75, 3.5, 1.75, 2.8)

        # Bottleneck classifier
        box(
            0.2,
            1.5,
            3.2,
            1.2,
            "Bottleneck Classifier\n512 (drop 0.5)\n128 (drop 0.3)\nN classes",
            colors["classifier"],
            6,
        )
        arrow(1.75, 1.5, 1.75, 0.8)

        # Output
        box(0.5, 0.1, 2.5, 0.6, "Predictions +\nOptimized Thresholds", colors["output"], 6)

        # Right side: key features
        features = [
            ("ASL Loss", "gamma_pos=0, gamma_neg=4", 8.5),
            ("Class Filtering", "Drop <10 samples", 7.5),
            ("Staged Unfreeze", "Epochs 0-10: head only", 6.5),
            ("Precision Floor", "Per-class threshold >= 0.10", 5.5),
            ("TTA Inference", "6 augmented views", 4.5),
            ("Label Smoothing", "alpha = 0.05", 3.5),
        ]
        for text, detail, y in features:
            box(5.0, y, 4.5, 0.7, f"{text}\n{detail}", "#F0F0F0", 6)

        ax.set_title(
            "RetinalFoundationHybridV2 — Precision Rescue Architecture",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_hybrid_v2_architecture")


# ---------------------------------------------------------------------------
# Master Generator
# ---------------------------------------------------------------------------


def generate_all_precision_rescue_plots(
    save_dir: str | Path = "outputs/plots/precision_rescue",
    before_metrics: dict | None = None,
    after_metrics: dict | None = None,
    threshold_report: dict | None = None,
    history: list | None = None,
    y_true: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
    class_counts: dict | None = None,
    all_classes: list | None = None,
    metrics_no_tta: dict | None = None,
    metrics_with_tta: dict | None = None,
    unfreeze_epoch: int = 10,
    min_samples: int = 10,
):
    """Generate all precision-rescue visualizations.

    Call with whatever data is available; skips plots for missing data.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Always generate these (no data required)
    plot_asl_vs_focal_loss(save_dir)
    generated.append("ASL vs Focal Loss")

    plot_hybrid_v2_architecture(save_dir)
    generated.append("HybridV2 Architecture")

    if before_metrics and after_metrics:
        plot_before_after_comparison(before_metrics, after_metrics, save_dir)
        generated.append("Before/After Comparison")

    if threshold_report:
        plot_threshold_heatmap(threshold_report, save_dir)
        generated.append("Threshold Heatmap")

    if all_classes and class_counts:
        plot_class_filtering_impact(all_classes, class_counts, min_samples, save_dir)
        generated.append("Class Filtering")

    if history:
        plot_staged_unfreezing(history, unfreeze_epoch, save_dir)
        generated.append("Staged Unfreezing")

    if y_true is not None and y_prob is not None:
        plot_precision_recall_tradeoff(y_true, y_prob, save_dir)
        generated.append("Precision-Recall Trade-off")

    if metrics_no_tta and metrics_with_tta:
        plot_tta_improvement(metrics_no_tta, metrics_with_tta, save_dir)
        generated.append("TTA Improvement")

    print(f"Generated {len(generated)} precision rescue plots -> {save_dir}")
    for name in generated:
        print(f"  - {name}")

    return generated
