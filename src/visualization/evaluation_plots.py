"""
Evaluation Visualizations - IEEE Publication Quality.
Confusion matrices, ROC/PR curves, per-class analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from src.visualization.ieee_style import (
    IEEE_COLORS,
    METRIC_COLORS,
    add_watermark,
    ieee_figure,
    ieee_style,
    save_ieee,
)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_names: list[str],
    save_dir: Path,
    top_k: int = 10,
):
    """Per-class ROC curves for top-K diseases."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.65)

        # Select top-K classes by AUC
        aucs = {}
        for i, name in enumerate(disease_names):
            if y_true[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                aucs[i] = auc(fpr, tpr)
        top_indices = sorted(aucs, key=aucs.get, reverse=True)[:top_k]

        # (a) Individual ROC curves
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_indices)))
        for idx, (i, c) in enumerate(zip(top_indices, colors)):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            axes[0].plot(fpr, tpr, color=c, lw=1.2,
                        label=f"{disease_names[i]} ({aucs[i]:.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(f"(a) ROC Curves (Top-{top_k} by AUC)")
        axes[0].legend(fontsize=6, loc="lower right", ncol=1)

        # (b) AUC bar chart
        names = [disease_names[i] for i in top_indices]
        vals = [aucs[i] for i in top_indices]
        bars = axes[1].barh(range(len(names)), vals,
                           color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))),
                           edgecolor="black", lw=0.3)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("AUC-ROC")
        axes[1].set_title(f"(b) AUC-ROC Ranking (Top-{top_k})")
        axes[1].axvline(0.5, ls="--", color="red", lw=0.8, alpha=0.5, label="Random")
        axes[1].legend(fontsize=7)

        # Annotate bars
        for bar, val in zip(bars, vals):
            axes[1].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=6)

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_roc_curves")


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_names: list[str],
    save_dir: Path,
    top_k: int = 10,
):
    """Per-class Precision-Recall curves."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.65)

        aps = {}
        for i, name in enumerate(disease_names):
            if y_true[:, i].sum() > 0:
                aps[i] = average_precision_score(y_true[:, i], y_prob[:, i])
        top_indices = sorted(aps, key=aps.get, reverse=True)[:top_k]

        colors = plt.cm.tab10(np.linspace(0, 1, len(top_indices)))
        for i, c in zip(top_indices, colors):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
            axes[0].plot(recall, precision, color=c, lw=1.2,
                        label=f"{disease_names[i]} ({aps[i]:.3f})")
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].set_title(f"(a) PR Curves (Top-{top_k} by AP)")
        axes[0].legend(fontsize=6, loc="upper right")

        # (b) AP bar chart
        names = [disease_names[i] for i in top_indices]
        vals = [aps[i] for i in top_indices]
        axes[1].barh(range(len(names)), vals,
                    color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))),
                    edgecolor="black", lw=0.3)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Average Precision")
        axes[1].set_title(f"(b) AP Ranking (Top-{top_k})")

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_precision_recall_curves")


def plot_confusion_matrix_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    disease_names: list[str],
    save_dir: Path,
    top_k: int = 15,
    model_name: str = "ViGNN",
):
    """Multi-label confusion matrix (notebook cell 36 enhanced)."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.85)

        # Select top-K diseases by positive count
        pos_counts = y_true.sum(axis=0)
        top_idx = np.argsort(pos_counts)[::-1][:top_k]

        # (a) Per-class TP/FP/FN/TN summary
        tp = ((y_pred == 1) & (y_true == 1)).sum(axis=0)[top_idx]
        fp = ((y_pred == 1) & (y_true == 0)).sum(axis=0)[top_idx]
        fn = ((y_pred == 0) & (y_true == 1)).sum(axis=0)[top_idx]
        tn = ((y_pred == 0) & (y_true == 0)).sum(axis=0)[top_idx]

        names = [disease_names[i] for i in top_idx]
        x = np.arange(len(names))
        w = 0.2
        axes[0].bar(x - 1.5*w, tp, w, label="TP", color=IEEE_COLORS["green"], edgecolor="black", lw=0.3)
        axes[0].bar(x - 0.5*w, fp, w, label="FP", color=IEEE_COLORS["orange"], edgecolor="black", lw=0.3)
        axes[0].bar(x + 0.5*w, fn, w, label="FN", color=IEEE_COLORS["red"], edgecolor="black", lw=0.3)
        axes[0].bar(x + 1.5*w, tn, w, label="TN", color=IEEE_COLORS["blue"], edgecolor="black", lw=0.3, alpha=0.3)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"(a) Per-Class TP/FP/FN/TN (Top-{top_k})")
        axes[0].legend(fontsize=7, ncol=4)
        axes[0].set_yscale("symlog", linthresh=1)

        # (b) Per-class F1/Precision/Recall heatmap
        metrics_data = []
        for i in top_idx:
            p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            f = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            metrics_data.append([p, r, f])

        metrics_arr = np.array(metrics_data)
        sns.heatmap(
            metrics_arr, ax=axes[1],
            xticklabels=["Precision", "Recall", "F1"],
            yticklabels=names,
            annot=True, fmt=".3f", annot_kws={"size": 6},
            cmap="YlGn", vmin=0, vmax=1,
            linewidths=0.3, linecolor="white",
        )
        axes[1].set_title("(b) Per-Class Metrics Heatmap")
        axes[1].tick_params(labelsize=6)

        fig.suptitle(f"{model_name} - Multi-Label Confusion Analysis", fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_confusion_matrix")


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_dir: Path,
    model_name: str = "ViGNN",
):
    """Optimal threshold selection analysis."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.5)

        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_macros, f1_micros, hamming = [], [], []
        from sklearn.metrics import hamming_loss as hl

        for t in thresholds:
            preds = (y_prob > t).astype(float)
            f1_macros.append(f1_score(y_true, preds, average="macro", zero_division=0))
            f1_micros.append(f1_score(y_true, preds, average="micro", zero_division=0))
            hamming.append(hl(y_true, preds))

        # (a) F1 vs threshold
        axes[0].plot(thresholds, f1_macros, "o-", ms=3, color=METRIC_COLORS["f1_macro"], label="F1 Macro")
        axes[0].plot(thresholds, f1_micros, "s-", ms=3, color=METRIC_COLORS["f1_micro"], label="F1 Micro")
        best_idx = np.argmax(f1_macros)
        axes[0].axvline(thresholds[best_idx], ls="--", color="red", lw=0.8)
        axes[0].annotate(
            f"Optimal: {thresholds[best_idx]:.2f}\nF1: {f1_macros[best_idx]:.4f}",
            xy=(thresholds[best_idx], f1_macros[best_idx]),
            xytext=(10, -20), textcoords="offset points", fontsize=7,
            arrowprops=dict(arrowstyle="->", color="red"),
        )
        axes[0].set_xlabel("Threshold")
        axes[0].set_ylabel("F1 Score")
        axes[0].set_title("(a) F1 Score vs Threshold")
        axes[0].legend()

        # (b) Hamming loss vs threshold
        axes[1].plot(thresholds, hamming, "o-", ms=3, color=METRIC_COLORS["hamming_loss"])
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Hamming Loss")
        axes[1].set_title("(b) Hamming Loss vs Threshold")

        fig.suptitle(f"{model_name} - Threshold Optimization", fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_threshold_analysis")

        return thresholds[best_idx]


def plot_metrics_summary_table(
    metrics: dict,
    save_dir: Path,
    model_name: str = "ViGNN",
):
    """Publication-ready metrics summary table figure."""
    with ieee_style():
        fig, ax = ieee_figure(1, 1, width="single", height_ratio=1.0)
        ax.set_axis_off()

        headers = ["Metric", "Value"]
        rows = [
            ["F1 Macro", f"{metrics.get('f1_macro', 0):.4f}"],
            ["F1 Micro", f"{metrics.get('f1_micro', 0):.4f}"],
            ["AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}"],
            ["mAP", f"{metrics.get('mAP', 0):.4f}"],
            ["Precision (Macro)", f"{metrics.get('precision_macro', 0):.4f}"],
            ["Recall (Macro)", f"{metrics.get('recall_macro', 0):.4f}"],
            ["Accuracy (Macro)", f"{metrics.get('accuracy_macro', 0):.4f}"],
            ["Accuracy (Micro)", f"{metrics.get('accuracy_micro', 0):.4f}"],
            ["Accuracy (Jaccard)", f"{metrics.get('accuracy_jaccard', 0):.4f}"],
            ["Accuracy (Subset)", f"{metrics.get('accuracy_subset', 0):.4f}"],
            ["Hamming Loss", f"{metrics.get('hamming_loss', 0):.4f}"],
            ["F1 Samples", f"{metrics.get('f1_samples', 0):.4f}"],
        ]

        table = ax.table(
            cellText=rows, colLabels=headers,
            cellLoc="center", loc="center",
            colColours=["#E8E8E8", "#E8E8E8"],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Highlight best metrics
        for i, row in enumerate(rows):
            val = float(row[1])
            if "Loss" not in row[0] and val > 0.5:
                table[i + 1, 1].set_facecolor("#d4edda")
            elif "Loss" in row[0] and val < 0.1:
                table[i + 1, 1].set_facecolor("#d4edda")

        ax.set_title(f"{model_name} - Evaluation Metrics", fontsize=10, fontweight="bold", pad=20)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_metrics_summary")


def plot_precision_floor_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_names: list[str],
    save_dir: Path,
    min_precision: float = 0.10,
    model_name: str = "HybridV2",
):
    """Per-class precision-floor threshold analysis (v2 precision rescue).

    Shows precision and recall at the optimized threshold for each class,
    with the precision floor marked.
    """
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.65)

        thresholds = np.arange(0.05, 0.96, 0.02)
        prec_macros, rec_macros, f1_macros = [], [], []

        for t in thresholds:
            preds = (y_prob >= t).astype(float)
            prec_macros.append(precision_score(y_true, preds, average="macro", zero_division=0))
            rec_macros.append(recall_score(y_true, preds, average="macro", zero_division=0))
            f1_macros.append(f1_score(y_true, preds, average="macro", zero_division=0))

        # (a) P/R/F1 vs global threshold with precision floor
        ax = axes[0]
        ax.plot(thresholds, prec_macros, "o-", ms=2, color=IEEE_COLORS["green"], label="Precision")
        ax.plot(thresholds, rec_macros, "s-", ms=2, color=IEEE_COLORS["purple"], label="Recall")
        ax.plot(thresholds, f1_macros, "^-", ms=2, color=IEEE_COLORS["blue"], label="F1")
        ax.axhline(min_precision, ls="--", color="red", lw=1.2, label=f"Precision floor ({min_precision})")

        # Find threshold where precision first meets floor
        for i, p in enumerate(prec_macros):
            if p >= min_precision:
                ax.axvline(thresholds[i], ls=":", color="green", lw=0.8, alpha=0.5)
                ax.annotate(
                    f"Floor met at t={thresholds[i]:.2f}",
                    xy=(thresholds[i], p), xytext=(15, -15),
                    textcoords="offset points", fontsize=6,
                    arrowprops=dict(arrowstyle="->", color="green"),
                )
                break

        ax.set_xlabel("Global Threshold")
        ax.set_ylabel("Score")
        ax.set_title("(a) Precision-Floor Trade-off")
        ax.legend(fontsize=6)
        ax.set_ylim(0, 1)

        # (b) Per-class precision at threshold=0.5 vs optimal
        ax = axes[1]
        n_classes = min(len(disease_names), y_prob.shape[1])
        per_class_prec_05 = []
        per_class_prec_opt = []
        valid_names = []

        for c in range(n_classes):
            if y_true[:, c].sum() == 0:
                continue
            valid_names.append(disease_names[c])
            pred_05 = (y_prob[:, c] >= 0.5).astype(float)
            per_class_prec_05.append(precision_score(y_true[:, c], pred_05, zero_division=0))
            # Find precision-floor optimal threshold for this class
            best_t_prec = 0
            for t in np.arange(0.95, 0.04, -0.02):
                pred_t = (y_prob[:, c] >= t).astype(float)
                p = precision_score(y_true[:, c], pred_t, zero_division=0)
                if p >= min_precision:
                    best_t_prec = p
            per_class_prec_opt.append(best_t_prec)

        if valid_names:
            x = np.arange(len(valid_names))
            w = 0.35
            ax.barh(x - w/2, per_class_prec_05, w, label="Fixed t=0.5",
                    color=IEEE_COLORS["gray"], edgecolor="black", lw=0.2)
            ax.barh(x + w/2, per_class_prec_opt, w, label="Precision-floor optimized",
                    color=IEEE_COLORS["green"], edgecolor="black", lw=0.2)
            ax.axvline(min_precision, ls="--", color="red", lw=1.0, label=f"Floor={min_precision}")
            ax.set_yticks(x)
            ax.set_yticklabels(valid_names, fontsize=5)
            ax.invert_yaxis()
            ax.set_xlabel("Precision")
            ax.set_title("(b) Per-Class Precision Improvement")
            ax.legend(fontsize=6, loc="lower right")

        fig.suptitle(f"{model_name} — Precision-Floor Threshold Analysis",
                     fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_precision_floor_analysis")


def generate_all_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_names: list[str],
    metrics: dict,
    save_dir: str | Path = "outputs/plots/evaluation",
    model_name: str = "ViGNN",
    threshold: float = 0.5,
    precision_floor: float | None = None,
):
    """Generate all evaluation plots.

    Parameters
    ----------
    precision_floor : float | None
        If set, also generates precision-floor threshold analysis plot (v2).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    threshold_arr = np.asarray(threshold)
    if np.isscalar(threshold_arr) or threshold_arr.ndim == 0:
        y_pred = (y_prob > float(threshold_arr)).astype(float)
    else:
        y_pred = (y_prob > threshold_arr.reshape(1, -1)).astype(float)

    print(f"Generating evaluation plots -> {save_dir}")

    plot_roc_curves(y_true, y_prob, disease_names, save_dir)
    print("  [1/6] ROC curves")

    plot_precision_recall_curves(y_true, y_prob, disease_names, save_dir)
    print("  [2/6] PR curves")

    plot_confusion_matrix_multilabel(y_true, y_pred, disease_names, save_dir, model_name=model_name)
    print("  [3/6] Confusion matrix")

    optimal_t = plot_threshold_analysis(y_true, y_prob, save_dir, model_name=model_name)
    print(f"  [4/6] Threshold analysis (optimal={optimal_t:.2f})")

    plot_metrics_summary_table(metrics, save_dir, model_name=model_name)
    print("  [5/6] Metrics summary")

    if precision_floor is not None:
        plot_precision_floor_threshold_analysis(
            y_true, y_prob, disease_names, save_dir,
            min_precision=precision_floor, model_name=model_name,
        )
        print(f"  [6/6] Precision-floor threshold analysis (floor={precision_floor})")

    print(f"  Evaluation plots saved to {save_dir}")
    return optimal_t
