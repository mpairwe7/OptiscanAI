"""
EDA Visualizations - IEEE Publication Quality.
Preserves and enhances notebook cells 9-17 plots.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.visualization.ieee_style import (
    ieee_style, ieee_figure, save_ieee, annotate_bars,
    IEEE_COLORS, CB_PALETTE, add_watermark,
)


def plot_disease_distribution(
    disease_counts: pd.Series,
    total_samples: int,
    save_dir: Path,
):
    """Top-20 disease frequency + log-scale count (notebook cell 25 enhanced)."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.45)

        freq = (disease_counts / total_samples * 100).sort_values(ascending=False)

        # (a) Frequency %
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, min(20, len(freq))))
        axes[0].barh(
            range(min(20, len(freq))),
            freq.values[: 20],
            color=colors,
            edgecolor="black",
            linewidth=0.4,
        )
        axes[0].set_yticks(range(min(20, len(freq))))
        axes[0].set_yticklabels(freq.index[: 20], fontsize=7)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Prevalence (%)")
        axes[0].set_title("(a) Top-20 Disease Prevalence")
        axes[0].axvline(x=1.0, color="red", ls="--", lw=1, alpha=0.6, label="1 % threshold")
        axes[0].legend(fontsize=7)

        # (b) Log-scale counts
        axes[1].bar(
            range(len(disease_counts)),
            disease_counts.sort_values(ascending=False).values,
            color="steelblue",
            edgecolor="black",
            linewidth=0.3,
        )
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Disease Rank")
        axes[1].set_ylabel("Sample Count (log)")
        axes[1].set_title("(b) Class Imbalance (Log Scale)")

        rare = (freq < 1.0).sum()
        axes[1].text(
            0.95, 0.95,
            f"Rare (<1%): {rare}/{len(freq)}",
            transform=axes[1].transAxes, ha="right", va="top",
            fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
        )

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_disease_distribution")


def plot_multilabel_statistics(
    labels_per_sample: pd.Series,
    save_dir: Path,
):
    """Label distribution analysis: histogram, box, violin, CDF (notebook cell 9)."""
    with ieee_style():
        fig, axes = ieee_figure(2, 2, width="double", height_ratio=0.45)

        # (a) Histogram + KDE
        axes[0, 0].hist(labels_per_sample, bins=range(int(labels_per_sample.max()) + 2),
                        color=IEEE_COLORS["blue"], alpha=0.7, edgecolor="black", lw=0.4, density=True)
        if len(labels_per_sample) > 1:
            try:
                kde_x = np.linspace(0, labels_per_sample.max(), 100)
                kde = stats.gaussian_kde(labels_per_sample)
                axes[0, 0].plot(kde_x, kde(kde_x), color=IEEE_COLORS["red"], lw=1.5, label="KDE")
                axes[0, 0].legend()
            except Exception:
                pass
        axes[0, 0].set_xlabel("Labels per Sample")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("(a) Label Distribution")

        # (b) Box plot
        bp = axes[0, 1].boxplot(labels_per_sample, vert=True, patch_artist=True,
                                boxprops=dict(facecolor=IEEE_COLORS["teal"], alpha=0.6))
        axes[0, 1].set_ylabel("Diseases per Sample")
        axes[0, 1].set_title("(b) Box Plot")

        # (c) Violin
        vp = axes[1, 0].violinplot(labels_per_sample, showmeans=True, showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(IEEE_COLORS["purple"])
            body.set_alpha(0.6)
        axes[1, 0].set_ylabel("Diseases per Sample")
        axes[1, 0].set_title("(c) Violin Plot")

        # (d) CDF
        sorted_vals = np.sort(labels_per_sample)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        axes[1, 1].plot(sorted_vals, cdf, color=IEEE_COLORS["orange"], lw=1.5)
        axes[1, 1].axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.5)
        axes[1, 1].set_xlabel("Number of Diseases")
        axes[1, 1].set_ylabel("Cumulative Probability")
        axes[1, 1].set_title("(d) CDF")

        # Stats annotation
        stats_text = (
            f"Mean: {labels_per_sample.mean():.2f}\n"
            f"Median: {labels_per_sample.median():.1f}\n"
            f"Max: {labels_per_sample.max():.0f}\n"
            f"Skew: {labels_per_sample.skew():.2f}"
        )
        axes[1, 1].text(
            0.95, 0.3, stats_text,
            transform=axes[1, 1].transAxes, ha="right", va="bottom",
            fontsize=7, bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8),
        )

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_multilabel_statistics")


def plot_cooccurrence_matrix(
    labels_df: pd.DataFrame,
    disease_columns: list[str],
    save_dir: Path,
    top_k: int = 20,
):
    """Disease co-occurrence heatmap (notebook cell 12 enhanced)."""
    with ieee_style():
        # Select top-k diseases
        counts = labels_df[disease_columns].sum().sort_values(ascending=False)
        top_diseases = counts.index[:top_k].tolist()
        sub = labels_df[top_diseases].apply(pd.to_numeric, errors="coerce").fillna(0)

        # Co-occurrence matrix
        co = sub.T.dot(sub)
        # Normalize by geometric mean of individual counts
        diag = np.diag(co.values).clip(1)
        norm = co.values / np.sqrt(np.outer(diag, diag))
        np.fill_diagonal(norm, 1.0)

        fig, ax = ieee_figure(1, 1, width="double", height_ratio=0.85)
        mask = np.triu(np.ones_like(norm, dtype=bool), k=1)
        sns.heatmap(
            norm, mask=mask, ax=ax,
            xticklabels=top_diseases, yticklabels=top_diseases,
            cmap="YlOrRd", vmin=0, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 6},
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Normalized Co-occurrence", "shrink": 0.8},
        )
        ax.set_title(f"Disease Co-occurrence Matrix (Top {top_k})")
        ax.tick_params(labelsize=7, labelrotation=45)

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_cooccurrence_matrix")


def plot_class_imbalance_analysis(
    disease_counts: pd.Series,
    total_samples: int,
    save_dir: Path,
):
    """Class imbalance analysis with prevalence categories (notebook cell 13)."""
    with ieee_style():
        fig, axes = ieee_figure(1, 3, width="double", height_ratio=0.5)
        freq = disease_counts / total_samples * 100

        # (a) Imbalance ratio visualization
        sorted_counts = disease_counts.sort_values(ascending=False)
        axes[0].fill_between(range(len(sorted_counts)), sorted_counts.values, alpha=0.3, color=IEEE_COLORS["blue"])
        axes[0].plot(range(len(sorted_counts)), sorted_counts.values, "o-", ms=3, lw=1, color=IEEE_COLORS["blue"])
        axes[0].set_xlabel("Disease Rank")
        axes[0].set_ylabel("Sample Count")
        axes[0].set_title("(a) Long-tail Distribution")

        # (b) Prevalence categories
        categories = {"Common (>5%)": 0, "Moderate (1-5%)": 0, "Rare (<1%)": 0, "Very Rare (<0.5%)": 0}
        for p in freq:
            if p > 5:
                categories["Common (>5%)"] += 1
            elif p > 1:
                categories["Moderate (1-5%)"] += 1
            elif p > 0.5:
                categories["Rare (<1%)"] += 1
            else:
                categories["Very Rare (<0.5%)"] += 1

        cat_colors = [IEEE_COLORS["green"], IEEE_COLORS["blue"], IEEE_COLORS["orange"], IEEE_COLORS["red"]]
        wedges, texts, autotexts = axes[1].pie(
            categories.values(), labels=categories.keys(),
            autopct="%1.0f%%", colors=cat_colors, startangle=90,
            textprops={"fontsize": 7},
        )
        axes[1].set_title("(b) Prevalence Categories")

        # (c) Imbalance ratio
        max_count = disease_counts.max()
        min_count = disease_counts[disease_counts > 0].min()
        ratio = max_count / min_count

        text = (
            f"Imbalance Ratio: {ratio:.1f}:1\n\n"
            f"Most common:\n  {disease_counts.idxmax()} ({max_count})\n\n"
            f"Least common:\n  {disease_counts[disease_counts > 0].idxmin()} ({min_count})\n\n"
            f"Strategy:\n  Focal Loss + Pos Weighting"
        )
        axes[2].text(0.1, 0.5, text, transform=axes[2].transAxes, fontsize=8,
                     va="center", family="monospace",
                     bbox=dict(boxstyle="round", fc="#f0f0f0", ec="#cccccc"))
        axes[2].set_axis_off()
        axes[2].set_title("(c) Imbalance Summary")

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_class_imbalance_analysis")


def plot_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    disease_columns: list[str],
    save_dir: Path,
):
    """Data split comparison (notebook cell 35 enhanced)."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.5)

        # (a) Split sizes
        sizes = [len(train_df), len(val_df), len(test_df)]
        labels = [f"Train\n{sizes[0]}", f"Val\n{sizes[1]}", f"Test\n{sizes[2]}"]
        colors = [IEEE_COLORS["blue"], IEEE_COLORS["green"], IEEE_COLORS["orange"]]
        axes[0].bar(labels, sizes, color=colors, edgecolor="black", lw=0.5)
        total = sum(sizes)
        for i, (s, l) in enumerate(zip(sizes, labels)):
            axes[0].text(i, s + total * 0.01, f"{s / total * 100:.1f}%",
                         ha="center", fontsize=8, fontweight="bold")
        axes[0].set_ylabel("Number of Samples")
        axes[0].set_title("(a) Dataset Split Distribution")

        # (b) Disease prevalence per split
        top_10 = train_df[disease_columns].sum().sort_values(ascending=False).index[:10]
        x = np.arange(len(top_10))
        w = 0.25
        for offset, (df, name, color) in enumerate([
            (train_df, "Train", colors[0]),
            (val_df, "Val", colors[1]),
            (test_df, "Test", colors[2]),
        ]):
            vals = df[top_10].sum().values / len(df) * 100
            axes[1].bar(x + offset * w, vals, w, label=name, color=color, edgecolor="black", lw=0.3)
        axes[1].set_xticks(x + w)
        axes[1].set_xticklabels(top_10, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("Prevalence (%)")
        axes[1].set_title("(b) Top-10 Disease Prevalence by Split")
        axes[1].legend(fontsize=7)

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_split_distribution")


def generate_all_eda_plots(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    disease_columns: list[str],
    save_dir: str | Path = "outputs/plots/eda",
):
    """Generate all EDA plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_labels = train_df[disease_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    disease_counts = train_labels.sum().sort_values(ascending=False)
    labels_per_sample = train_labels.sum(axis=1)
    total = len(train_df)

    print(f"Generating EDA plots -> {save_dir}")

    plot_disease_distribution(disease_counts, total, save_dir)
    print("  [1/5] Disease distribution")

    plot_multilabel_statistics(labels_per_sample, save_dir)
    print("  [2/5] Multi-label statistics")

    plot_cooccurrence_matrix(train_df, disease_columns, save_dir)
    print("  [3/5] Co-occurrence matrix")

    plot_class_imbalance_analysis(disease_counts, total, save_dir)
    print("  [4/5] Class imbalance analysis")

    plot_split_distribution(train_df, val_df, test_df, disease_columns, save_dir)
    print("  [5/5] Split distribution")

    print(f"  EDA plots saved to {save_dir}")
