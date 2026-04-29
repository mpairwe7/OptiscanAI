"""
IEEE Access 2026 Publication-Quality Visualization Framework.

Standards enforced:
  - Column width: 3.5 in (single) / 7.0 in (double)
  - Resolution: 300 DPI minimum, 600 DPI for camera-ready
  - Fonts: 8-10 pt labels, 10-12 pt titles, serif or sans-serif
  - Line width: 0.75-1.5 pt
  - Outputs: PDF (vector) + PNG (bitmap)
  - Color: accessible palette (colorblind-safe)
"""

from __future__ import annotations
import contextlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
IEEE_COLORS = {
    "blue": "#2171B5",
    "orange": "#E6550D",
    "green": "#31A354",
    "red": "#CB181D",
    "purple": "#756BB1",
    "teal": "#00897B",
    "gold": "#FDD835",
    "gray": "#636363",
    "pink": "#E377C2",
    "brown": "#8C564B",
}

# Colorblind-safe sequential
CB_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#999999",
]

MODEL_COLORS = {
    "ViGNN": "#2171B5",
    "GraphCLIP": "#E6550D",
    "VisualLanguageGNN": "#31A354",
    "SceneGraphTransformer": "#756BB1",
    "ViT": "#00897B",
    "EfficientNet": "#CB181D",
    "ResNet50-GCN": "#FDD835",
    "HybridV1": "#1B9E77",
    "HybridV2": "#D95F02",
    "HybridV2-TTA": "#7570B3",
    "RETFound-MLP": "#E7298A",
}

METRIC_COLORS = {
    "f1_macro": "#2171B5",
    "f1_micro": "#6BAED6",
    "auc_roc": "#E6550D",
    "precision_macro": "#31A354",
    "recall_macro": "#756BB1",
    "mAP": "#00897B",
    "hamming_loss": "#CB181D",
    "accuracy_macro": "#E7298A",
    "accuracy_micro": "#D95F02",
    "accuracy_jaccard": "#7570B3",
    "accuracy_subset": "#66A61E",
}

# ---------------------------------------------------------------------------
# Global IEEE style context
# ---------------------------------------------------------------------------
_IEEE_RC = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,

    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlepad": 6,
    "axes.labelsize": 9,
    "axes.labelweight": "bold",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,

    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,

    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",

    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",

    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    "figure.constrained_layout.use": True,
}


@contextlib.contextmanager
def ieee_style():
    """Context manager that applies IEEE publication styling."""
    with plt.rc_context(_IEEE_RC):
        yield


def ieee_figure(
    nrows: int = 1,
    ncols: int = 1,
    width: str = "single",
    height_ratio: float = 0.7,
    **kwargs,
):
    """
    Create an IEEE-sized figure.

    Args:
        width: 'single' (3.5 in) or 'double' (7.0 in)
        height_ratio: height = width * ratio
    """
    w = 3.5 if width == "single" else 7.0
    h = w * height_ratio / max(ncols, 1) * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=100, **kwargs)
    return fig, axes


def save_ieee(fig, path: str | Path, dpi: int = 300, formats=("pdf", "png")):
    """Save figure in multiple IEEE-quality formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)


def annotate_bars(ax, fmt="{:.3f}", fontsize=7, offset=0.01):
    """Add value labels on top of bar chart bars."""
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + offset,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
            )


def add_watermark(fig, text="MLOps Pipeline 2026"):
    """Add a subtle watermark."""
    fig.text(
        0.99, 0.01, text,
        ha="right", va="bottom",
        fontsize=6, color="#CCCCCC", style="italic",
        transform=fig.transFigure,
    )
