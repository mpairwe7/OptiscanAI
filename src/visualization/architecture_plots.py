"""
Model Architecture Visualizations - IEEE Publication Quality.
Extracted from notebook cell 43 (ModelArchitectureExplainer).
Generates architecture diagrams for all 4 models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.visualization.ieee_style import add_watermark, ieee_style, save_ieee

COLORS = {
    "input": "#E8F4F8",
    "conv": "#B8E0F6",
    "attention": "#FFE5B4",
    "graph": "#D4F1D4",
    "output": "#F5B7B1",
    "fusion": "#D5B8F5",
    "text": "#FDEBD0",
    "ensemble": "#AED6F1",
    "spatial": "#F9E79F",
}


def _draw_box(ax, x, y, w, h, text, color, fontsize=7):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="black", linewidth=0.8
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        wrap=True,
    )


def _draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="gray", lw=1.2)
    )


def plot_graphclip_architecture(save_dir: Path):
    """GraphCLIP architecture diagram."""
    with ieee_style():
        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_axis_off()

        _draw_box(ax, 0.2, 4.5, 1.8, 0.8, "Input Image\n224x224x3", COLORS["input"])
        _draw_box(ax, 0.2, 2.8, 1.8, 0.8, "Multi-Res\nEncoder\n(ViT-Small)", COLORS["conv"])
        _draw_arrow(ax, 1.1, 4.5, 1.1, 3.6)

        _draw_box(ax, 2.8, 4.5, 1.8, 0.8, "Disease\nEmbeddings\n(Learnable)", COLORS["text"])
        _draw_box(ax, 2.8, 2.8, 1.8, 0.8, "Dynamic Graph\nAdjacency\nGeneration", COLORS["graph"])
        _draw_arrow(ax, 3.7, 4.5, 3.7, 3.6)

        _draw_box(
            ax, 2.8, 1.2, 1.8, 0.8, "Graph Sparse\nAttention\n(top-k=16)", COLORS["attention"]
        )
        _draw_arrow(ax, 3.7, 2.8, 3.7, 2.0)

        _draw_box(
            ax, 5.4, 2.8, 1.8, 0.8, "Cross-Modal\nSparse Attention\n(top-k=24)", COLORS["fusion"]
        )
        _draw_arrow(ax, 2.0, 3.2, 5.4, 3.2)
        _draw_arrow(ax, 4.6, 1.6, 5.4, 3.0)

        _draw_box(ax, 7.6, 2.8, 1.8, 0.8, "Classifier\n384→256→45", COLORS["output"])
        _draw_arrow(ax, 7.2, 3.2, 7.6, 3.2)

        ax.text(
            5,
            0.5,
            "~45M params | Sparse Attention | Dynamic KG",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )
        ax.set_title("GraphCLIP Architecture", fontsize=11, fontweight="bold", pad=10)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_arch_graphclip")


def plot_vlgnn_architecture(save_dir: Path):
    """VisualLanguageGNN architecture diagram."""
    with ieee_style():
        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_axis_off()

        _draw_box(ax, 0.2, 4.2, 1.6, 0.8, "Image\n224x224", COLORS["input"])
        _draw_box(ax, 0.2, 2.5, 1.6, 0.8, "Multi-Res\nEncoder", COLORS["conv"])
        _draw_arrow(ax, 1.0, 4.2, 1.0, 3.3)

        _draw_box(ax, 2.5, 2.5, 1.6, 0.8, "Adaptive\nRegion\nImportance", COLORS["attention"])
        _draw_arrow(ax, 1.8, 2.9, 2.5, 2.9)

        _draw_box(ax, 5.0, 4.2, 1.6, 0.8, "Disease Text\nEmbeddings\n(256-d)", COLORS["text"])
        _draw_box(ax, 5.0, 2.5, 1.6, 0.8, "Cross-Modal\nSparse Attn\n(top-k=20)", COLORS["fusion"])
        _draw_arrow(ax, 5.8, 4.2, 5.8, 3.3)
        _draw_arrow(ax, 4.1, 2.9, 5.0, 2.9)

        _draw_box(ax, 7.4, 2.5, 1.6, 0.8, "Classifier\n768→256→45", COLORS["output"])
        _draw_arrow(ax, 6.6, 2.9, 7.4, 2.9)

        ax.text(
            5,
            0.5,
            "~48M params | Cross-Modal Fusion | Region Selection",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )
        ax.set_title("VisualLanguageGNN Architecture", fontsize=11, fontweight="bold", pad=10)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_arch_vlgnn")


def plot_sgt_architecture(save_dir: Path):
    """SceneGraphTransformer architecture diagram."""
    with ieee_style():
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.set_axis_off()

        _draw_box(ax, 0.2, 5.5, 1.6, 0.8, "Image\n224x224", COLORS["input"])
        _draw_box(ax, 0.2, 3.8, 1.6, 0.8, "Multi-Res\nEncoder", COLORS["conv"])
        _draw_arrow(ax, 1.0, 5.5, 1.0, 4.6)

        _draw_box(ax, 2.5, 3.8, 1.6, 0.8, "Region\nSampling\n(12 regions)", COLORS["spatial"])
        _draw_arrow(ax, 1.8, 4.2, 2.5, 4.2)

        # 3 ensemble branches
        for i, y in enumerate([5.5, 3.8, 2.1]):
            _draw_box(ax, 4.8, y, 1.4, 0.7, f"Branch {i+1}\nTransformer", COLORS["ensemble"])
            _draw_arrow(ax, 4.1, 4.2, 4.8, y + 0.35)

        _draw_box(ax, 6.8, 3.8, 1.6, 0.8, "Ensemble\nFusion +\nUncertainty", COLORS["fusion"])
        for y in [5.5, 3.8, 2.1]:
            _draw_arrow(ax, 6.2, y + 0.35, 6.8, 4.2)

        _draw_box(ax, 6.8, 2.0, 1.6, 0.8, "Relation\nAttention\n(top-k=8)", COLORS["attention"])
        _draw_arrow(ax, 7.6, 3.8, 7.6, 2.8)

        _draw_box(ax, 6.8, 0.5, 1.6, 0.7, "Calibrated\nOutput (45)", COLORS["output"])
        _draw_arrow(ax, 7.6, 2.0, 7.6, 1.2)

        ax.text(
            5,
            0.1,
            "~52M params | 3-Branch Ensemble | Uncertainty Estimation",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )
        ax.set_title("SceneGraphTransformer Architecture", fontsize=11, fontweight="bold", pad=10)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_arch_sgt")


def plot_vignn_architecture(save_dir: Path):
    """ViGNN architecture diagram."""
    with ieee_style():
        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_axis_off()

        _draw_box(ax, 0.2, 4.2, 1.6, 0.8, "Image\n224x224", COLORS["input"])
        _draw_box(ax, 0.2, 2.5, 1.6, 0.8, "Multi-Res\nEncoder\n(ViT-Small)", COLORS["conv"])
        _draw_arrow(ax, 1.0, 4.2, 1.0, 3.3)

        _draw_box(ax, 2.5, 4.2, 1.6, 0.8, "Patch\nProjection\n(196 patches)", COLORS["conv"])
        _draw_arrow(ax, 1.8, 2.9, 2.5, 4.6)

        _draw_box(ax, 2.5, 2.5, 1.6, 0.8, "Adaptive\nEdge Weight\nGeneration", COLORS["graph"])
        _draw_arrow(ax, 3.3, 4.2, 3.3, 3.3)

        _draw_box(ax, 4.8, 3.3, 1.6, 0.8, "Graph\nMessage\nPassing (x3)", COLORS["graph"])
        _draw_arrow(ax, 4.1, 2.9, 4.8, 3.7)

        _draw_box(
            ax, 4.8, 1.5, 1.6, 0.8, "Disease-Aware\nAttention\n(top-k=64)", COLORS["attention"]
        )

        _draw_box(
            ax, 7.0, 2.5, 1.8, 0.8, "Global + Disease\nClassifier\n768→512→256→45", COLORS["output"]
        )
        _draw_arrow(ax, 6.4, 3.7, 7.0, 2.9)
        _draw_arrow(ax, 6.4, 1.9, 7.0, 2.9)

        ax.text(
            5,
            0.4,
            "~26M params | Graph Message Passing | Disease Prototypes",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )
        ax.set_title("ViGNN Architecture", fontsize=11, fontweight="bold", pad=10)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_arch_vignn")


def plot_all_architectures_comparison(save_dir: Path):
    """Side-by-side architecture summary comparison."""
    with ieee_style():
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
        ax.set_axis_off()

        models = [
            ("GraphCLIP", "~45M", "Dynamic Graph\n+ Sparse Attn", COLORS["graph"]),
            ("VisualLanguageGNN", "~48M", "Cross-Modal\nFusion", COLORS["text"]),
            ("SceneGraphTransformer", "~52M", "Ensemble +\nUncertainty", COLORS["ensemble"]),
            ("ViGNN", "~26M", "Graph Message\nPassing", COLORS["attention"]),
        ]

        for i, (name, params, desc, color) in enumerate(models):
            x = 0.5 + i * 2.3
            _draw_box(ax, x, 1.8, 2.0, 1.2, f"{name}\n{params}\n\n{desc}", color, fontsize=7)

        # Common base
        _draw_box(
            ax,
            1.5,
            0.3,
            5.5,
            0.7,
            "Shared: Multi-Resolution ViT-Small Encoder + ClinicalKnowledgeGraph + SparseTopKAttention",
            "#F0F0F0",
            fontsize=7,
        )
        for i in range(4):
            x = 1.5 + i * 2.3
            _draw_arrow(ax, x, 1.8, 4.25, 1.0)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3.5)
        ax.set_title("Model Architecture Comparison", fontsize=11, fontweight="bold", pad=10)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_arch_comparison")


def generate_all_architecture_plots(save_dir: str | Path = "outputs/plots/architecture"):
    """Generate all architecture diagrams."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating architecture diagrams -> {save_dir}")
    plot_graphclip_architecture(save_dir)
    print("  [1/5] GraphCLIP")
    plot_vlgnn_architecture(save_dir)
    print("  [2/5] VisualLanguageGNN")
    plot_sgt_architecture(save_dir)
    print("  [3/5] SceneGraphTransformer")
    plot_vignn_architecture(save_dir)
    print("  [4/5] ViGNN")
    plot_all_architectures_comparison(save_dir)
    print("  [5/5] Architecture comparison")
    print(f"  Architecture plots saved to {save_dir}")
