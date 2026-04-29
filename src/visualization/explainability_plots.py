"""
Explainability Visualizations - IEEE Publication Quality.
GradCAM, attention maps, clinical knowledge graph, feature attribution.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image

from src.visualization.ieee_style import (
    ieee_style, ieee_figure, save_ieee, add_watermark,
    IEEE_COLORS, CB_PALETTE,
)


def plot_gradcam_grid(
    images: list[np.ndarray],
    heatmaps: list[np.ndarray],
    predictions: list[str],
    confidences: list[float],
    save_dir: Path,
    model_name: str = "ViGNN",
    n_samples: int = 4,
):
    """GradCAM visualization grid (notebook cell 38 enhanced)."""
    n = min(n_samples, len(images))
    with ieee_style():
        fig, axes = plt.subplots(2, n, figsize=(7, 3.5), dpi=100)
        if n == 1:
            axes = axes.reshape(2, 1)

        for i in range(n):
            # Original image
            img = images[i]
            if img.max() > 1:
                img = img / 255.0
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"{predictions[i]}\n({confidences[i]:.1%})", fontsize=7)
            axes[0, i].axis("off")

            # Heatmap overlay
            heatmap = heatmaps[i]
            if heatmap.ndim == 2:
                heatmap_color = cv2.applyColorMap(
                    (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

                if img.shape[:2] != heatmap_color.shape[:2]:
                    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))

                overlay = 0.6 * img + 0.4 * heatmap_color
                overlay = np.clip(overlay, 0, 1)
            else:
                overlay = heatmap

            axes[1, i].imshow(overlay)
            axes[1, i].axis("off")

        axes[0, 0].set_ylabel("Original", fontsize=8, fontweight="bold")
        axes[1, 0].set_ylabel("GradCAM", fontsize=8, fontweight="bold")

        fig.suptitle(f"{model_name} - GradCAM Attention Analysis", fontsize=10, fontweight="bold")
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_gradcam_grid")


def plot_clinical_knowledge_graph(
    adjacency: np.ndarray,
    disease_names: list[str],
    save_dir: Path,
    top_k: int = 15,
):
    """Clinical knowledge graph visualization with edge weights."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.8)

        # Select top-K diseases with most connections
        edge_counts = (adjacency > 0.1).sum(axis=1) - 1  # Exclude self-loops
        top_idx = np.argsort(edge_counts)[::-1][:top_k]
        sub_adj = adjacency[np.ix_(top_idx, top_idx)]
        sub_names = [disease_names[i] for i in top_idx]

        # (a) Adjacency heatmap
        import seaborn as sns
        mask = np.eye(len(sub_names), dtype=bool)
        sns.heatmap(
            sub_adj, ax=axes[0], mask=mask,
            xticklabels=sub_names, yticklabels=sub_names,
            cmap="YlOrRd", vmin=0, vmax=0.8,
            annot=True, fmt=".2f", annot_kws={"size": 5},
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Edge Weight", "shrink": 0.8},
        )
        axes[0].set_title("(a) Clinical Relationship Weights")
        axes[0].tick_params(labelsize=6, labelrotation=45)

        # (b) Graph visualization (circular layout)
        n = len(sub_names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 1.0
        positions = np.column_stack([np.cos(angles), np.sin(angles)]) * radius

        # Draw edges
        for i in range(n):
            for j in range(i + 1, n):
                w = sub_adj[i, j]
                if w > 0.1:
                    axes[1].plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        color="gray", alpha=min(w * 1.5, 0.8), lw=w * 3,
                    )

        # Draw nodes
        node_sizes = edge_counts[top_idx]
        norm_sizes = (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min() + 1e-8)
        colors = plt.cm.viridis(norm_sizes * 0.8 + 0.1)

        for i, (pos, name, c) in enumerate(zip(positions, sub_names, colors)):
            axes[1].scatter(pos[0], pos[1], s=200 + norm_sizes[i] * 400,
                           c=[c], edgecolors="black", linewidth=0.5, zorder=3)
            axes[1].annotate(name, pos, fontsize=6, ha="center", va="center", fontweight="bold")

        axes[1].set_xlim(-1.5, 1.5)
        axes[1].set_ylim(-1.5, 1.5)
        axes[1].set_aspect("equal")
        axes[1].axis("off")
        axes[1].set_title("(b) Clinical Knowledge Graph")

        fig.suptitle("Clinical Knowledge Graph - Disease Relationships", fontsize=10, fontweight="bold")
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_clinical_knowledge_graph")


def plot_knowledge_graph_comprehensive(
    kg,
    save_dir: Path,
):
    """
    Comprehensive 4-panel KG visualization:
    (a) Adjacency heatmap, (b) Uganda prevalence, (c) Category distribution, (d) Most connected diseases.
    """
    import seaborn as sns
    with ieee_style():
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=100)

        # (a) Adjacency Matrix
        adj = kg.get_adjacency_matrix()
        sns.heatmap(adj, cmap='YlOrRd', ax=axes[0, 0], cbar_kws={'label': 'Relationship Strength'})
        axes[0, 0].set_title('(a) Disease Relationship Adjacency Matrix', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Disease Index', fontsize=9)
        axes[0, 0].set_ylabel('Disease Index', fontsize=9)

        # (b) Uganda Prevalence
        prev = kg.uganda_prevalence
        if prev:
            diseases_p = list(prev.keys())
            vals_p = list(prev.values())
            colors_p = plt.cm.RdYlGn_r([p for p in vals_p])
            axes[0, 1].barh(diseases_p, vals_p, color=colors_p, edgecolor='black', linewidth=0.5)
            axes[0, 1].set_xlabel('Prevalence Weight', fontsize=9)
            axes[0, 1].set_title('(b) Uganda-Specific Disease Prevalence', fontsize=11, fontweight='bold')
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].grid(axis='x', alpha=0.3, linestyle='--')
            for i, v in enumerate(vals_p):
                axes[0, 1].text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=7, fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No prevalence data', ha='center', va='center', transform=axes[0, 1].transAxes)

        # (c) Disease Categories
        cat_counts = {cat: len(ds) for cat, ds in kg.categories.items()}
        if cat_counts:
            cats = list(cat_counts.keys())
            cnts = list(cat_counts.values())
            colors_c = plt.cm.Set3(range(len(cats)))
            wedges, texts, autotexts = axes[1, 0].pie(
                cnts, labels=cats, autopct='%1.0f%%', colors=colors_c, startangle=90,
                textprops={'fontsize': 7, 'fontweight': 'bold'}
            )
            for at in autotexts:
                at.set_fontsize(7)
            axes[1, 0].set_title('(c) Disease Categories', fontsize=11, fontweight='bold')

        # (d) Most Connected Diseases
        cooc = kg.cooccurrence
        cooc_counts = {d: len(r) for d, r in cooc.items()}
        top_d = sorted(cooc_counts.items(), key=lambda x: x[1], reverse=True)[:12]
        if top_d:
            names_d = [d[0] for d in top_d]
            cnts_d = [d[1] for d in top_d]
            colors_d = plt.cm.viridis([c / max(cnts_d) for c in cnts_d])
            axes[1, 1].barh(names_d, cnts_d, color=colors_d, edgecolor='black', linewidth=0.5)
            axes[1, 1].set_xlabel('Related Diseases', fontsize=9)
            axes[1, 1].set_title('(d) Most Connected Diseases', fontsize=11, fontweight='bold')
            axes[1, 1].invert_yaxis()
            axes[1, 1].grid(axis='x', alpha=0.3, linestyle='--')
            for i, v in enumerate(cnts_d):
                axes[1, 1].text(v + 0.1, i, str(v), va='center', fontsize=7, fontweight='bold')

        fig.suptitle('Clinical Knowledge Graph - Advanced Reasoning Analysis', fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_knowledge_graph_comprehensive")


def plot_severity_and_risk(
    kg,
    save_dir: Path,
):
    """Severity levels + risk factors visualization."""
    with ieee_style():
        fig, axes = ieee_figure(1, 2, width="double", height_ratio=0.6)

        # (a) Severity levels
        sev = kg.severity_levels
        if sev:
            sorted_sev = sorted(sev.items(), key=lambda x: x[1], reverse=True)
            names_s = [s[0] for s in sorted_sev[:20]]
            vals_s = [s[1] for s in sorted_sev[:20]]
            color_map = {0: '#4CAF50', 1: '#FFC107', 2: '#FF9800', 3: '#F44336'}
            colors_s = [color_map.get(v, '#999') for v in vals_s]
            axes[0].barh(names_s, vals_s, color=colors_s, edgecolor='black', lw=0.3)
            axes[0].set_xlabel('Severity (0=None, 3=Severe)')
            axes[0].set_title('(a) Disease Severity Levels')
            axes[0].invert_yaxis()
            axes[0].set_xticks([0, 1, 2, 3])
            axes[0].set_xticklabels(['None', 'Mild', 'Moderate', 'Severe'], fontsize=7)

        # (b) Age risk factors
        age_rf = kg.age_risk_factors
        if age_rf:
            names_a = list(age_rf.keys())
            vals_a = list(age_rf.values())
            axes[1].barh(names_a, vals_a, color=IEEE_COLORS["orange"], edgecolor='black', lw=0.3)
            axes[1].set_xlabel('Age Risk Factor')
            axes[1].set_title('(b) Age-Adjusted Risk Factors')
            axes[1].invert_yaxis()
            axes[1].set_xlim(0, 1)
            for i, v in enumerate(vals_a):
                axes[1].text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=7)

        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_severity_and_risk")


def plot_feature_attribution(
    attributions: dict[str, np.ndarray],
    save_dir: Path,
    model_name: str = "ViGNN",
):
    """Feature attribution comparison across methods."""
    n_methods = len(attributions)
    if n_methods == 0:
        return

    with ieee_style():
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(7, 2.5), dpi=100)

        for i, (method, attr) in enumerate(attributions.items()):
            if attr.ndim == 3:
                attr = np.abs(attr).mean(axis=2)  # Average across channels
            # Normalize
            if attr.max() > attr.min():
                attr = (attr - attr.min()) / (attr.max() - attr.min())

            im = axes[i + 1].imshow(attr, cmap="hot", vmin=0, vmax=1)
            axes[i + 1].set_title(method, fontsize=8)
            axes[i + 1].axis("off")

        # First panel: original image placeholder
        axes[0].text(0.5, 0.5, "Original\nImage", ha="center", va="center",
                    fontsize=9, transform=axes[0].transAxes)
        axes[0].set_axis_off()

        fig.suptitle(f"{model_name} - Feature Attribution Methods", fontsize=10, fontweight="bold")
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_feature_attribution")


def plot_prediction_confidence_distribution(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    disease_names: list[str],
    save_dir: Path,
    top_k: int = 6,
):
    """Confidence distribution for positive vs negative samples per class."""
    pos_counts = y_true.sum(axis=0)
    top_idx = np.argsort(pos_counts)[::-1][:top_k]

    with ieee_style():
        rows = (top_k + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(7, rows * 2), dpi=100)
        axes = axes.flatten()

        for plot_i, class_idx in enumerate(top_idx):
            ax = axes[plot_i]
            pos_probs = y_prob[y_true[:, class_idx] == 1, class_idx]
            neg_probs = y_prob[y_true[:, class_idx] == 0, class_idx]

            bins = np.linspace(0, 1, 25)
            if len(neg_probs) > 0:
                ax.hist(neg_probs, bins=bins, alpha=0.6, color=IEEE_COLORS["blue"],
                       label=f"Neg (n={len(neg_probs)})", density=True)
            if len(pos_probs) > 0:
                ax.hist(pos_probs, bins=bins, alpha=0.6, color=IEEE_COLORS["red"],
                       label=f"Pos (n={len(pos_probs)})", density=True)
            ax.axvline(0.5, ls="--", color="black", lw=0.8, alpha=0.5)
            ax.set_title(disease_names[class_idx], fontsize=8)
            ax.legend(fontsize=5)
            ax.set_xlabel("Probability", fontsize=7)

        # Hide unused axes
        for j in range(len(top_idx), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Prediction Confidence Distribution by Class", fontsize=10, fontweight="bold")
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_confidence_distribution")


def generate_all_explainability_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    disease_names: list[str],
    adjacency: np.ndarray | None = None,
    knowledge_graph=None,
    gradcam_data: dict | None = None,
    save_dir: str | Path = "outputs/plots/explainability",
    model_name: str = "ViGNN",
):
    """Generate all explainability plots including comprehensive KG analysis."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating explainability plots -> {save_dir}")

    plot_prediction_confidence_distribution(y_prob, y_true, disease_names, save_dir)
    print("  [1/5] Confidence distribution")

    if adjacency is not None:
        plot_clinical_knowledge_graph(adjacency, disease_names, save_dir)
        print("  [2/5] Clinical knowledge graph (adjacency + network)")
    else:
        print("  [2/5] Skipped (no adjacency matrix)")

    if knowledge_graph is not None:
        plot_knowledge_graph_comprehensive(knowledge_graph, save_dir)
        print("  [3/5] KG comprehensive (prevalence, categories, connections)")
        plot_severity_and_risk(knowledge_graph, save_dir)
        print("  [4/5] Severity levels + age risk factors")
    else:
        print("  [3/5] Skipped (no knowledge graph)")
        print("  [4/5] Skipped (no knowledge graph)")

    if gradcam_data:
        plot_gradcam_grid(
            gradcam_data["images"], gradcam_data["heatmaps"],
            gradcam_data["predictions"], gradcam_data["confidences"],
            save_dir, model_name,
        )
        print("  [5/5] GradCAM grid")
    else:
        print("  [5/5] Skipped (no GradCAM data)")

    print(f"  Explainability plots saved to {save_dir}")
