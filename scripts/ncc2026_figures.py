#!/usr/bin/env python3
"""Generate the camera-ready figures for the NCC 2026 paper.

fig1_architecture.pdf -- pipeline with explicitly numbered components, which is
                         what the reviewer asked for
fig2_evidence.pdf     -- (a) reliability diagram before/after recalibration
                         (b) per-disease AUC against test prevalence
                         (c) referral-decision ROC

Usage:
    python3 scripts/ncc2026_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/ncc2026"
FIGS = REPO / "docs/Reports/figs"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

INK = "#1a1a1a"
ACCENT = "#1f5fa9"
WARM = "#b3541e"
MUTED = "#6b7280"


def figure_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    stages = [
        ("Fundus\nimage", None, "#f3f4f6"),
        ("Quality\ngate", "1", "#e8eef7"),
        ("RETFound\nViT-L + LoRA", "2", "#dbe6f5"),
        ("Bottleneck\nhead, 24 cls", "3", "#dbe6f5"),
        ("Calibration\n+ thresholds", "4", "#f7e9dd"),
        ("Knowledge\ngraph", "5", "#f7e9dd"),
        ("Explain-\nability", "6", "#eceff3"),
        ("Referral\n+ audit", "7", "#eceff3"),
    ]

    x, w, h, y = 1.0, 10.4, 12.0, 13.0
    gap = 2.1
    centres = []
    for label, num, colour in stages:
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.25,rounding_size=0.8",
            linewidth=0.8,
            edgecolor=INK,
            facecolor=colour,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h / 2 - 0.6, label, ha="center", va="center", fontsize=6.3, color=INK
        )
        if num:
            ax.text(
                x + 1.1,
                y + h - 1.2,
                num,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="circle,pad=0.18", facecolor=ACCENT, edgecolor="none"),
            )
        centres.append(x + w / 2)
        x += w + gap

    for i in range(len(stages) - 1):
        x0 = centres[i] + w / 2
        ax.annotate(
            "",
            xy=(x0 + gap, y + h / 2),
            xytext=(x0, y + h / 2),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8),
        )

    # on-device path
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (centres[2] - w / 2, 1.2),
            centres[5] - centres[2] + w,
            7.4,
            boxstyle="round,pad=0.25,rounding_size=0.8",
            linewidth=0.8,
            linestyle="--",
            edgecolor=WARM,
            facecolor="#fdf6f0",
        )
    )
    ax.text(
        (centres[2] + centres[5]) / 2,
        4.9,
        "8  On-device path: MobileNetV3 student distilled from (2)-(3), INT8 ONNX, "
        "offline-first, deferred sync",
        ha="center",
        va="center",
        fontsize=6.8,
        color=WARM,
    )
    ax.annotate(
        "",
        xy=(centres[3], 8.6),
        xytext=(centres[3], y),
        arrowprops=dict(arrowstyle="-|>", color=WARM, lw=0.8, linestyle="--"),
    )
    ax.annotate(
        "",
        xy=(centres[6], y),
        xytext=(centres[6], 8.6),
        arrowprops=dict(arrowstyle="-|>", color=WARM, lw=0.8, linestyle="--"),
    )

    ax.text(
        1.0,
        30.5,
        "Server path (full model)",
        fontsize=7,
        color=MUTED,
    )
    fig.tight_layout(pad=0.15)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig1_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_architecture.pdf")


def _reliability(ax, probs, labels, colour, label, bins=12):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    p, y = probs.ravel(), labels.ravel()
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() >= 20:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    ax.plot(xs, ys, "o-", color=colour, label=label, markersize=3, linewidth=1.1)


def figure_evidence() -> None:
    raw = np.load(OUT / "probs_test_fp32.npz", allow_pickle=True)
    cal = np.load(OUT / "probs_test_calibrated.npz", allow_pickle=True)
    classes = [str(c) for c in raw["classes"]]
    y = raw["labels"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=0.9, label="perfect")
    _reliability(ax, raw["probs"], y, WARM, "as trained")
    _reliability(ax, cal["probs"], y, ACCENT, "recalibrated")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("(a) Reliability")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    prev = y.mean(axis=0)
    aucs = np.array(
        [
            roc_auc_score(y[:, c], raw["probs"][:, c]) if 0 < y[:, c].sum() < len(y) else np.nan
            for c in range(len(classes))
        ]
    )
    ax.scatter(prev * 100, aucs, s=16, color=ACCENT, zorder=3)
    for c, pv, au in zip(classes, prev, aucs):
        if pv > 0.10 or au < 0.72:
            ax.annotate(c, (pv * 100, au), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0.5, ls=":", color=MUTED, lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("test prevalence (%, log scale)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("(b) Per-disease AUC vs prevalence")

    ax = axes[2]
    df = pd.read_csv(OUT / "cache/test_labels.csv", encoding="utf-8-sig")
    yr = df["Disease_Risk"].to_numpy()
    score = raw["probs"].max(axis=1)
    fpr, tpr, _ = roc_curve(yr, score)
    ax.plot(fpr, tpr, color=ACCENT, lw=1.3)
    ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=0.9)
    ref = json.loads((OUT / "referral.json").read_text())
    op = ref["operating_points"]["sens95"]["held_out_test"]
    ax.scatter([1 - op["specificity"]], [op["sensitivity"]], color=WARM, zorder=4, s=22)
    ax.annotate(
        f"sens {op['sensitivity']:.2f}\nspec {op['specificity']:.2f}",
        (1 - op["specificity"], op["sensitivity"]),
        fontsize=6,
        xytext=(6, -14),
        textcoords="offset points",
        color=WARM,
    )
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("sensitivity")
    ax.set_title(f"(c) Referral decision, AUC {ref['auc']:.3f}")

    fig.tight_layout(pad=0.4)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig2_evidence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_evidence.pdf")


if __name__ == "__main__":
    figure_architecture()
    figure_evidence()
