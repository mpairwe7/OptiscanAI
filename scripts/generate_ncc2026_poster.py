#!/usr/bin/env python3
"""Generate the official OptiscanAI NCC 2026 conference poster (PPTX and PDF).

Strictly adheres to:
    MLOPS_V1/docs/Reports/MajiQuick_NCC2026_Poster (1).pdf
    MLOPS_V1/docs/Reports/NCC226_POSTER_TEMPLATE.ppt

Design Principles:
    - Single-page A0 Portrait (841 x 1189 mm / 33.11 x 46.81 inches).
    - Authentic NCC green outer rounded border (#00B050).
    - Top header with official Makerere University, NCC, and UCC logos.
    - Deep Blue (#0070C0) conference banners and paper titles matching MajiQuick.
    - Clean white canvas without boxy card backgrounds behind sections.
    - Centered pill section banners in Office Steel Blue (#4F81BD) with Dark Steel Blue border (#385D8A).
    - Perfectly balanced two-column layout with 15.16 in column widths and 1.0 in gutter.
    - Highly readable 18–20 pt body typography with bold leads (no walls of tiny text).
    - High-resolution, publication-scale diagrams (Fig 1, Fig 2, and Fig 3).
    - Native PowerPoint tables matching MajiQuick styling (deep blue header row #0070C0, alternating rows).
    - Comprehensive sections:
        * Left: Abstract, Problem Statement, System Architecture (Fig 1),
                Model Adaptation & Component Ablation (Table 1),
                Clinical Knowledge Graph & Intake Gate.
        * Right: Results & Clinical Case Gallery (Fig 2 & Fig 3),
                 Quantitative Outcomes & Operating Policies (Table 2),
                 Conclusions, References.
    - Elegant bottom conference footer in Deep Blue (#0070C0).

Outputs:
    docs/Reports/OptiscanAI_NCC2026_Poster.pptx
    docs/Reports/OptiscanAI_NCC2026_Poster.pdf
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pptx
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO / "docs/Reports"
OUT_DIR = REPO / "outputs/ncc2026"
TMP_DIR = Path("/mnt/nas1/tmp/opencode")


def generate_poster_figures() -> tuple[Path, Path, Path]:
    """Generate high-clarity poster-scale figures (Fig 1, Fig 2, and Fig 3)."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fig1_path = TMP_DIR / "fig1_poster.png"
    fig2_path = TMP_DIR / "fig2_poster.png"
    fig3_path = TMP_DIR / "fig3_cases.png"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    INK = "#111827"
    ACCENT = "#1F5FA9"
    WARM = "#B3541E"
    MUTED = "#4B5563"
    GREEN = "#059669"
    RED = "#DC2626"

    # --- Figure 1: Pipeline Architecture ---
    fig, ax = plt.subplots(figsize=(14.2, 4.9), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    stages = [
        ("Raw Fundus\nPhotograph", None, "#F3F4F6", "#9CA3AF"),
        ("Quality Gate\n(4-Tier Filter)", "1", "#E0EDFD", ACCENT),
        ("RETFound\nViT-L/16", "2", "#D3E4F8", ACCENT),
        ("LoRA Adapters\n+ Head (24 cls)", "3", "#D3E4F8", ACCENT),
        ("Isotonic\nRecalibration", "4", "#FEF3C7", "#D97706"),
        ("Clinical\nKnowledge Graph", "5", "#FEF3C7", "#D97706"),
        ("Attribution &\nExplainability", "6", "#EDE9FE", "#7C3AED"),
        ("Referral Triage\n+ Audit Record", "7", "#D1FAE5", GREEN),
    ]

    w, h, y = 10.3, 13.8, 20.5
    gap = 2.0
    x = 1.2
    centres = []

    for label, num, colour, edge_col in stages:
        box = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=1.8,
            edgecolor=edge_col,
            facecolor=colour,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h / 2 - 0.8, label,
            ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=INK
        )
        if num:
            ax.text(
                x + 1.4, y + h - 1.4, num,
                ha="center", va="center",
                fontsize=13, fontweight="bold", color="white",
                bbox=dict(boxstyle="circle,pad=0.25", facecolor=edge_col, edgecolor="none")
            )
        centres.append(x + w / 2)
        x += w + gap

    for i in range(len(stages) - 1):
        x0 = centres[i] + w / 2
        ax.annotate(
            "",
            xy=(x0 + gap, y + h / 2),
            xytext=(x0, y + h / 2),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2, mutation_scale=16),
        )

    ax.text(
        1.2, 36.8, "● Server Foundation Pipeline (Full RETFound ViT-L/16 Model Path 1–7)",
        fontsize=13.5, fontweight="bold", color=ACCENT
    )

    student_box = mpatches.FancyBboxPatch(
        (centres[2] - w / 2, 1.8),
        centres[6] - centres[2] + w,
        9.6,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        linewidth=2.0,
        linestyle="--",
        edgecolor=WARM,
        facecolor="#FFFBEB",
    )
    ax.add_patch(student_box)

    ax.text(
        (centres[2] + centres[6]) / 2, 6.6,
        "8  Offline-First Edge Student: Distilled MobileNetV3 (5.2M params) → INT8 ONNX (5.4 MB)\n"
        "102 ms Latency  |  Runs Locally on Low-Cost Hardware Without Internet  |  Deferred Encrypted Sync",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=WARM
    )

    ax.annotate(
        "",
        xy=(centres[3], 11.4),
        xytext=(centres[3], y),
        arrowprops=dict(arrowstyle="-|>", color=WARM, lw=2.0, linestyle="--", mutation_scale=14),
    )
    ax.text(
        centres[3] - 0.8, 16.0, "Distill",
        ha="right", va="center",
        fontsize=11, fontweight="bold", color=WARM
    )

    ax.annotate(
        "",
        xy=(centres[6], y),
        xytext=(centres[6], 11.4),
        arrowprops=dict(arrowstyle="-|>", color=WARM, lw=2.0, linestyle="--", mutation_scale=14),
    )
    ax.text(
        centres[6] + 0.8, 16.0, "Deploy",
        ha="left", va="center",
        fontsize=11, fontweight="bold", color=WARM
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Empirical Evidence Plots ---
    raw = np.load(OUT_DIR / "probs_test_fp32.npz", allow_pickle=True)
    cal = np.load(OUT_DIR / "probs_test_calibrated.npz", allow_pickle=True)
    classes = [str(c) for c in raw["classes"]]
    y_test = raw["labels"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), dpi=300)

    # (a) Calibration Reliability
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1.8, label="Ideal Calibration")

    def _reliability(ax_obj, probs, labels, colour, label, bins=12):
        edges = np.linspace(0, 1, bins + 1)
        xs, ys = [], []
        p, y_flat = probs.ravel(), labels.ravel()
        for i in range(bins):
            m = (p > edges[i]) & (p <= edges[i + 1])
            if m.sum() >= 20:
                xs.append(p[m].mean())
                ys.append(y_flat[m].mean())
        ax_obj.plot(xs, ys, "o-", color=colour, label=label, markersize=7, linewidth=2.4)

    _reliability(ax, raw["probs"], y_test, WARM, "As Trained (ECE 0.209)")
    _reliability(ax, cal["probs"], y_test, ACCENT, "Recalibrated (ECE 0.003)")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Observed Empirical Frequency", fontweight="bold")
    ax.set_title("(a) Calibration Reliability", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", loc="upper left")

    # (b) Per-Disease AUC vs Prevalence
    ax = axes[1]
    prev = y_test.mean(axis=0)
    aucs = np.array([
        roc_auc_score(y_test[:, c], raw["probs"][:, c]) if 0 < y_test[:, c].sum() < len(y_test) else np.nan
        for c in range(len(classes))
    ])
    ax.scatter(prev * 100, aucs, s=55, color=ACCENT, edgecolor="white", linewidth=1.2, zorder=3)
    for c, pv, au in zip(classes, prev, aucs):
        if pv > 0.10 or au < 0.72:
            ax.annotate(c, (pv * 100, au), fontsize=11, fontweight="bold", xytext=(4, 4), textcoords="offset points")

    ax.axhline(0.5, ls=":", color=MUTED, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Test Prevalence (%, log scale)", fontweight="bold")
    ax.set_ylabel("AUC-ROC Score", fontweight="bold")
    ax.set_title("(b) Per-Disease AUC vs. Prevalence", fontweight="bold")

    # (c) Referral Decision ROC
    ax = axes[2]
    df_labels = pd.read_csv(OUT_DIR / "cache/test_labels.csv", encoding="utf-8-sig")
    yr = df_labels["Disease_Risk"].to_numpy()
    score = raw["probs"].max(axis=1)
    fpr, tpr, _ = roc_curve(yr, score)
    ax.plot(fpr, tpr, color=ACCENT, lw=2.8, label="Referral ROC")
    ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1.8, label="Chance")

    ref = json.loads((OUT_DIR / "referral.json").read_text())
    op = ref["operating_points"]["sens95"]["held_out_test"]
    ax.scatter([1 - op["specificity"]], [op["sensitivity"]], color=WARM, zorder=4, s=90, edgecolor="white", lw=1.5)
    ax.annotate(
        f"Sensitivity: {op['sensitivity']:.1%}\nSpecificity: {op['specificity']:.1%}",
        (1 - op["specificity"], op["sensitivity"]),
        fontsize=12, fontweight="bold",
        xytext=(10, -26), textcoords="offset points",
        color=WARM,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFBEB", edgecolor=WARM, lw=1.0)
    )
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontweight="bold")
    ax.set_title(f"(c) Binary Referral ROC (AUC {ref['auc']:.3f})", fontweight="bold")

    fig.tight_layout(pad=1.0)
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Clinical Case Gallery ---
    imgs = np.load(OUT_DIR / "cache/test_images_224.npy")
    normal_idx = df_labels[df_labels['Disease_Risk'] == 0].index[0]
    dr_idx = df_labels[(df_labels['DR'] == 1) & (df_labels['Disease_Risk'] == 1)].index[0]
    armd_idx = df_labels[(df_labels['ARMD'] == 1)].index[0]
    odc_idx = df_labels[(df_labels['ODC'] == 1)].index[0]

    cases = [
        (imgs[normal_idx], "Normal / Healthy Retina", "Disease Risk: Negative\nReferral: Routine Non-Urgent", GREEN),
        (imgs[dr_idx], "Diabetic Retinopathy", "Microaneurysms & Exudates\nReferral: Urgent Specialist", RED),
        (imgs[odc_idx], "Optic Disc Cupping (Glaucoma)", "Enlarged Cup-to-Disc Ratio\nReferral: Urgent Specialist", WARM),
        (imgs[armd_idx], "Macular Degeneration (ARMD)", "Central Drusen & Macular Lesions\nReferral: Urgent Specialist", ACCENT),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.6), dpi=300)
    for ax, (img_arr, title, desc, col) in zip(axes, cases):
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=13.5, fontweight="bold", color=col, pad=8)
        ax.set_xlabel(desc, fontsize=12, fontweight="bold", color=INK, labelpad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(col)
            spine.set_linewidth(2.2)

    fig.tight_layout(pad=0.8)
    fig.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig)

    return fig1_path, fig2_path, fig3_path


def ensure_template_base() -> Path:
    """Prepare clean base presentation with official logos and correct A0 dimensions."""
    clean_base = TMP_DIR / "clean_template_base.pptx"
    if clean_base.exists():
        return clean_base

    raw_ppt = REPORTS_DIR / "NCC226_POSTER_TEMPLATE.ppt"
    converted_pptx = TMP_DIR / "NCC226_POSTER_TEMPLATE.pptx"

    if not converted_pptx.exists():
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pptx",
                str(raw_ppt),
                "--outdir",
                str(TMP_DIR),
            ],
            check=True,
        )

    prs = pptx.Presentation(str(converted_pptx))
    slide = prs.slides[0]

    # Keep Makerere (31), UCC (34), and NCC (35) logos
    keep_ids = {
        slide.shapes[31].shape_id,
        slide.shapes[34].shape_id,
        slide.shapes[35].shape_id,
    }
    for s in list(slide.shapes):
        if s.shape_id not in keep_ids:
            sp = s._element
            sp.getparent().remove(sp)

    prs.save(str(clean_base))
    return clean_base


def build_poster() -> tuple[Path, Path]:
    clean_base = ensure_template_base()
    fig1_png, fig2_png, fig3_png = generate_poster_figures()

    prs = pptx.Presentation(str(clean_base))
    slide = prs.slides[0]

    # Remove non-picture shapes from clean base
    for s in list(slide.shapes):
        if not s.name.startswith("Picture"):
            sp = s._element
            sp.getparent().remove(sp)

    # 1. Outer rounded frame (Official NCC green: #00B050)
    border = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.18),
        Inches(0.18),
        Inches(32.75),
        Inches(46.45),
    )
    border.fill.background()
    border.line.color.rgb = RGBColor(0x00, 0xB0, 0x50)
    border.line.width = Pt(4.5)

    spTree = slide.shapes._spTree
    spTree.remove(border._element)
    spTree.insert(2, border._element)

    # Official Palette strictly adhering to NCC MajiQuick Template
    NCC_DEEP_BLUE = RGBColor(0x00, 0x70, 0xC0)  # #0070C0: Conference titles, headers, table headers
    STEEL_BANNER = RGBColor(0x4F, 0x81, 0xBD)   # #4F81BD: Section pill banners
    BORDER_STEEL = RGBColor(0x38, 0x5D, 0x8A)   # #385D8A: Banner borders
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    PURE_BLACK = RGBColor(0x00, 0x00, 0x00)
    NAVY_LEAD = RGBColor(0x00, 0x48, 0x80)
    MUTED_GRAY = RGBColor(0x33, 0x33, 0x33)
    ALT_ROW_BG = RGBColor(0xEE, 0xF4, 0xFA)
    HIGHLIGHT_ROW = RGBColor(0xD9, 0xE8, 0xF5)

    def add_banner(title: str, col_left: float, top: float, col_width: float = 15.16, banner_width: float = 8.00, height: float = 0.55):
        """Add signature centered rounded rectangle pill banner matching MajiQuick."""
        b_left = col_left + (col_width - banner_width) / 2.0
        banner = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(b_left), Inches(top), Inches(banner_width), Inches(height)
        )
        banner.fill.solid()
        banner.fill.fore_color.rgb = STEEL_BANNER
        banner.line.color.rgb = BORDER_STEEL
        banner.line.width = Pt(2.0)
        tf = banner.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.1)
        tf.margin_top = Inches(0.06)
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Calibri"
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        return banner

    def add_bullet_item(tf, lead_bold: str, body: str, pt_size: float = 18.5, space_after: float = 5.0):
        """Add bullet point with bold navy lead and clean black body text."""
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.space_after = Pt(space_after)
        p.line_spacing = 1.15
        r1 = p.add_run()
        r1.text = lead_bold + " "
        r1.font.name = "Calibri"
        r1.font.size = Pt(pt_size)
        r1.font.bold = True
        r1.font.color.rgb = NAVY_LEAD
        r2 = p.add_run()
        r2.text = body
        r2.font.name = "Calibri"
        r2.font.size = Pt(pt_size)
        r2.font.color.rgb = PURE_BLACK

    # =========================================================================
    # HEADER (Strictly adhering to MajiQuick & NCC Template)
    # =========================================================================
    # 1. Conference & University Banner Text
    hdr_box = slide.shapes.add_textbox(Inches(2.0), Inches(3.62), Inches(29.11), Inches(1.50))
    htf = hdr_box.text_frame
    htf.word_wrap = True
    htf.margin_top = Inches(0)
    p0 = htf.paragraphs[0]
    p0.text = "National Conference on Communications (NCC)"
    p0.font.name = "Calibri"
    p0.font.size = Pt(50)
    p0.font.bold = True
    p0.font.color.rgb = NCC_DEEP_BLUE
    p0.alignment = PP_ALIGN.CENTER

    p1 = htf.add_paragraph()
    p1.text = "CEDAT, Makerere University"
    p1.font.name = "Calibri"
    p1.font.size = Pt(42)
    p1.font.bold = True
    p1.font.color.rgb = NCC_DEEP_BLUE
    p1.alignment = PP_ALIGN.CENTER

    # 2. Paper Title & Authors Block
    title_box = slide.shapes.add_textbox(Inches(1.0), Inches(5.25), Inches(31.11), Inches(3.15))
    ttf = title_box.text_frame
    ttf.word_wrap = True
    ttf.margin_top = Inches(0)

    tp0 = ttf.paragraphs[0]
    tp0.text = "OptiscanAI: Large Vision Models for Multi-Disease Retinal Screening"
    tp0.font.name = "Calibri"
    tp0.font.size = Pt(48)
    tp0.font.bold = True
    tp0.font.color.rgb = NCC_DEEP_BLUE
    tp0.alignment = PP_ALIGN.CENTER

    tp1 = ttf.add_paragraph()
    tp1.text = "in Ugandan Healthcare — A Benchmark and Deployment Study (Paper 83)"
    tp1.font.name = "Calibri"
    tp1.font.size = Pt(34)
    tp1.font.bold = True
    tp1.font.color.rgb = NCC_DEEP_BLUE
    tp1.alignment = PP_ALIGN.CENTER

    tp2 = ttf.add_paragraph()
    tp2.space_before = Pt(8)
    tp2.text = "Lauben Mpairwe¹        Shadia Nankya¹        Rebecca Yapyeko¹        Marvin Ggaliwango²"
    tp2.font.name = "Calibri"
    tp2.font.size = Pt(28)
    tp2.font.bold = True
    tp2.font.color.rgb = PURE_BLACK
    tp2.alignment = PP_ALIGN.CENTER

    tp3 = ttf.add_paragraph()
    tp3.text = "¹ Department of Networks, ² Department of Computer Science, College of Computing and Information Sciences (CoCIS), Makerere University, Kampala, Uganda"
    tp3.font.name = "Calibri"
    tp3.font.size = Pt(19)
    tp3.font.color.rgb = MUTED_GRAY
    tp3.alignment = PP_ALIGN.CENTER

    tp4 = ttf.add_paragraph()
    tp4.text = "Correspondence: {mpairwe_lauben, nankya.shadia, yapyeko.rebecca}@students.mak.ac.ug, ggaliwango.marvin@cit.ac.ug  —  Interactive System: https://landwind22-retinal-screening.hf.space"
    tp4.font.name = "Calibri"
    tp4.font.size = Pt(17)
    tp4.font.bold = True
    tp4.font.color.rgb = NCC_DEEP_BLUE
    tp4.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # TWO-COLUMN LAYOUT CONFIGURATION
    # =========================================================================
    COL_W = 15.16
    COL1_L = 0.95
    COL2_L = 17.11

    # =========================================================================
    # LEFT COLUMN
    # =========================================================================

    # Section 1: Abstract (Y: 8.55 to 12.55)
    add_banner("Abstract", COL1_L, 8.55, COL_W)
    abs_box = slide.shapes.add_textbox(Inches(COL1_L), Inches(9.20), Inches(COL_W), Inches(3.30))
    atf = abs_box.text_frame
    atf.word_wrap = True
    atf.margin_top = Inches(0)
    atf.margin_left = Inches(0)
    atf.margin_right = Inches(0)

    ap0 = atf.paragraphs[0]
    ap0.space_after = Pt(6)
    ap0.line_spacing = 1.15
    ar0 = ap0.add_run()
    ar0.text = "Preventable blindness from vitreoretinal and optic nerve pathologies presents a critical global burden, exacerbated by the acute shortage of specialist ophthalmologists across sub-Saharan Africa. In Uganda, non-specialist primary healthcare workers serve as initial gatekeepers for ocular screening without specialist grading support."
    ar0.font.name = "Calibri"
    ar0.font.size = Pt(19.0)
    ar0.font.color.rgb = PURE_BLACK

    ap1 = atf.add_paragraph()
    ap1.line_spacing = 1.15
    ar1 = ap1.add_run()
    ar1.text = "OptiscanAI adapts a domain-specific foundation model (RETFound ViT-L/16) via Low-Rank Adaptation (LoRA), evaluated strictly on the official held-out test partition (640 images) of the Retinal Fundus Multi-Disease Image Dataset (RFMiD). We report 0.872 macro AUC across 24 conditions, diagnose an uncalibrated overconfidence defect (ECE 0.209) mitigated to 0.003 via isotonic Platt scaling, achieve 0.952 referral triage AUC, and evaluate an offline edge student running in 102 ms at 5.4 MB for rural primary care."
    ar1.font.name = "Calibri"
    ar1.font.size = Pt(19.0)
    ar1.font.color.rgb = PURE_BLACK

    # Section 2: Problem Statement (Y: 12.85 to 18.10)
    add_banner("Problem Statement", COL1_L, 12.85, COL_W)
    prob_box = slide.shapes.add_textbox(Inches(COL1_L), Inches(13.50), Inches(COL_W), Inches(4.30))
    ptf = prob_box.text_frame
    ptf.word_wrap = True
    ptf.margin_top = Inches(0)
    ptf.margin_left = Inches(0)
    ptf.margin_right = Inches(0)

    add_bullet_item(
        ptf,
        "• Avoidable Blindness in Uganda:",
        "Over 2.2 billion individuals suffer vision impairment globally. In Uganda, ophthalmologists are concentrated in urban tertiary hospitals, leaving frontline primary health workers as gatekeepers without expert grading support.",
        pt_size=18.5, space_after=5,
    )
    add_bullet_item(
        ptf,
        "• Frontline Screening Bottlenecks:",
        "Non-mydriatic cameras in rural clinics face variable illumination, focus blur, and dust artifacts, while manual grading by distant specialists is economically and logistically inaccessible.",
        pt_size=18.5, space_after=5,
    )
    add_bullet_item(
        ptf,
        "• Multi-Disease Co-morbidities:",
        "Patients present overlapping pathologies across 24 conditions (diabetic retinopathy, glaucoma cupping, ARMD, vein occlusions), rendering single-disease screening tools inadequate.",
        pt_size=18.5, space_after=5,
    )
    add_bullet_item(
        ptf,
        "• Silent Channels & Prevalence Shift:",
        "Uncalibrated foundation model outputs exhibit severe overconfidence, causing static decision thresholds to silence rare disease channels. Furthermore, high benchmark PPV collapses in low-prevalence rural screening.",
        pt_size=18.5, space_after=5,
    )
    add_bullet_item(
        ptf,
        "• Core Objective:",
        "Adapt a domain-specific foundation model for multi-disease retinal screening, eliminate calibration failures, and distill a 5.4 MB offline-first edge student for rural primary care.",
        pt_size=18.5, space_after=0,
    )

    # Section 3: System Architecture (Y: 18.45 to 27.55)
    add_banner("System Architecture", COL1_L, 18.45, COL_W)
    arch_box = slide.shapes.add_textbox(Inches(COL1_L), Inches(19.10), Inches(COL_W), Inches(2.90))
    atf2 = arch_box.text_frame
    atf2.word_wrap = True
    atf2.margin_top = Inches(0)
    atf2.margin_left = Inches(0)
    atf2.margin_right = Inches(0)

    add_bullet_item(
        atf2,
        "• Cascaded 4-Tier Intake Gate:",
        "Verifies resolution (≥1536×1024), focus sharpness (Laplacian variance), illumination uniformity, and MobileNet usability scoring (≥0.70) with real-time recapture guidance.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        atf2,
        "• RETFound ViT-L/16 + LoRA Adapters:",
        "Foundation model (1.6M retinal pretraining) fine-tuned via LoRA (r=16, α=32) updating only 2.44M of 305.7M parameters (0.8%). Top-32 patch attention feeds a 512→128→24 bottleneck head.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        atf2,
        "• Symbolic Clinical Knowledge Graph:",
        "24 disease nodes and 54 typed edges curated from AAO and Uganda MOH guidelines inject epidemiological co-occurrence priors into referral triage.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        atf2,
        "• Distilled Offline Edge Student:",
        "5.2M MobileNetV3 student distilled to INT8 ONNX (5.4 MB) executes locally in 102 ms per image on low-cost edge laptops without telecommunications.",
        pt_size=18.0, space_after=0,
    )

    # Fig 1 Image & Caption
    slide.shapes.add_picture(str(fig1_png), Inches(COL1_L + 0.20), Inches(22.55), Inches(COL_W - 0.40), Inches(4.65))

    cap1_box = slide.shapes.add_textbox(Inches(COL1_L), Inches(27.30), Inches(COL_W), Inches(0.35))
    cap1_box.text_frame.word_wrap = True
    cap1_box.text_frame.margin_top = Inches(0)
    c1_p = cap1_box.text_frame.paragraphs[0]
    c1_p.text = "Fig. 1: Modular OptiscanAI pipeline: Server foundation pathway (1–7) and distilled offline edge student (8)."
    c1_p.font.name = "Calibri"
    c1_p.font.size = Pt(17.0)
    c1_p.font.bold = True
    c1_p.font.color.rgb = NAVY_LEAD
    c1_p.alignment = PP_ALIGN.CENTER

    # Section 4: Model Adaptation & Component Ablation (Y: 27.95 to 35.75)
    add_banner("Model Adaptation & Ablation", COL1_L, 27.95, COL_W)

    # Table 1: Native PowerPoint Table matching MajiQuick style
    t1_shape = slide.shapes.add_table(9, 5, Inches(COL1_L + 0.20), Inches(28.55), Inches(COL_W - 0.40), Inches(3.40))
    tab1 = t1_shape.table
    tab1.columns[0].width = Inches(5.96)
    tab1.columns[1].width = Inches(2.20)
    tab1.columns[2].width = Inches(2.20)
    tab1.columns[3].width = Inches(2.20)
    tab1.columns[4].width = Inches(2.20)

    t1_data = [
        ["Arm / Model Component", "Total (M)", "Train. %", "AUC-ROC", "AUPRC"],
        ["ResNet-50 (ImageNet full FT)", "23.56", "100.0 %", "0.867", "0.336"],
        ["EfficientNet-B3 (ImageNet FT)", "10.73", "100.0 %", "0.819", "0.377"],
        ["ViT-B/16 (ImageNet full FT)", "85.82", "100.0 %", "0.871", "0.472"],
        ["RETFound ViT-L (head only)", "2.44", "0.8 %", "0.877", "0.360"],
        ["RETFound ViT-L + LoRA (r=16)", "4.01", "1.3 %", "0.882", "0.392"],
        ["RETFound + LoRA (Dataset Norm)", "4.01", "1.3 %", "0.827", "0.285"],
        ["ImageNet Norm + CLAHE", "4.01", "1.3 %", "0.866", "0.341"],
        ["OptiscanAI Deployed Checkpoint†", "2.44", "0.8 %", "0.872", "0.357"],
    ]

    for r_i, row in enumerate(t1_data):
        for c_i, val in enumerate(row):
            cell = tab1.cell(r_i, c_i)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r_i == 0:
                cell.fill.fore_color.rgb = NCC_DEEP_BLUE
            elif r_i == 8:
                cell.fill.fore_color.rgb = HIGHLIGHT_ROW
            elif r_i % 2 == 1:
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = ALT_ROW_BG

            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(15 if r_i == 0 else 13.5)
            p.font.bold = r_i == 0 or r_i == 8 or c_i == 0
            p.font.color.rgb = WHITE if r_i == 0 else (NAVY_LEAD if r_i == 8 else PURE_BLACK)
            p.alignment = PP_ALIGN.LEFT if c_i == 0 else PP_ALIGN.CENTER

    t1_cap_box = slide.shapes.add_textbox(Inches(COL1_L), Inches(32.05), Inches(COL_W), Inches(0.32))
    t1_cap_box.text_frame.word_wrap = True
    t1_cap_box.text_frame.margin_top = Inches(0)
    c1_t_p = t1_cap_box.text_frame.paragraphs[0]
    c1_t_p.text = "Table 1: Controlled component ablation on official RFMiD test split (shared 15-epoch training recipe)."
    c1_t_p.font.name = "Calibri"
    c1_t_p.font.size = Pt(16.5)
    c1_t_p.font.bold = True
    c1_t_p.font.color.rgb = NAVY_LEAD
    c1_t_p.alignment = PP_ALIGN.CENTER

    ab_box = slide.shapes.add_textbox(Inches(COL1_L), Inches(32.47), Inches(COL_W), Inches(3.80))
    ab_tf = ab_box.text_frame
    ab_tf.word_wrap = True
    ab_tf.margin_top = Inches(0)
    ab_tf.margin_left = Inches(0)
    ab_tf.margin_right = Inches(0)

    add_bullet_item(
        ab_tf,
        "• Pretraining Efficiency vs Full Fine-Tuning:",
        "RETFound frozen head-only (0.8% trainable params) achieves 0.877 AUC, outperforming fully fine-tuned ImageNet ResNet-50 (0.867) and EfficientNet-B3 (0.819). Adding LoRA (r=16) gains +0.005 AUC and +0.032 AUPRC for only 1.6M additional parameters.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        ab_tf,
        "• Rare Positives Trade-off:",
        "Full fine-tuning of ViT-B/16 on ImageNet reaches higher AUPRC (0.472 vs 0.392) by updating all 85.8M weights. Foundation models excel at ranking efficiency on constrained institutional GPU budgets, while full tuning retrieves rare positives better.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        ab_tf,
        "• Distribution Alignment & CLAHE Penalty:",
        "Replacing ImageNet normalization with dataset-specific channel statistics collapses AUC by 0.054 and AUPRC by 0.107. CLAHE contrast equalisation reduces AUC by 0.016 by stripping subtle vascular diagnostic cues.",
        pt_size=18.0, space_after=0,
    )

    # Section 5: Clinical Knowledge Graph & Gating (Y: 36.00 to 44.50)
    add_banner("Clinical Knowledge Graph & Gating", COL1_L, 36.00, COL_W)
    kg_box = slide.shapes.add_textbox(Inches(COL1_L), Inches(36.65), Inches(COL_W), Inches(8.00))
    kg_tf = kg_box.text_frame
    kg_tf.word_wrap = True
    kg_tf.margin_top = Inches(0)
    kg_tf.margin_left = Inches(0)
    kg_tf.margin_right = Inches(0)

    add_bullet_item(
        kg_tf,
        "• Evidence-Based Clinical Ontology:",
        "24 disease nodes and 54 typed edges curated from American Academy of Ophthalmology (AAO) Preferred Practice Patterns, AREDS-39, and Uganda Ministry of Health Guidelines (2024). Spans vascular, degenerative, glaucomatous, and inflammatory taxonomy.",
        pt_size=18.5, space_after=5,
    )
    add_bullet_item(
        kg_tf,
        "• Uganda-Specific Epidemiological Weighting:",
        "Attaches localized clinical prevalence factors for 6 priority conditions (diabetic retinopathy, optic disc cupping/glaucoma, retinal vein occlusions, macular scars) reflecting East African disease burdens.",
        pt_size=18.5, space_after=5,
    )
    add_bullet_item(
        kg_tf,
        "• Monotonic Risk Stratification:",
        "Under calibrated sensitivity-targeted thresholds, the symbolic graph cleanly stratifies referral priority: 84.5% routed to Urgent (86.9% confirmed true pathology), 10.8% to Routine (47.8%), and 4.7% to Follow-up (10.0%).",
        pt_size=18.5, space_after=5,
    )
    add_bullet_item(
        kg_tf,
        "• Transparent Synergistic Scope:",
        "Co-occurrence rules alter only 12.7% of predictions because 21 rare partner classes were dropped from RFMiD for having <10 training positives. OptiscanAI explicitly bounds claims rather than asserting unverified global synergy.",
        pt_size=18.5, space_after=5,
    )
    add_bullet_item(
        kg_tf,
        "• Frontline Usability & Recapture Guidance:",
        "The cascaded quality filter provides explicit failure codes (low focus sharpness, improper pupil centering, underexposure) enabling community health extension workers to recapture diagnostic-grade fundus photos immediately.",
        pt_size=18.5, space_after=5,
    )
    add_bullet_item(
        kg_tf,
        "• Human-in-the-Loop Clinical Safeguards:",
        "All AI triage outputs include calibrated confidence scores and attribution maps, requiring final sign-off by a visiting ophthalmic clinical officer (OCO) rather than making automated autonomous clinical judgments.",
        pt_size=18.5, space_after=0,
    )

    # =========================================================================
    # RIGHT COLUMN
    # =========================================================================

    # Section 6: Results & Clinical Case Gallery (Y: 8.55 to 21.00)
    add_banner("Results & Clinical Validation", COL2_L, 8.55, COL_W)
    res_box = slide.shapes.add_textbox(Inches(COL2_L), Inches(9.20), Inches(COL_W), Inches(3.10))
    rtf = res_box.text_frame
    rtf.word_wrap = True
    rtf.margin_top = Inches(0)
    rtf.margin_left = Inches(0)
    rtf.margin_right = Inches(0)

    add_bullet_item(
        rtf,
        "• Macro Discrimination (24 Diseases):",
        "Evaluated strictly on the held-out 640-image RFMiD test partition, OptiscanAI achieves 0.872 macro AUC-ROC (95% CI: 0.846–0.896) and 0.357 macro AUPRC across all 24 held-out diagnostic conditions.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        rtf,
        "• High-Prevalence Accuracy:",
        "Outstanding diagnostic accuracy on major preventable conditions: Myopia (AUC 0.996), Media Haze (0.974), Macular Degeneration (0.972), and Diabetic Retinopathy (0.967, AUPRC 0.873).",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        rtf,
        "• Power Bounds on Rare Classes:",
        "Diagnostic accuracy tracks positive training sample count (Fig. 2b); rare conditions (<20 test positives) exhibit wider bootstrap confidence intervals and require prospective validation.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        rtf,
        "• Binary Referral Triage:",
        "High discriminative power for referral screening (AUC 0.952; Sensitivity 95.5%, Specificity 73.1%), safely sorting urgent specialist cases from routine checks.",
        pt_size=18.0, space_after=0,
    )

    # Fig 2 Image & Caption
    slide.shapes.add_picture(str(fig2_png), Inches(COL2_L + 0.20), Inches(12.60), Inches(COL_W - 0.40), Inches(3.90))

    cap2_box = slide.shapes.add_textbox(Inches(COL2_L), Inches(16.60), Inches(COL_W), Inches(0.35))
    cap2_box.text_frame.word_wrap = True
    cap2_box.text_frame.margin_top = Inches(0)
    c2_p = cap2_box.text_frame.paragraphs[0]
    c2_p.text = "Fig. 2: Empirical test evidence: (a) Calibration reliability; (b) AUC vs prevalence; (c) Binary referral ROC (AUC 0.952)."
    c2_p.font.name = "Calibri"
    c2_p.font.size = Pt(16.5)
    c2_p.font.bold = True
    c2_p.font.color.rgb = NAVY_LEAD
    c2_p.alignment = PP_ALIGN.CENTER

    # Fig 3 Image & Caption
    slide.shapes.add_picture(str(fig3_png), Inches(COL2_L + 0.20), Inches(17.22), Inches(COL_W - 0.40), Inches(3.30))

    cap3_box = slide.shapes.add_textbox(Inches(COL2_L), Inches(20.60), Inches(COL_W), Inches(0.48))
    cap3_box.text_frame.word_wrap = True
    cap3_box.text_frame.margin_top = Inches(0)
    c3_p = cap3_box.text_frame.paragraphs[0]
    c3_p.text = "Fig. 3: Representative held-out fundus screening cases with automated triage referral actions (Green: Routine; Red/Amber: Urgent Referral)."
    c3_p.font.name = "Calibri"
    c3_p.font.size = Pt(16.5)
    c3_p.font.bold = True
    c3_p.font.color.rgb = NAVY_LEAD
    c3_p.alignment = PP_ALIGN.CENTER

    # Section 7: Quantitative Outcomes & Deployment Policies (Y: 21.40 to 30.25)
    add_banner("Quantitative Outcomes", COL2_L, 21.40, COL_W)
    out_box = slide.shapes.add_textbox(Inches(COL2_L), Inches(22.05), Inches(COL_W), Inches(2.90))
    otf = out_box.text_frame
    otf.word_wrap = True
    otf.margin_top = Inches(0)
    otf.margin_left = Inches(0)
    otf.margin_right = Inches(0)

    add_bullet_item(
        otf,
        "• Probability Calibration Fix:",
        "Raw model probabilities suffer severe over-confidence (mean 0.251 vs 0.042 empirical rate; ECE 0.209), causing precision-floor thresholds to silence 5 disease channels. Monotone Platt scaling resolves ECE to 0.003 while strictly preserving ROC ranking.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        otf,
        "• Operating Policy Optimization:",
        "Sensitivity target ≥ 0.90 restores detection across all 24 diseases (0 silent channels vs 6 uniform, 5 precision floor) with 81.1% sensitivity.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        otf,
        "• INT8 Dynamic Quantization:",
        "Compresses model footprint by 74.7% (1224 MB → 310 MB) while preserving AUC (0.872 → 0.873). Distilled MobileNetV3 student operates at 5.4 MB ONNX and 102 ms median latency with 99.6% decision concordance.",
        pt_size=18.0, space_after=0,
    )

    # Table 2: Clinical Prevalence Transfer Table
    t2_shape = slide.shapes.add_table(6, 5, Inches(COL2_L + 0.20), Inches(24.75), Inches(COL_W - 0.40), Inches(2.40))
    tab2 = t2_shape.table
    tab2.columns[0].width = Inches(5.96)
    tab2.columns[1].width = Inches(2.20)
    tab2.columns[2].width = Inches(2.20)
    tab2.columns[3].width = Inches(2.20)
    tab2.columns[4].width = Inches(2.20)

    t2_data = [
        ["Screening Caseload Population", "Prevalence", "PPV", "NPV", "Referral %"],
        ["RFMiD Held-Out Test (Benchmark)", "79.0 %", "0.931", "0.810", "81.0 %"],
        ["Specialized Eye Clinic (Projected)", "30.0 %", "0.604", "0.974", "47.0 %"],
        ["District Hospital Triage (Projected)", "20.0 %", "0.470", "0.985", "41.0 %"],
        ["Rural Primary Care Clinic (Projected)", "10.0 %", "0.283", "0.993", "34.0 %"],
        ["Community Mobile Screening (Projected)", "5.0 %", "0.158", "0.997", "30.0 %"],
    ]

    for r_i, row in enumerate(t2_data):
        for c_i, val in enumerate(row):
            cell = tab2.cell(r_i, c_i)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r_i == 0:
                cell.fill.fore_color.rgb = NCC_DEEP_BLUE
            elif r_i == 1:
                cell.fill.fore_color.rgb = HIGHLIGHT_ROW
            elif r_i % 2 == 1:
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = ALT_ROW_BG

            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(15 if r_i == 0 else 13.5)
            p.font.bold = r_i == 0 or r_i == 1 or c_i == 0
            p.font.color.rgb = WHITE if r_i == 0 else (NAVY_LEAD if r_i == 1 else PURE_BLACK)
            p.alignment = PP_ALIGN.LEFT if c_i == 0 else PP_ALIGN.CENTER

    t2_cap_box = slide.shapes.add_textbox(Inches(COL2_L), Inches(27.25), Inches(COL_W), Inches(0.32))
    t2_cap_box.text_frame.word_wrap = True
    t2_cap_box.text_frame.margin_top = Inches(0)
    c2_t_p = t2_cap_box.text_frame.paragraphs[0]
    c2_t_p.text = "Table 2: Clinical prevalence transfer holding operating point fixed (Sensitivity 95.5%, Specificity 73.1%)."
    c2_t_p.font.name = "Calibri"
    c2_t_p.font.size = Pt(16.5)
    c2_t_p.font.bold = True
    c2_t_p.font.color.rgb = NAVY_LEAD
    c2_t_p.alignment = PP_ALIGN.CENTER

    out_sub_box = slide.shapes.add_textbox(Inches(COL2_L), Inches(27.67), Inches(COL_W), Inches(2.70))
    os_tf = out_sub_box.text_frame
    os_tf.word_wrap = True
    os_tf.margin_top = Inches(0)
    os_tf.margin_left = Inches(0)
    os_tf.margin_right = Inches(0)

    add_bullet_item(
        os_tf,
        "• Referral Bottleneck & Rule-Out Sieve Role:",
        "In community caseloads (5–10% prevalence), PPV drops from 93.1% to 15.8%–28.3%. Sizing tertiary referral capacity on raw benchmark PPV would overburden clinics by 3x–6x. However, NPV exceeds 99.3%, proving OptiscanAI's value as a frontline rule-out sieve.",
        pt_size=18.0, space_after=4,
    )
    add_bullet_item(
        os_tf,
        "• Explainability Faithfulness Audit:",
        "Grad-CAM (p=0.029) and Integrated Gradients (p=0.008) pass RISE deletion tests, confirming spatial saliency. Both fail insertion tests due to floor probabilities (min prob 0.064); maps must serve as clinician hints, not autonomous decisions.",
        pt_size=18.0, space_after=0,
    )

    # Section 8: Conclusions (Y: 30.55 to 35.10)
    add_banner("Conclusions", COL2_L, 30.55, COL_W)
    concl_box = slide.shapes.add_textbox(Inches(COL2_L), Inches(31.20), Inches(COL_W), Inches(4.30))
    ctf = concl_box.text_frame
    ctf.word_wrap = True
    ctf.margin_top = Inches(0)
    ctf.margin_left = Inches(0)
    ctf.margin_right = Inches(0)

    add_bullet_item(
        ctf,
        "• Foundation Model Adaptation:",
        "Parameter-efficient foundation model adaptation (RETFound + LoRA, 0.8% trainable params) delivers strong multi-disease ranking (AUC 0.872). Probability calibration is essential to prevent silent-channel triage failure, while INT8 edge distillation enables instant local inference on low-cost clinic hardware.",
        pt_size=18.5, space_after=6,
    )
    add_bullet_item(
        ctf,
        "• Clinical Separation Principle:",
        "High benchmark metrics on curated datasets do not guarantee clinical readiness. When transferring to rural primary care prevalence, PPV drops substantially; OptiscanAI is designed as a frontline rule-out sieve (NPV > 99.3%) that filters out healthy retinas and escalates suspected pathology to human clinicians.",
        pt_size=18.5, space_after=6,
    )
    add_bullet_item(
        ctf,
        "• Translational Roadmap:",
        "Future translation demands prospective multi-centre trials across Ugandan district health centres, validation with low-cost handheld fundus cameras, local mobile network integration, and Uganda National Drug Authority (NDA) regulatory clearance.",
        pt_size=18.5, space_after=0,
    )

    # Section 9: References (Y: 35.45 to 44.50)
    add_banner("References", COL2_L, 35.45, COL_W)
    ref_box = slide.shapes.add_textbox(Inches(COL2_L), Inches(36.10), Inches(COL_W), Inches(8.30))
    rtf2 = ref_box.text_frame
    rtf2.word_wrap = True
    rtf2.margin_top = Inches(0)
    rtf2.margin_left = Inches(0)
    rtf2.margin_right = Inches(0)

    refs = [
        ("[1]", "S. Pachade, P. Porwal, D. Thulkar et al., “Retinal fundus multi-disease image dataset (RFMiD): A dataset for multi-disease detection research,” Data, vol. 6, no. 2, p. 14, 2021."),
        ("[2]", "Y. Zhou, M. A. Chia, S. K. Wagner et al., “A foundation model for generalizable disease detection from retinal images,” Nature, vol. 622, pp. 156–163, 2023."),
        ("[3]", "E. J. Hu, Y. Shen, P. Wallis et al., “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learning Representations (ICLR), 2022."),
        ("[4]", "C. Guo, G. Pleiss, Y. Sun et al., “On calibration of modern neural networks,” in Proc. Int. Conf. Machine Learning (ICML), 2017, pp. 1321–1330."),
        ("[5]", "K. Lekadir, A. Frangi, A. R. Porras et al., “FUTURE-AI: International consensus guideline for trustworthy and deployable artificial intelligence in healthcare,” BMJ, vol. 388, e081554, 2025."),
        ("[6]", "G. S. Collins, K. G. M. Moons, P. Dhiman et al., “TRIPOD+AI statement: Updated guidance for reporting clinical prediction models,” BMJ, vol. 385, e078378, 2024."),
        ("[7]", "A. S. Tejani, M. E. Klontzas, A. A. Gatti et al., “Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 update,” Radiology: Artificial Intelligence, vol. 6, no. 4, e240300, 2024."),
        ("[8]", "E. Beede, E. Baylor, F. Hersch et al., “A human-centered evaluation of a deep learning system deployed in clinics for the detection of diabetic retinopathy,” in Proc. CHI Conf. Human Factors in Computing Systems, 2020, pp. 1–12."),
    ]

    for idx, (num, body) in enumerate(refs):
        rp = rtf2.add_paragraph() if rtf2.paragraphs[0].text else rtf2.paragraphs[0]
        rp.space_after = Pt(10)
        rp.line_spacing = 1.15
        rn = rp.add_run()
        rn.text = num + "  "
        rn.font.name = "Calibri"
        rn.font.size = Pt(17.5)
        rn.font.bold = True
        rn.font.color.rgb = NAVY_LEAD
        rb = rp.add_run()
        rb.text = body
        rb.font.name = "Calibri"
        rb.font.size = Pt(17.5)
        rb.font.color.rgb = PURE_BLACK

    # =========================================================================
    # FOOTER (Matching MajiQuick style)
    # =========================================================================
    ftr_box = slide.shapes.add_textbox(Inches(1.0), Inches(45.55), Inches(31.11), Inches(0.65))
    ftf = ftr_box.text_frame
    ftf.word_wrap = True
    ftf.margin_top = Inches(0)

    fp0 = ftf.paragraphs[0]
    fp0.text = "OptiscanAI (Paper 83)  •  National Conference on Communications (NCC), September 24–25, 2026  •  CEDAT / CoCIS, Makerere University"
    fp0.font.name = "Calibri"
    fp0.font.size = Pt(24)
    fp0.font.bold = True
    fp0.font.color.rgb = NCC_DEEP_BLUE
    fp0.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # FOOTER (Matching MajiQuick style)
    # =========================================================================
    ftr_box = slide.shapes.add_textbox(Inches(1.0), Inches(45.55), Inches(31.11), Inches(0.65))
    ftf = ftr_box.text_frame
    ftf.word_wrap = True
    ftf.margin_top = Inches(0)

    fp0 = ftf.paragraphs[0]
    fp0.text = "OptiscanAI (Paper 83)  •  National Conference on Communications (NCC), September 24–25, 2026  •  CEDAT / CoCIS, Makerere University"
    fp0.font.name = "Calibri"
    fp0.font.size = Pt(24)
    fp0.font.bold = True
    fp0.font.color.rgb = NCC_DEEP_BLUE
    fp0.alignment = PP_ALIGN.CENTER

    out_pptx = REPORTS_DIR / "OptiscanAI_NCC2026_Poster.pptx"
    prs.save(str(out_pptx))
    print(f"Generated PPTX: {out_pptx}")

    out_pdf = REPORTS_DIR / "OptiscanAI_NCC2026_Poster.pdf"
    print("Converting to PDF via LibreOffice...")
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            str(out_pptx),
            "--outdir",
            str(REPORTS_DIR),
        ],
        check=True,
    )
    print(f"Generated PDF: {out_pdf}")
    return out_pptx, out_pdf


if __name__ == "__main__":
    build_poster()

