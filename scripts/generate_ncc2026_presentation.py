#!/usr/bin/env python3
"""Generate the official OptiscanAI NCC 2026 conference presentation (PPTX and PDF).

Uses the visual layout and slide master of:
    docs/Reports/NCC2026_Belinda_1DCNN_EdgeForecasting.pptx

Style:
    - Steve Jobs style: Bold visual focus, dramatic storytelling, clean cards, rule of three.
    - Dr. Jordan Peterson touch/tone: Deep moral seriousness regarding avoidable blindness,
      intellectual rigor, confronting model calibration failure and prevalence collapse head-on.
    - Presenter: Mpairwe Lauben (Dept. of Networks, Makerere University).

Generates:
    docs/Reports/OptiscanAI_NCC2026_Presentation.pptx
    docs/Reports/OptiscanAI_NCC2026_Presentation.pdf
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pptx
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO / "docs/Reports"
TMP_DIR = Path("/mnt/nas1/tmp/opencode")
TEMPLATE_PPTX = REPORTS_DIR / "NCC2026_Belinda_1DCNN_EdgeForecasting.pptx"


def build_presentation() -> tuple[Path, Path]:
    prs = pptx.Presentation(str(TEMPLATE_PPTX))

    # Clear placeholder slides from template while keeping the slide master and layouts
    for i in range(len(prs.slides) - 1, -1, -1):
        rId = prs.slides._sldIdLst[i].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[i]

    # Typography & Color Palette
    FONT_HEAD = "Gill Sans MT"
    FONT_BODY = "Gill Sans MT"

    ORANGE = RGBColor(0xFF, 0x79, 0x00)       # Template Brand Orange (#FF7900)
    DARK_ORANGE = RGBColor(0xD9, 0x5D, 0x00)  # Accent Dark Orange
    NAVY = RGBColor(0x18, 0x40, 0x66)         # Scholarly Deep Navy (#184066)
    BLUE_ACCENT = RGBColor(0x00, 0x6F, 0xC2)  # Template Presenter Blue (#006FC2)
    DARK = RGBColor(0x1F, 0x24, 0x2E)         # Near Black for High Contrast (#1F242E)
    MUTED = RGBColor(0x55, 0x5E, 0x6D)        # Subtle Charcoal (#555E6D)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    RED = RGBColor(0xC5, 0x22, 0x1F)          # Warning / Pathology Red
    GREEN = RGBColor(0x05, 0x82, 0x48)        # Triage Success Green
    CARD_BG = RGBColor(0xFA, 0xFA, 0xFA)      # Soft Off-White
    CARD_BORDER = RGBColor(0xE2, 0xE8, 0xF0)  # Clean Subtle Gray Border

    # Helper: Slide Header + Circular Page Badge
    def add_slide_header(slide, title_text: str, slide_num: str, title_sz: float = 27.0):
        # Title Box
        tb = slide.shapes.add_textbox(Inches(0.59), Inches(0.20), Inches(8.40), Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0)
        tf.margin_left = Inches(0)
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = FONT_HEAD
        p.font.size = Pt(title_sz)
        p.font.bold = True
        p.font.color.rgb = ORANGE

        # Top-Right Slide Number Badge (Wider oval to prevent 2-digit vertical wrapping)
        badge = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(9.15), Inches(0.26), Inches(0.56), Inches(0.48)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = ORANGE
        badge.line.fill.background()
        btf = badge.text_frame
        btf.word_wrap = False
        btf.margin_top = Inches(0.04)
        btf.margin_left = Inches(0)
        btf.margin_right = Inches(0)
        bp = btf.paragraphs[0]
        bp.text = slide_num
        bp.font.name = FONT_HEAD
        bp.font.size = Pt(13)
        bp.font.bold = True
        bp.font.color.rgb = WHITE
        bp.alignment = PP_ALIGN.CENTER

    # Helper: Stat Card with vertical accent stripe (Belinda's signature pattern)
    def add_stat_card(slide, left: float, top: float, width: float, height: float,
                      stat_val: str, label: str, desc: str = "", accent_col=ORANGE, bg=CARD_BG):
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(left), Inches(top), Inches(width), Inches(height)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = bg
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.0)

        # Vertical left stripe
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(left), Inches(top), Inches(0.08), Inches(height)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_col
        bar.line.fill.background()

        tb = slide.shapes.add_textbox(
            Inches(left + 0.16), Inches(top + 0.06), Inches(width - 0.22), Inches(height - 0.12)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0)
        tf.margin_left = Inches(0)
        tf.margin_right = Inches(0)

        p0 = tf.paragraphs[0]
        p0.text = stat_val
        p0.font.name = FONT_HEAD
        p0.font.size = Pt(21)
        p0.font.bold = True
        p0.font.color.rgb = accent_col

        p1 = tf.add_paragraph()
        p1.text = label
        p1.font.name = FONT_HEAD
        p1.font.size = Pt(12.5)
        p1.font.bold = True
        p1.font.color.rgb = DARK

        if desc:
            p2 = tf.add_paragraph()
            p2.text = desc
            p2.font.name = FONT_BODY
            p2.font.size = Pt(10.5)
            p2.font.color.rgb = MUTED
        return card

    # Helper: Bullets text box
    def add_bullet_list(slide, items: list[tuple[str, str]], left: float, top: float, width: float, height: float, pt_size=14.5, space_after=5.0):
        tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0)
        tf.margin_left = Inches(0)
        tf.margin_right = Inches(0)

        for i, (lead, body) in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(space_after)
            p.line_spacing = 1.15

            r0 = p.add_run()
            r0.text = "• "
            r0.font.name = FONT_BODY
            r0.font.size = Pt(pt_size)
            r0.font.bold = True
            r0.font.color.rgb = ORANGE

            r1 = p.add_run()
            r1.text = lead + " "
            r1.font.name = FONT_HEAD
            r1.font.size = Pt(pt_size)
            r1.font.bold = True
            r1.font.color.rgb = DARK

            r2 = p.add_run()
            r2.text = body
            r2.font.name = FONT_BODY
            r2.font.size = Pt(pt_size)
            r2.font.color.rgb = MUTED
        return tb

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    slide1 = prs.slides.add_slide(prs.slide_layouts[0])

    tb_c = slide1.shapes.add_textbox(Inches(0.59), Inches(0.40), Inches(8.80), Inches(1.30))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = "THE 10TH NATIONAL CONFERENCE ON COMMUNICATIONS (NCC 2026)"
    p_c.font.name = FONT_HEAD
    p_c.font.size = Pt(24)
    p_c.font.bold = True
    p_c.font.color.rgb = ORANGE
    p_c.alignment = PP_ALIGN.CENTER

    tb_t = slide1.shapes.add_textbox(Inches(0.40), Inches(1.85), Inches(9.20), Inches(1.70))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    p_t0 = tf_t.paragraphs[0]
    p_t0.text = "OptiscanAI: Large Vision Models for Multi-Disease Retinal Screening"
    p_t0.font.name = FONT_HEAD
    p_t0.font.size = Pt(26)
    p_t0.font.bold = True
    p_t0.font.color.rgb = DARK
    p_t0.alignment = PP_ALIGN.CENTER

    p_t1 = tf_t.add_paragraph()
    p_t1.text = "A Benchmark and Deployment Study in Ugandan Healthcare  •  Paper 83"
    p_t1.font.name = FONT_HEAD
    p_t1.font.size = Pt(18)
    p_t1.font.bold = True
    p_t1.font.color.rgb = NAVY
    p_t1.alignment = PP_ALIGN.CENTER

    p_t2 = tf_t.add_paragraph()
    p_t2.text = "Lauben Mpairwe¹, Shadia Nankya¹, Rebecca Yapyeko¹, Marvin Ggaliwango²"
    p_t2.font.name = FONT_BODY
    p_t2.font.size = Pt(14)
    p_t2.font.color.rgb = MUTED
    p_t2.alignment = PP_ALIGN.CENTER

    tb_p = slide1.shapes.add_textbox(Inches(1.50), Inches(4.00), Inches(7.00), Inches(1.10))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    pp0 = tf_p.paragraphs[0]
    pp0.text = "Presenter: Mpairwe Lauben"
    pp0.font.name = FONT_HEAD
    pp0.font.size = Pt(20)
    pp0.font.bold = True
    pp0.font.color.rgb = BLUE_ACCENT
    pp0.alignment = PP_ALIGN.CENTER

    pp1 = tf_p.add_paragraph()
    pp1.text = "Dept. of Networks, College of Computing and Information Sciences (CoCIS)\nMakerere University, Kampala, Uganda"
    pp1.font.name = FONT_BODY
    pp1.font.size = Pt(15)
    pp1.font.color.rgb = BLUE_ACCENT
    pp1.alignment = PP_ALIGN.CENTER

    tb_d = slide1.shapes.add_textbox(Inches(1.50), Inches(5.30), Inches(7.00), Inches(0.40))
    p_d = tb_d.text_frame.paragraphs[0]
    p_d.text = "September 24, 2026"
    p_d.font.name = FONT_HEAD
    p_d.font.size = Pt(16)
    p_d.font.bold = True
    p_d.font.color.rgb = DARK
    p_d.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: THE CORE THESIS (Three Uncomfortable Realities)
    # =========================================================================
    slide2 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide2, "Three Core Realities in Retinal AI", "02", title_sz=26.0)

    add_stat_card(slide2, 0.59, 1.25, 2.75, 3.85, "01", "The Clinical Tragedy",
                  "2.2 billion people suffer visual impairment globally. In Uganda, <45 specialists serve 48 million citizens. The doctors are in Kampala; the blindness is in rural districts.", ORANGE)
    add_stat_card(slide2, 3.62, 1.25, 2.75, 3.85, "02", "The Benchmark Illusion",
                  "A 93% PPV on an enriched international benchmark collapses to 16% in rural primary care. We cannot confuse academic leaderboards with clinical readiness.", NAVY)
    add_stat_card(slide2, 6.65, 1.25, 2.75, 3.85, "03", "The Silent Defect",
                  "Severe probability over-confidence (ECE 0.209) silently disabled 5 whole disease channels. We confront this failure head-on and prove the isotonic remedy.", RED)

    tk_box = slide2.shapes.add_textbox(Inches(0.75), Inches(5.30), Inches(8.50), Inches(0.80))
    tk_box.text_frame.word_wrap = True
    tk_p = tk_box.text_frame.paragraphs[0]
    tk_p.text = "“Before deploying AI into human healthcare, we have a moral obligation to state where it breaks down — and engineer the remedy before it reaches the clinic.”"
    tk_p.font.name = FONT_HEAD
    tk_p.font.size = Pt(13)
    tk_p.font.bold = True
    tk_p.font.color.rgb = DARK
    tk_p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 3: MOTIVATION & UGANDA HEALTHCARE CRISIS
    # =========================================================================
    slide3 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide3, "The Crisis: Avoidable Blindness in Uganda", "03", title_sz=27.0)

    add_stat_card(slide3, 0.59, 1.25, 4.25, 1.30, "2.2 Billion", "Global Vision Impairment (WHO)",
                  "Over 50% is entirely preventable if diagnosed early.", ORANGE)
    add_stat_card(slide3, 5.14, 1.25, 4.25, 1.30, "< 45 Specialists", "Ophthalmologists in Uganda",
                  "Serving 48M citizens — heavily concentrated in Mulago.", RED)

    add_bullet_list(slide3, [
        ("The Asymmetric Clinical Cost:",
         "Diabetic retinopathy, glaucoma, and macular degeneration inflict irreversible blindness. Missing an insidious lesion strips individuals of agency, dignity, and livelihood."),
        ("The Frontline Bottleneck:",
         "Primary nurses at Level III/IV health centres are frontline gatekeepers with zero ophthalmic diagnostic tools and minimal specialist training."),
        ("The Technological Dilemma:",
         "Cloud-tethered 1 GB transformers fail when rural power drops and networks vanish. Screening requires an offline, highly specific rule-out sieve."),
    ], 0.59, 2.80, 8.80, 2.90, pt_size=15.0, space_after=8.0)

    # =========================================================================
    # SLIDE 4: SYSTEM ARCHITECTURE (Modular Pipeline)
    # =========================================================================
    slide4 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide4, "OptiscanAI: System Architecture", "04", title_sz=28.0)

    fig1_path = TMP_DIR / "fig1_poster.png"
    if fig1_path.exists():
        slide4.shapes.add_picture(str(fig1_path), Inches(0.59), Inches(1.15), Inches(8.82), Inches(2.70))

    add_bullet_list(slide4, [
        ("(1) Quality Gate:", "Cascaded 4-tier intake filter (0.6s_stat + 0.4s_learned ≥ 0.70) rejects ungradable blurry photos."),
        ("(2–3) RETFound ViT-L + LoRA:", "Pretrained on 1.6M retinal images; LoRA (r=16) adapts only 0.8% of weights (2.44M params)."),
        ("(4–5) Calibration & Knowledge Graph:", "Isotonic recalibration + 54-edge clinical ontology encoding AAO & Uganda MoH guidelines."),
        ("(6–7) Referral Triage & Audit:", "Referral prioritization (Urgent vs Routine) with immutable cryptographic audit logging."),
    ], 0.59, 3.95, 8.82, 2.00, pt_size=13.0, space_after=3.0)

    # =========================================================================
    # SLIDE 5: OFFLINE EDGE DISTILLATION
    # =========================================================================
    slide5 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide5, "Offline Edge Distillation for Rural Clinics", "05", title_sz=27.0)

    add_stat_card(slide5, 0.59, 1.25, 2.75, 1.35, "5.4 MB", "Model Size (INT8 ONNX)",
                  "Compressed 74.7% from 20.8 MB FP32 student.", GREEN)
    add_stat_card(slide5, 3.62, 1.25, 2.75, 1.35, "102 ms", "Median CPU Latency",
                  "Real-time inference on low-cost edge hardware.", ORANGE)
    add_stat_card(slide5, 6.65, 1.25, 2.75, 1.35, "99.6 %", "Decision Concordance",
                  "Matches server FP32 decisions at τ = 0.50.", BLUE_ACCENT)

    add_bullet_list(slide5, [
        ("The Reality of the Rural Clinic:",
         "In Soroti, Gulu, or Karamoja, the electrical grid drops for days. Bandwidth to stream 15 MB uncompressed TIFF scans to cloud servers is pure fantasy."),
        ("Multi-Objective Knowledge Distillation:",
         "A compact MobileNetV3-Large student (5.2M parameters) learns from the 305.7M-parameter RETFound ViT-L teacher via combined soft logits and feature alignment."),
        ("Zero-Network Autonomous Operation:",
         "Triage executes 100% on-device. Encrypted audit records stage locally and sync opportunistically only when connectivity returns."),
    ], 0.59, 2.85, 8.80, 2.90, pt_size=14.5, space_after=8.0)

    # =========================================================================
    # SLIDE 6: EMPIRICAL BENCHMARK SETUP
    # =========================================================================
    slide6 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide6, "Rigorous Empirical Setup on RFMiD", "06", title_sz=27.0)

    add_stat_card(slide6, 0.59, 1.25, 2.05, 1.30, "640", "Held-Out Test", "Evaluated strictly post-convergence.", NAVY)
    add_stat_card(slide6, 2.84, 1.25, 2.05, 1.30, "24", "Retained Classes", "Excluded rare labels (n < 10).", ORANGE)
    add_stat_card(slide6, 5.09, 1.25, 2.05, 1.30, "1,000", "Bootstrap CIs", "95% percentile confidence bounds.", BLUE_ACCENT)
    add_stat_card(slide6, 7.34, 1.25, 2.05, 1.30, "CLAIM", "Reporting Standard", "Conforming to TRIPOD+AI 2024.", GREEN)

    add_bullet_list(slide6, [
        ("Strict Train / Val / Test Partition:",
         "1,920 train, 640 validation, and 640 evaluation images. All threshold selections and calibrations were locked strictly on validation data."),
        ("Input Normalisation Integrity:",
         "Retained ImageNet normalisation because the foundation backbone was pretrained in that feature space; switching costs 0.054 AUC."),
        ("No Post-Hoc Cherry-Picking:",
         "We report exact per-disease sensitivity, specificity, PPV, and false-negative rates with 95% bootstrap intervals across all 24 categories."),
    ], 0.59, 2.80, 8.80, 2.90, pt_size=15.0, space_after=8.0)

    # =========================================================================
    # SLIDE 7: MULTI-DISEASE PERFORMANCE
    # =========================================================================
    slide7 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide7, "Multi-Disease Performance on RFMiD Test", "07", title_sz=26.0)

    fig2_path = TMP_DIR / "fig2_poster.png"
    if fig2_path.exists():
        slide7.shapes.add_picture(str(fig2_path), Inches(0.59), Inches(1.15), Inches(8.82), Inches(2.75))

    add_bullet_list(slide7, [
        ("Macro Discrimination (0.872 AUC):",
         "Macro AUC reaches 0.872 (95% CI: 0.846–0.896) and Macro AUPRC reaches 0.357 across 24 multi-disease retinal categories."),
        ("High-Prevalence Excellence:",
         "Myopia (0.996 AUC), Media Haze (0.974), Age-Related Macular Degeneration (0.972), and Diabetic Retinopathy (0.967 AUC, 0.873 AUPRC)."),
        ("The Statistical Boundary (Petersonian Honesty):",
         "Rare conditions with <20 test positives collapse: Optic Disc Pallor (0.663 AUC), Macular Hole (0.674). We refuse to claim clinical readiness where statistical power is absent."),
    ], 0.59, 4.05, 8.82, 1.95, pt_size=13.0, space_after=3.0)

    # =========================================================================
    # SLIDE 8: THE SILENT CHANNEL DEFECT & CALIBRATION FIX
    # =========================================================================
    slide8 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide8, "The Silent Channel Defect & Remedy", "08", title_sz=26.0)

    add_stat_card(slide8, 0.59, 1.25, 2.75, 1.35, "0.209", "Raw ECE (Failure)",
                  "Mean prob 0.251 vs 0.042 base rate. Extreme over-confidence.", RED)
    add_stat_card(slide8, 3.62, 1.25, 2.75, 1.35, "5 Channels", "Silenced Entirely",
                  "Static thresholding set fallback τ=0.95: zero sensitivity.", RED)
    add_stat_card(slide8, 6.65, 1.25, 2.75, 1.35, "0.003", "Recalibrated ECE",
                  "Platt scaling restores reliable probabilities, 0 silent channels.", GREEN)

    add_bullet_list(slide8, [
        ("The Hidden Bug Nobody Measures:",
         "Asymmetric loss clipping (0.05) and label smoothing prevent the network from ever outputting confident zeros. Minimum probability over the entire test split was 0.064!"),
        ("The Fatal Interaction:",
         "Searching for precision ≥ 0.10 on rare classes forced thresholds to the 0.95 ceiling, permanently muting 5 diagnostic channels (AH, ODP, PT, EDN, MHL)."),
        ("The Monotone Solution:",
         "Per-class Platt recalibration resolves ECE to 0.003 while preserving AUC ranking. Adopting a sensitivity target ≥ 0.90 eliminates all silent channels and reduces misses by 59%."),
    ], 0.59, 2.85, 8.80, 2.90, pt_size=14.5, space_after=7.0)

    # =========================================================================
    # SLIDE 9: COMPONENT ABLATION
    # =========================================================================
    slide9 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide9, "Ablation: Foundation vs. ImageNet", "09", title_sz=25.5)

    t_shape = slide9.shapes.add_table(7, 5, Inches(0.59), Inches(1.20), Inches(8.82), Inches(2.60))
    tab = t_shape.table
    tab.columns[0].width = Inches(3.62)
    tab.columns[1].width = Inches(1.30)
    tab.columns[2].width = Inches(1.30)
    tab.columns[3].width = Inches(1.30)
    tab.columns[4].width = Inches(1.30)

    abl_data = [
        ["Model Architecture & Adaptation", "Params (M)", "Train. %", "AUC-ROC", "AUPRC"],
        ["ResNet-50 (ImageNet Full FT)", "23.56", "100.0 %", "0.867", "0.336"],
        ["EfficientNet-B3 (ImageNet FT)", "10.73", "100.0 %", "0.819", "0.377"],
        ["ViT-B/16 (ImageNet Full FT)", "85.82", "100.0 %", "0.871", "0.472"],
        ["RETFound ViT-L (Head Only)", "2.44", "0.8 %", "0.877", "0.360"],
        ["RETFound ViT-L + LoRA (r=16)", "4.01", "1.3 %", "0.882", "0.392"],
        ["OptiscanAI Deployed Checkpoint", "2.44", "0.8 %", "0.872", "0.357"],
    ]

    for r_i, row in enumerate(abl_data):
        for c_i, val in enumerate(row):
            cell = tab.cell(r_i, c_i)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r_i == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r_i == 5:
                cell.fill.fore_color.rgb = RGBColor(0xFE, 0xF3, 0xC7)
            elif r_i == 6:
                cell.fill.fore_color.rgb = RGBColor(0xE0, 0xED, 0xFD)
            elif r_i % 2 == 1:
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = RGBColor(0xF1, 0xF5, 0xF9)

            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_HEAD
            p.font.size = Pt(13 if r_i == 0 else 12)
            p.font.bold = (r_i == 0 or r_i >= 5 or c_i == 0)
            p.font.color.rgb = WHITE if r_i == 0 else DARK
            p.alignment = PP_ALIGN.LEFT if c_i == 0 else PP_ALIGN.CENTER

    add_bullet_list(slide9, [
        ("Pretraining Efficiency:", "Freezing RETFound and tuning only 0.8% params achieves 0.877 AUC, beating fully trained ImageNet CNNs. LoRA adds +0.005 AUC and +0.032 AUPRC."),
        ("The Fine-Tuning Nuance:", "ViT-B/16 full fine-tuning attains higher AUPRC (0.472 vs 0.392). Retinal foundation models give parameter efficiency, but full fine-tuning retrieves rare positives better."),
        ("Normalisation Sensitivity:", "Substituting RFMiD channel statistics for ImageNet normalisation costs 0.054 AUC and 0.107 AUPRC. Preserving pretraining distribution alignment is critical."),
    ], 0.59, 4.00, 8.82, 2.00, pt_size=12.5, space_after=3.0)

    # =========================================================================
    # SLIDE 10: CLINICAL CASE GALLERY
    # =========================================================================
    slide10 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide10, "Clinical Screening Gallery: Real Cases", "10", title_sz=26.0)

    fig3_path = TMP_DIR / "fig3_cases.png"
    if fig3_path.exists():
        slide10.shapes.add_picture(str(fig3_path), Inches(0.59), Inches(1.15), Inches(8.82), Inches(2.55))

    add_bullet_list(slide10, [
        ("Healthy Control Eye:", "Disease Risk Negative → Routine follow-up. Prevents needless tertiary hospital visits."),
        ("Diabetic Retinopathy (DR):", "Microaneurysms detected → Automated Urgent Referral for laser photocoagulation."),
        ("Optic Disc Cupping (ODC):", "C/D ratio > 0.6 detected → Urgent triage for intraocular pressure and visual field testing."),
        ("Macular Degeneration (ARMD):", "Central drusen lesions detected → Urgent ophthalmology referral for anti-VEGF intervention."),
    ], 0.59, 3.85, 8.82, 2.10, pt_size=13.0, space_after=3.0)

    # =========================================================================
    # SLIDE 11: PREVALENCE REALITY GAP
    # =========================================================================
    slide11 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide11, "Prevalence Shift: Benchmark vs. Clinic", "11", title_sz=26.0)

    t_prev = slide11.shapes.add_table(5, 5, Inches(0.59), Inches(1.20), Inches(8.82), Inches(2.20))
    ptab = t_prev.table
    ptab.columns[0].width = Inches(3.62)
    ptab.columns[1].width = Inches(1.30)
    ptab.columns[2].width = Inches(1.30)
    ptab.columns[3].width = Inches(1.30)
    ptab.columns[4].width = Inches(1.30)

    prev_rows = [
        ["Screening Caseload / Setting", "Prevalence", "PPV", "NPV", "Referral %"],
        ["RFMiD Benchmark (Held-Out Test)", "79.0 %", "0.931", "0.810", "81.0 %"],
        ["District Hospital Caseload", "20.0 %", "0.470", "0.985", "41.0 %"],
        ["Rural Primary Care Clinic", "10.0 %", "0.283", "0.993", "34.0 %"],
        ["Community Mobile Screening", "5.0 %", "0.158", "0.997", "30.0 %"],
    ]

    for r_i, row in enumerate(prev_rows):
        for c_i, val in enumerate(row):
            cell = ptab.cell(r_i, c_i)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r_i == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r_i == 1:
                cell.fill.fore_color.rgb = RGBColor(0xFE, 0xF3, 0xC7)
            elif r_i % 2 == 1:
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = RGBColor(0xF1, 0xF5, 0xF9)

            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_HEAD
            p.font.size = Pt(13 if r_i == 0 else 12)
            p.font.bold = (r_i == 0 or r_i == 1 or c_i == 0)
            p.font.color.rgb = WHITE if r_i == 0 else DARK
            p.alignment = PP_ALIGN.LEFT if c_i == 0 else PP_ALIGN.CENTER

    add_bullet_list(slide11, [
        ("The Referral Bottleneck Tragedy:",
         "At 5% community prevalence, PPV drops from 93.1% to 15.8%. Sizing referral clinics on benchmark numbers will overburden ophthalmologists by 3x–6x with non-pathology cases."),
        ("The True Role: A Safe Rule-Out Sieve:",
         "Negative Predictive Value (NPV) exceeds 99.7% in low prevalence. OptiscanAI is not an autonomous doctor; it is a high-confidence sieve to protect specialists."),
    ], 0.59, 3.65, 8.82, 2.30, pt_size=14.5, space_after=8.0)

    # =========================================================================
    # SLIDE 12: LIMITATIONS & CONCLUSION (Split Layout)
    # =========================================================================
    slide12 = prs.slides.add_slide(prs.slide_layouts[0])
    add_slide_header(slide12, "Limitations and Conclusion", "12", title_sz=28.0)

    # Divider line
    div = slide12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.88), Inches(1.15), Inches(0.02), Inches(4.75))
    div.fill.solid()
    div.fill.fore_color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
    div.line.fill.background()

    # Left: Limitations
    tb_lim_hdr = slide12.shapes.add_textbox(Inches(0.59), Inches(1.15), Inches(4.10), Inches(0.40))
    tb_lim_hdr.text_frame.paragraphs[0].text = "LIMITATIONS & PRECONDITIONS"
    tb_lim_hdr.text_frame.paragraphs[0].font.name = FONT_HEAD
    tb_lim_hdr.text_frame.paragraphs[0].font.size = Pt(15.5)
    tb_lim_hdr.text_frame.paragraphs[0].font.bold = True
    tb_lim_hdr.text_frame.paragraphs[0].font.color.rgb = RED

    add_bullet_list(slide12, [
        ("Zero Ugandan Patient Images:", "Evaluated on international benchmark (RFMiD). Multi-centre clinical trials in Ugandan district hospitals remain mandatory."),
        ("Handheld Optical Shift:", "Validation on table-top fundus cameras cannot guarantee performance on low-cost smartphone adapters."),
        ("Regulatory Preconditions:", "Requires Uganda National Drug Authority (NDA) and Data Protection Act (2019) clearance before clinical trials."),
    ], 0.59, 1.70, 4.10, 4.20, pt_size=13.0, space_after=8.0)

    # Right: Key Takeaways
    tb_tk_hdr = slide12.shapes.add_textbox(Inches(5.05), Inches(1.15), Inches(4.35), Inches(0.40))
    tb_tk_hdr.text_frame.paragraphs[0].text = "KEY SCIENTIFIC TAKEAWAYS"
    tb_tk_hdr.text_frame.paragraphs[0].font.name = FONT_HEAD
    tb_tk_hdr.text_frame.paragraphs[0].font.size = Pt(15.5)
    tb_tk_hdr.text_frame.paragraphs[0].font.bold = True
    tb_tk_hdr.text_frame.paragraphs[0].font.color.rgb = GREEN

    add_stat_card(slide12, 5.05, 1.65, 4.35, 0.90, "0.872 AUC / 0.952 Triage", "Robust Ranking & Referral", "Evaluated strictly on 640 held-out test images.", NAVY)
    add_stat_card(slide12, 5.05, 2.70, 4.35, 0.90, "0.003 Calibrated ECE", "Silent Channels Resolved", "Platt recalibration restores 5 disabled disease channels.", GREEN)
    add_stat_card(slide12, 5.05, 3.75, 4.35, 0.90, "5.4 MB / 102 ms", "Offline Edge Distillation", "Runs autonomously on mobile hardware in rural health centres.", ORANGE)

    tb_final = slide12.shapes.add_textbox(Inches(5.05), Inches(4.80), Inches(4.35), Inches(1.10))
    tb_final.text_frame.word_wrap = True
    tb_final.text_frame.margin_left = Inches(0)
    p_fin = tb_final.text_frame.paragraphs[0]
    p_fin.text = "▸ OptiscanAI establishes a disciplined scientific framework that separates benchmark excellence from clinical deployment reality."
    p_fin.font.name = FONT_HEAD
    p_fin.font.size = Pt(13)
    p_fin.font.bold = True
    p_fin.font.color.rgb = DARK

    # =========================================================================
    # SLIDE 13: THANK YOU / Q&A
    # =========================================================================
    slide13 = prs.slides.add_slide(prs.slide_layouts[0])

    tb_ty = slide13.shapes.add_textbox(Inches(0.59), Inches(1.20), Inches(8.82), Inches(1.20))
    p_ty = tb_ty.text_frame.paragraphs[0]
    p_ty.text = "THANK YOU"
    p_ty.font.name = FONT_HEAD
    p_ty.font.size = Pt(44)
    p_ty.font.bold = True
    p_ty.font.color.rgb = ORANGE
    p_ty.alignment = PP_ALIGN.CENTER

    tb_q = slide13.shapes.add_textbox(Inches(0.59), Inches(2.30), Inches(8.82), Inches(1.00))
    p_q = tb_q.text_frame.paragraphs[0]
    p_q.text = "Questions & Discussion"
    p_q.font.name = FONT_HEAD
    p_q.font.size = Pt(30)
    p_q.font.bold = True
    p_q.font.color.rgb = DARK
    p_q.alignment = PP_ALIGN.CENTER

    tb_meta = slide13.shapes.add_textbox(Inches(1.00), Inches(3.50), Inches(8.00), Inches(2.20))
    tf_meta = tb_meta.text_frame
    tf_meta.word_wrap = True

    pm0 = tf_meta.paragraphs[0]
    pm0.text = "OptiscanAI: Large Vision Models for Multi-Disease Retinal Screening"
    pm0.font.name = FONT_HEAD
    pm0.font.size = Pt(17)
    pm0.font.bold = True
    pm0.font.color.rgb = NAVY
    pm0.alignment = PP_ALIGN.CENTER

    pm1 = tf_meta.add_paragraph()
    pm1.text = "Paper 83  •  The 10th National Conference on Communications (NCC 2026)"
    pm1.font.name = FONT_HEAD
    pm1.font.size = Pt(14)
    pm1.font.bold = True
    pm1.font.color.rgb = ORANGE
    pm1.alignment = PP_ALIGN.CENTER

    pm2 = tf_meta.add_paragraph()
    pm2.text = "Presenter: Mpairwe Lauben (mpairwe_lauben@students.mak.ac.ug)"
    pm2.font.name = FONT_BODY
    pm2.font.size = Pt(14)
    pm2.font.bold = True
    pm2.font.color.rgb = DARK
    pm2.alignment = PP_ALIGN.CENTER

    pm3 = tf_meta.add_paragraph()
    pm3.text = "Department of Networks & Department of Computer Science, Makerere University"
    pm3.font.name = FONT_BODY
    pm3.font.size = Pt(13)
    pm3.font.color.rgb = MUTED
    pm3.alignment = PP_ALIGN.CENTER

    pm4 = tf_meta.add_paragraph()
    pm4.text = "Live Interactive Prototype: https://landwind22-retinal-screening.hf.space"
    pm4.font.name = FONT_HEAD
    pm4.font.size = Pt(13.5)
    pm4.font.bold = True
    pm4.font.color.rgb = BLUE_ACCENT
    pm4.alignment = PP_ALIGN.CENTER

    # Save PPTX & Export PDF
    out_pptx = REPORTS_DIR / "OptiscanAI_NCC2026_Presentation.pptx"
    prs.save(str(out_pptx))
    print(f"Generated Presentation PPTX: {out_pptx}")

    out_pdf = REPORTS_DIR / "OptiscanAI_NCC2026_Presentation.pdf"
    print("Converting presentation to PDF via LibreOffice...")
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
    print(f"Generated Presentation PDF: {out_pdf}")
    return out_pptx, out_pdf


if __name__ == "__main__":
    build_presentation()
