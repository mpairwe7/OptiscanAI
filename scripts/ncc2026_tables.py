#!/usr/bin/env python3
"""Emit the camera-ready LaTeX tables straight from the measured result files.

Every number in the paper is \\input from here, so the manuscript cannot drift
away from what the experiments actually produced.

Usage:
    python3 scripts/ncc2026_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/ncc2026"
TEX = REPO / "docs/Reports/tables"

RFMID_NAMES = {
    "DR": "diabetic retinopathy",
    "ARMD": "age-related macular degeneration",
    "MH": "media haze",
    "DN": "drusen",
    "MYA": "myopia",
    "BRVO": "branch retinal vein occlusion",
    "TSLN": "tessellation",
    "ERM": "epiretinal membrane",
    "LS": "laser scar",
    "MS": "macular scar",
    "CSR": "central serous retinopathy",
    "ODC": "optic disc cupping",
    "CRVO": "central retinal vein occlusion",
    "AH": "asteroid hyalosis",
    "ODP": "optic disc pallor",
    "ODE": "optic disc edema",
    "AION": "anterior ischaemic optic neuropathy",
    "PT": "parafoveal telangiectasia",
    "RT": "retinal traction",
    "RS": "retinitis",
    "CRS": "chorioretinitis",
    "EDN": "exudation",
    "RPEC": "retinal pigment epithelium changes",
    "MHL": "macular hole",
}

ARM_LABEL = {
    "resnet50": r"ResNet-50 (ImageNet, full FT)",
    "tf_efficientnet_b3": r"EfficientNet-B3 (ImageNet, FT)",
    "vit_base_patch16_224": r"ViT-B/16 (ImageNet, full FT)",
    "retfound_head": r"RETFound ViT-L, head only",
    "retfound_lora": r"RETFound ViT-L + LoRA $r{=}16$",
}
ARM_ORDER = [
    "resnet50",
    "tf_efficientnet_b3",
    "vit_base_patch16_224",
    "retfound_head",
    "retfound_lora",
]


def f(x: float, n: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def per_disease_table() -> str:
    m = json.loads((OUT / "metrics_test_fp32.json").read_text())
    rows = []
    for c, e in m["per_class"].items():
        ci = e.get("auc_ci", [float("nan")] * 2)
        d = e["perclass"]
        rows.append(
            f"{c} & {e['n_positive']} & {e['prevalence'] * 100:.1f} & "
            f"{f(e['auc'])} & [{f(ci[0], 2)}, {f(ci[1], 2)}] & {f(e['auprc'])} & "
            f"{e['tau_perclass']:.2f} & {f(d['sensitivity'])} & {f(d['specificity'])} & "
            f"{f(d['ppv'])} & {f(d['fnr'])} \\\\"
        )
    op = m["operating_points"]["perclass"]["macro"]
    macro = (
        r"\midrule Macro & " + f"{sum(e['n_positive'] for e in m['per_class'].values())} & -- & "
        f"{f(m['macro_auc'])} & [{f(m['macro_auc_ci'][0], 2)}, {f(m['macro_auc_ci'][1], 2)}] & "
        f"{f(m['macro_auprc'])} & -- & {f(op['sensitivity'])} & {f(op['specificity'])} & "
        f"{f(op['ppv'])} & {f(op['fnr'])} \\\\"
    )
    body = "\n".join(rows) + "\n" + macro
    legend = "; ".join(f"{k}, {v}" for k, v in RFMID_NAMES.items() if k in m["per_class"])
    return rf"""\begin{{table*}}[!t]
\centering
\caption{{Per-disease performance on the held-out official RFMiD test split ($n{{=}}640$ images,
24 retained classes). AUC and AUPRC are threshold-free; the operating-point columns use the
per-class thresholds $\tau_c$ fitted on the validation split under the deployed precision-floor
policy. Brackets give 95\% percentile bootstrap confidence intervals over 1{{,}}000 image
resamples. FNR is the false-negative rate and the false-positive rate is its complement on the
negative side, $\mathrm{{FPR}} = 1 - \mathrm{{Spec.}}$; a dash marks a quantity that is undefined
because the class never fires.}}
\label{{tab:perdisease}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrcccrcccc}}
\toprule
Class & $n_+$ & Prev.\ (\%) & AUC & 95\% CI & AUPRC & $\tau_c$ & Sens. & Spec. & PPV & FNR \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\vspace{{2pt}}
\parbox{{\textwidth}}{{\scriptsize RFMiD label abbreviations: {legend}.}}
\end{{table*}}
"""


def baseline_table() -> str:
    """One table covering every component the reviewers asked us to isolate:
    backbone, LoRA, preprocessing, the knowledge graph and quantisation."""
    m = json.loads((OUT / "metrics_test_fp32.json").read_text())
    rows = []

    def section(title: str) -> None:
        rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{title}}}}} \\")

    section("Backbone and adaptation (shared 15-epoch recipe)")
    for arm in ARM_ORDER:
        p = OUT / f"arm_{arm}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows.append(
            f"\\quad {ARM_LABEL[arm]} & {d['params_total_M']:.1f} & "
            f"{d['params_trainable_M']:.2f} & {d['trainable_fraction'] * 100:.1f} & "
            f"{f(d['test_macro_auc'])} & {f(d['test_macro_ap'])} \\\\"
        )

    # Preprocessing variants, run on the strongest arm
    preproc = [
        ("retfound_lora_datasetnorm", "RFMiD-specific normalisation"),
        ("retfound_lora_clahe", "ImageNet norm.\\ + CLAHE"),
    ]
    available = [(k, lab) for k, lab in preproc if (OUT / f"arm_{k}.json").exists()]
    if available:
        section("Input preprocessing (RETFound + LoRA arm)")
        for key, lab in available:
            d = json.loads((OUT / f"arm_{key}.json").read_text())
            rows.append(
                f"\\quad {lab} & -- & -- & -- & {f(d['test_macro_auc'])} & "
                f"{f(d['test_macro_ap'])} \\\\"
            )

    section("Post-hoc components (deployed checkpoint)")
    kg = json.loads((OUT / "kg_ablation.json").read_text())
    rows.append(
        f"\\quad + knowledge-graph reasoning & -- & -- & -- & "
        f"{f(kg['after']['macro_auc'])} & {f(kg['after']['macro_auprc'])} \\\\"
    )
    q = json.loads((OUT / "metrics_test_int8.json").read_text())
    rows.append(
        f"\\quad + INT8 dynamic quantisation & -- & -- & -- & "
        f"{f(q['macro_auc'])} & {f(q['macro_auprc'])} \\\\"
    )
    rows.append(r"\midrule")
    rows.append(
        r"Deployed checkpoint\textsuperscript{$\dagger$} & 305.7 & 2.44 & 0.8 & "
        f"{f(m['macro_auc'])} & {f(m['macro_auprc'])} \\\\"
    )

    kg_note = (
        f"The graph alters {kg['share_images_changed'] * 100:.1f}\\% of images and only one class "
        f"({', '.join(kg['classes_ever_adjusted'])}), because the partner conditions its "
        f"co-occurrence rules encode are mostly among the 21 classes dropped for having too "
        f"few training positives."
    )
    return rf"""\begin{{table}}[!t]
\centering
\caption{{Component ablation on the RFMiD test split. The adaptation arms share the data, the
asymmetric loss, a 15-epoch schedule, selection on validation mAP and the same threshold
optimisation, so differences are attributable to the component named. Post-hoc rows are applied
to the deployed checkpoint, so they are exact rather than re-trained. {kg_note}
\textsuperscript{{$\dagger$}}The deployed checkpoint additionally uses 25 epochs, staged backbone
unfreezing and six-view test-time augmentation.}}
\label{{tab:ablation}}
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lrrrcc}}
\toprule
Arm or component & Params & Train. & \% & AUC & AUPRC \\
 & (M) & (M) & train. & & \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def operating_point_table() -> str:
    raw = json.loads((OUT / "operating_points.json").read_text())
    cal = json.loads((OUT / "operating_points_calibrated.json").read_text())
    label = {
        "uniform": r"Uniform $\tau{=}0.5$",
        "deployed": r"Precision floor (deployed)",
        "sens90": r"Sensitivity target $\geq 0.90$",
    }
    rows = []
    for tag, src in (("raw", raw), ("calibrated", cal)):
        head = (
            r"\multicolumn{7}{l}{\emph{As trained}} \\"
            if tag == "raw"
            else r"\midrule \multicolumn{7}{l}{\emph{After per-class Platt recalibration}} \\"
        )
        rows.append(head)
        for pol in ("uniform", "deployed", "sens90"):
            r = src["policies"][pol]
            mm, t = r["macro"], r["totals"]
            rows.append(
                f"\\quad {label[pol]} & {f(mm['sensitivity'])} & {f(mm['specificity'])} & "
                f"{f(mm['ppv'])} & {t['fn']} & {t['fp']} & {len(r['silent_classes'])} \\\\"
            )
    return rf"""\begin{{table}}[!t]
\centering
\caption{{What the threshold policy costs on each side of the trade-off, on the RFMiD test split.
Counts are summed over all 24 classes and 640 images. ``Silent'' counts classes whose threshold
is so high that the class never fires, so its sensitivity is identically zero.}}
\label{{tab:operating}}
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lccccrr}}
\toprule
Policy & Sens. & Spec. & PPV & FN & FP & Silent \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def quantization_table() -> str:
    """FP32 vs INT8 on identical CPU threading, plus the artefact sizes."""
    sizes = json.loads((OUT / "artifact_sizes.json").read_text())
    rows, stats = [], {}
    for variant, name, size in (
        ("fp32_cpu", "FP32", sizes["teacher_fp32_torchscript_MB"]),
        ("int8", "INT8 dynamic", sizes["teacher_int8_MB"]),
    ):
        m = json.loads((OUT / f"metrics_test_{variant}.json").read_text())
        r = json.loads((OUT / f"runmeta_test_{variant}.json").read_text())
        stats[variant] = (m, r)
        op = m["operating_points"]["perclass"]["macro"]
        rows.append(
            f"{name} & {size:.0f} & {f(m['macro_auc'])} & {f(m['macro_auprc'])} & "
            f"{f(op['sensitivity'])} & {f(op['ppv'])} & "
            f"{r['seconds_per_image'] * 1000:.0f} \\\\"
        )
    a, ra = stats["fp32_cpu"]
    b, rb = stats["int8"]
    oa = a["operating_points"]["perclass"]["macro"]
    ob = b["operating_points"]["perclass"]["macro"]
    rows.append(r"\midrule")
    rows.append(
        f"Change & $-${sizes['teacher_reduction_pct']:.1f}\\% & "
        f"{b['macro_auc'] - a['macro_auc']:+.3f} & {b['macro_auprc'] - a['macro_auprc']:+.3f} & "
        f"{ob['sensitivity'] - oa['sensitivity']:+.3f} & {ob['ppv'] - oa['ppv']:+.3f} & "
        f"$-${100 * (1 - rb['seconds_per_image'] / ra['seconds_per_image']):.0f}\\% \\\\"
    )
    return rf"""\begin{{table}}[!t]
\centering
\caption{{Effect of 8-bit dynamic quantisation on the reference model, on the same 640 test images
with identical CPU threading, so accuracy, size and latency are directly comparable. The deployed
on-device student compresses further, from {sizes['student_fp32_onnx_MB']:.1f}\,MB to
{sizes['student_int8_onnx_MB']:.1f}\,MB ONNX ($-${sizes['student_reduction_pct']:.1f}\%), agreeing
with its FP32 export on {sizes['student_int8_binary_agreement'] * 100:.1f}\% of binary decisions at
$\tau{{=}}0.5$ and running at {sizes['student_int8_latency_p50_ms']:.0f}\,ms median per image.}}
\label{{tab:quant}}
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lrccccr}}
\toprule
Precision & MB & AUC & AUPRC & Sens. & PPV & ms/img \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def explainability_table() -> str:
    e = json.loads((OUT / "explainability_faithfulness.json").read_text())
    audit = json.loads(
        (REPO / "outputs/reasoner_comparison_real/faithfulness_audit.json").read_text()
    )
    m = e["methods"]
    rows = []
    for key, name in (
        ("gradcam", "Grad-CAM"),
        ("integrated_gradients", "Integrated Grad."),
        ("random", "Random (control)"),
    ):
        d = m[key]
        dp = f"$p={d['deletion_vs_random_p']:.3f}$" if "deletion_vs_random_p" in d else "--"
        ip = f"$p={d['insertion_vs_random_p']:.3f}$" if "insertion_vs_random_p" in d else "--"
        rows.append(
            f"{name} & {d['deletion_auc_mean']:.3f} & {dp} & "
            f"{d['insertion_auc_mean']:.3f} & {ip} \\\\"
        )
    # Prefer a narrator that always produces a report and never drops a finding;
    # among those, the one that inflates urgency least.
    usable = {
        k: v
        for k, v in audit["variants"].items()
        if v["generation_rate"] == 1.0 and v["omission_rate"] == 0.0
    } or audit["variants"]
    v = usable[min(usable, key=lambda k: usable[k]["severity_escalation_rate"])]

    def pct(x: float, n: int = 0) -> str:
        return rf"{x * 100:.{n}f}\%"

    note = (
        rf"Narrative layer ($n{{=}}{audit['n_test']}$ reports from the distilled narrator that "
        rf"always generates and never omits a detected finding): probability infidelity "
        rf"{pct(v['prob_infidelity_rate'])} (95\% CI {pct(v['prob_infidelity_ci95'][0])}--"
        rf"{pct(v['prob_infidelity_ci95'][1])}), finding omission {pct(v['omission_rate'])}, "
        rf"unsupported severity escalation {pct(v['severity_escalation_rate'])} "
        rf"(95\% CI {pct(v['severity_escalation_ci95'][0])}--"
        rf"{pct(v['severity_escalation_ci95'][1])})."
    )
    return rf"""\begin{{table}}[!t]
\centering
\caption{{Faithfulness of the explainability layer on {e['n_pairs']} confidently detected true
positives. Deletion AUC is better when lower, insertion AUC when higher; $p$-values are one-sided
Wilcoxon signed-rank tests against the random-saliency control on the identical images.
{note}}}
\label{{tab:explain}}
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lcccc}}
\toprule
Attribution & Del.\ AUC & vs.\ rand. & Ins.\ AUC & vs.\ rand. \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def prevalence_table() -> str:
    r = json.loads((OUT / "referral.json").read_text())
    op = r["operating_points"]["sens95"]["held_out_test"]
    proj = r["prevalence_projection"]["sens95"]
    rows = [
        f"RFMiD test (as measured) & {r['test_positive_rate'] * 100:.0f} & "
        f"{op['ppv']:.3f} & {op['npv']:.3f} & {op['referral_rate'] * 100:.0f} \\\\",
        r"\midrule",
    ]
    for k in ("prev_30pct", "prev_20pct", "prev_10pct", "prev_5pct"):
        p = proj[k]
        pct = k.split("_")[1].replace("pct", "")
        rows.append(
            f"Projected at {pct}\\% prev. & {pct} & {p['ppv']:.3f} & {p['npv']:.3f} & "
            f"{p['referral_rate'] * 100:.0f} \\\\"
        )
    return rf"""\begin{{table}}[!t]
\centering
\caption{{Why a benchmark result is not a clinical result. The referral operating point is held
fixed at the measured test sensitivity {op['sensitivity']:.3f} and specificity
{op['specificity']:.3f}; only the prevalence of the screened population changes. RFMiD is an
enriched research corpus, so its positive rate is far above any primary-care caseload.}}
\label{{tab:prevalence}}
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{lrccr}}
\toprule
Population & Prev.\ (\%) & PPV & NPV & Refer (\%) \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    TEX.mkdir(parents=True, exist_ok=True)
    builders = {
        "tab_perdisease.tex": per_disease_table,
        "tab_ablation.tex": baseline_table,
        "tab_operating.tex": operating_point_table,
        "tab_quant.tex": quantization_table,
        "tab_explain.tex": explainability_table,
        "tab_prevalence.tex": prevalence_table,
    }
    for name, fn in builders.items():
        try:
            (TEX / name).write_text(fn())
            print(f"wrote {name}")
        except FileNotFoundError as exc:
            print(f"SKIP {name}: {exc}")


if __name__ == "__main__":
    main()
