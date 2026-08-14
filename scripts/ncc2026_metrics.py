#!/usr/bin/env python3
"""Turn cached probabilities into the per-disease evidence tables for NCC 2026.

Computes, for every retained RFMiD class and for each operating point:
sensitivity, specificity, PPV, NPV, FNR, FPR, F1, AUC-ROC and AUPRC, each with
a 95% percentile bootstrap confidence interval over images.

The two operating points are the ones the paper contrasts:
  * uniform   -- a single tau = 0.5 for every class (no precision rescue)
  * perclass  -- the validation-fitted per-class tau_c under a precision floor

Usage:
    python3 scripts/ncc2026_metrics.py --variant fp32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs/ncc2026"

N_BOOT = 1000
SEED = 42

# Conditions that cost sight if missed; used for the screening-relevant subset.
SIGHT_THREATENING = ["DR", "ARMD", "CRVO", "BRVO", "ODC", "ODE", "AION", "MH", "MHL", "RS"]


def confusion(y: np.ndarray, p: np.ndarray, tau: float) -> tuple[int, int, int, int]:
    pred = p >= tau
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def rates(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    def div(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    sens = div(tp, tp + fn)
    spec = div(tn, tn + fp)
    ppv = div(tp, tp + fp)
    npv = div(tn, tn + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    return {
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "fnr": div(fn, tp + fn),
        "fpr": div(fp, fp + tn),
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def bootstrap_ci(
    fn_stat, y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float]:
    """Percentile CI by resampling images with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn_stat(y[idx], p[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def macro_bootstrap(
    labels: np.ndarray, probs: np.ndarray, taus: np.ndarray, stat: str, seed: int = SEED
) -> tuple[float, float]:
    """CI for a macro-averaged statistic, resampling images jointly.

    Images are resampled together across classes so the interval reflects
    patient-level sampling variability, which is what a test-set CI should
    represent. The resulting AUPRC interval is right-skewed rather than
    symmetric: average precision is a bounded, skewed statistic when a class
    has only a handful of positives. That is expected, not an artefact.
    """
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        y_b, p_b = labels[idx], probs[idx]
        per = []
        for c in range(labels.shape[1]):
            yc, pc = y_b[:, c], p_b[:, c]
            if stat == "auc":
                v = _safe_auc(yc, pc)
            elif stat == "ap":
                v = _safe_ap(yc, pc)
            else:
                v = rates(*confusion(yc, pc, taus[c]))[stat]
            if not np.isnan(v):
                per.append(v)
        if per:
            vals.append(np.mean(per))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def analyse(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    probs, labels = d["probs"], d["labels"]
    classes = [str(c) for c in d["classes"]]
    tau_pc = d["thresholds"].astype(np.float64)
    tau_uniform = np.full(len(classes), 0.5)

    report: dict = {
        "source": npz_path.name,
        "n_images": int(labels.shape[0]),
        "n_classes": len(classes),
        "classes": classes,
        "per_class": {},
        "operating_points": {},
    }

    for ci, cname in enumerate(classes):
        y, p = labels[:, ci].astype(int), probs[:, ci].astype(float)
        entry = {
            "n_positive": int(y.sum()),
            "prevalence": float(y.mean()),
            "auc": _safe_auc(y, p),
            "auprc": _safe_ap(y, p),
            "tau_perclass": float(tau_pc[ci]),
        }
        if 0 < y.sum() < len(y):
            entry["auc_ci"] = bootstrap_ci(_safe_auc, y, p)
            entry["auprc_ci"] = bootstrap_ci(_safe_ap, y, p)
        for name, taus in (("uniform", tau_uniform), ("perclass", tau_pc)):
            entry[name] = rates(*confusion(y, p, taus[ci]))
            if y.sum() > 0:
                entry[name]["sensitivity_ci"] = bootstrap_ci(
                    lambda yy, pp, t=taus[ci]: rates(*confusion(yy, pp, t))["sensitivity"], y, p
                )
                entry[name]["ppv_ci"] = bootstrap_ci(
                    lambda yy, pp, t=taus[ci]: rates(*confusion(yy, pp, t))["ppv"], y, p
                )
        report["per_class"][cname] = entry

    evaluable = [c for c in classes if report["per_class"][c]["n_positive"] > 0]
    for name, taus in (("uniform", tau_uniform), ("perclass", tau_pc)):
        per = [report["per_class"][c][name] for c in evaluable]
        tp = sum(e["tp"] for e in per)
        fp = sum(e["fp"] for e in per)
        fn = sum(e["fn"] for e in per)
        tn = sum(e["tn"] for e in per)
        macro = {
            k: float(np.nanmean([e[k] for e in per]))
            for k in ("sensitivity", "specificity", "ppv", "f1", "fnr", "fpr")
        }
        st = [c for c in evaluable if c in SIGHT_THREATENING]
        macro_st = {
            k: float(np.nanmean([report["per_class"][c][name][k] for c in st]))
            for k in ("sensitivity", "ppv", "fnr")
        }
        report["operating_points"][name] = {
            "macro": macro,
            "macro_ci": {
                "sensitivity": macro_bootstrap(labels, probs, taus, "sensitivity"),
                "ppv": macro_bootstrap(labels, probs, taus, "ppv"),
                "f1": macro_bootstrap(labels, probs, taus, "f1"),
            },
            "micro": rates(tp, fp, fn, tn),
            "sight_threatening_macro": macro_st,
            "total_fp": fp,
            "total_fn": fn,
            "total_tp": tp,
        }

    aucs = [report["per_class"][c]["auc"] for c in evaluable]
    aps = [report["per_class"][c]["auprc"] for c in evaluable]
    report["macro_auc"] = float(np.nanmean(aucs))
    report["macro_auprc"] = float(np.nanmean(aps))
    report["macro_auc_ci"] = macro_bootstrap(labels, probs, tau_pc, "auc")
    report["macro_auprc_ci"] = macro_bootstrap(labels, probs, tau_pc, "ap")
    report["n_evaluable_classes"] = len(evaluable)
    report["classes_without_test_positives"] = [c for c in classes if c not in evaluable]
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    npz = OUT_DIR / f"probs_{args.split}_{args.variant}.npz"
    rep = analyse(npz)
    out = OUT_DIR / f"metrics_{args.split}_{args.variant}.json"
    out.write_text(json.dumps(rep, indent=2))

    print(f"\n{npz.name}: {rep['n_images']} images, {rep['n_evaluable_classes']} evaluable classes")
    print(
        f"macro AUC   {rep['macro_auc']:.3f}  CI {tuple(round(v, 3) for v in rep['macro_auc_ci'])}"
    )
    print(
        f"macro AUPRC {rep['macro_auprc']:.3f}  "
        f"CI {tuple(round(v, 3) for v in rep['macro_auprc_ci'])}"
    )
    for name in ("uniform", "perclass"):
        op = rep["operating_points"][name]
        m = op["macro"]
        print(
            f"{name:9s} sens {m['sensitivity']:.3f} spec {m['specificity']:.3f} "
            f"PPV {m['ppv']:.3f} F1 {m['f1']:.3f} | FP {op['total_fp']} FN {op['total_fn']}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
