#!/usr/bin/env python3
"""Measure — and correct — the probability calibration of the screening model.

The deployed threshold policy degenerates on high-prevalence classes. This
script tests the hypothesis behind that: the model ranks well but its sigmoid
outputs are badly over-confident, so no threshold separates anything. It then
fits per-class Platt scaling on the validation split and re-measures.

Writes calibrated probability files that the operating-point script can consume.

Usage:
    python3 scripts/ncc2026_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/ncc2026"
BINS = 15


def ece(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    """Expected calibration error, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    err, n = 0.0, len(p)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum():
            err += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(err)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(
    pv: np.ndarray, yv: np.ndarray, pt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-class Platt scaling: a monotone, rank-preserving recalibration.

    Returns the recalibrated validation and test probabilities plus the (scale,
    shift) pair per class, so the same transform can be installed on the model
    as sigmoid(scale * logit + shift).
    """
    n_classes = pv.shape[1]
    cal_t = np.zeros_like(pt)
    cal_v = np.zeros_like(pv)
    scale = np.ones(n_classes, dtype=np.float32)
    shift = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        if yv[:, c].sum() < 2:
            cal_t[:, c], cal_v[:, c] = pt[:, c], pv[:, c]
            continue
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv[:, c]).reshape(-1, 1), yv[:, c])
        cal_t[:, c] = lr.predict_proba(logit(pt[:, c]).reshape(-1, 1))[:, 1]
        cal_v[:, c] = lr.predict_proba(logit(pv[:, c]).reshape(-1, 1))[:, 1]
        scale[c] = float(lr.coef_[0][0])
        shift[c] = float(lr.intercept_[0])
    return cal_v, cal_t, scale, shift


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(
        np.mean(
            [
                roc_auc_score(y[:, c], p[:, c])
                for c in range(y.shape[1])
                if 0 < y[:, c].sum() < len(y)
            ]
        )
    )


def main() -> None:
    val = np.load(OUT_DIR / "probs_val_fp32.npz", allow_pickle=True)
    test = np.load(OUT_DIR / "probs_test_fp32.npz", allow_pickle=True)
    classes = [str(c) for c in test["classes"]]
    pv, yv, pt, yt = val["probs"], val["labels"], test["probs"], test["labels"]

    cal_v, cal_t, scale, shift = fit_platt(pv, yv, pt)

    report = {
        "method": "per-class Platt scaling fitted on the RFMiD validation split",
        "n_val": int(len(yv)),
        "n_test": int(len(yt)),
        "before": {
            "flat_ece": ece(pt.ravel(), yt.ravel()),
            "macro_ece": float(np.mean([ece(pt[:, c], yt[:, c]) for c in range(len(classes))])),
            "mean_probability": float(pt.mean()),
            "min_probability": float(pt.min()),
            "macro_auc": macro_auc(yt, pt),
        },
        "after": {
            "flat_ece": ece(cal_t.ravel(), yt.ravel()),
            "macro_ece": float(np.mean([ece(cal_t[:, c], yt[:, c]) for c in range(len(classes))])),
            "mean_probability": float(cal_t.mean()),
            "min_probability": float(cal_t.min()),
            "macro_auc": macro_auc(yt, cal_t),
        },
        "empirical_positive_rate": float(yt.mean()),
        "per_class": [
            {
                "class": c,
                "prevalence": float(yt[:, i].mean()),
                "mean_prob_before": float(pt[:, i].mean()),
                "mean_prob_after": float(cal_t[:, i].mean()),
                "ece_before": ece(pt[:, i], yt[:, i]),
                "ece_after": ece(cal_t[:, i], yt[:, i]),
            }
            for i, c in enumerate(classes)
        ],
    }
    (OUT_DIR / "calibration.json").write_text(json.dumps(report, indent=2))

    for split, probs, labels, ids in (
        ("val", cal_v, yv, val.get("ids", np.arange(len(yv)))),
        ("test", cal_t, yt, test["ids"]),
    ):
        np.savez_compressed(
            OUT_DIR / f"probs_{split}_calibrated.npz",
            probs=probs,
            labels=labels,
            ids=ids,
            classes=test["classes"],
            thresholds=test["thresholds"],
        )

    # Deployable artefact: the calibration and the thresholds chosen against it
    # travel together, because either alone puts the model at an operating point
    # neither was fitted for.
    from ncc2026_operating_points import fit_sens_target  # noqa: E402

    taus = fit_sens_target(yv, cal_v)
    artefact = {
        "policy": "platt_calibration + sensitivity>=0.90 thresholds",
        "fitted_on": "RFMiD validation split",
        "classes": classes,
        "calibration": {"scale": scale.tolist(), "shift": shift.tolist()},
        "thresholds": [float(t) for t in taus],
        "measured_on_test": {
            "ece_before": report["before"]["flat_ece"],
            "ece_after": report["after"]["flat_ece"],
            "macro_auc": report["after"]["macro_auc"],
        },
    }
    (OUT_DIR / "calibration_artifact.json").write_text(json.dumps(artefact, indent=2))

    b, a = report["before"], report["after"]
    print(f"ECE      {b['flat_ece']:.4f} -> {a['flat_ece']:.4f}")
    print(f"macroECE {b['macro_ece']:.4f} -> {a['macro_ece']:.4f}")
    print(
        f"mean p   {b['mean_probability']:.4f} -> {a['mean_probability']:.4f} "
        f"(true rate {report['empirical_positive_rate']:.4f})"
    )
    print(f"macroAUC {b['macro_auc']:.4f} -> {a['macro_auc']:.4f} (rank preserving)")


if __name__ == "__main__":
    main()
