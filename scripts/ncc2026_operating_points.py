#!/usr/bin/env python3
"""Compare decision-threshold policies, and what each costs in missed disease.

Reviewers asked what the false-positive story costs on the false-negative side.
This script fits three per-class threshold policies on the RFMiD validation
split and scores all of them on the held-out test split:

  uniform    tau_c = 0.5 for every class
  precfloor  lowest tau_c whose validation precision stays >= 0.10  (as deployed)
  sens90     highest tau_c whose validation sensitivity stays >= 0.90, which
             maximises specificity subject to a screening sensitivity target

Usage:
    python3 scripts/ncc2026_operating_points.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs/ncc2026"

GRID = np.arange(0.02, 0.99, 0.01)
SENS_TARGET = 0.90
PREC_FLOOR = 0.10
FALLBACK = 0.95


def counts(y: np.ndarray, p: np.ndarray, tau: float):
    pred = p >= tau
    return (
        int(np.sum(pred & (y == 1))),
        int(np.sum(pred & (y == 0))),
        int(np.sum(~pred & (y == 1))),
        int(np.sum(~pred & (y == 0))),
    )


def fit_sens_target(yv: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """Per class, the largest tau that still reaches SENS_TARGET on validation."""
    taus = np.full(yv.shape[1], 0.5)
    for c in range(yv.shape[1]):
        y, p = yv[:, c], pv[:, c]
        if y.sum() == 0:
            taus[c] = FALLBACK
            continue
        best = None
        for t in GRID:
            tp, _fp, fn, _tn = counts(y, p, t)
            if tp + fn > 0 and tp / (tp + fn) >= SENS_TARGET:
                best = t  # grid ascends, so the last qualifying tau is the largest
        taus[c] = best if best is not None else float(GRID[0])
    return taus


def fit_prec_floor(yv: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """Reimplementation of the deployed policy, fitted on the same split."""
    taus = np.full(yv.shape[1], FALLBACK)
    for c in range(yv.shape[1]):
        y, p = yv[:, c], pv[:, c]
        chosen = None
        for t in sorted(GRID, reverse=True):
            tp, fp, _fn, _tn = counts(y, p, t)
            if tp + fp > 0 and tp / (tp + fp) >= PREC_FLOOR:
                chosen = t  # descending grid, keep lowering while the floor holds
            elif chosen is not None:
                break
        taus[c] = chosen if chosen is not None else FALLBACK
    return taus


def score(y: np.ndarray, p: np.ndarray, taus: np.ndarray, classes: list[str]) -> dict:
    per, tot = {}, {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for c, name in enumerate(classes):
        tp, fp, fn, tn = counts(y[:, c], p[:, c], taus[c])
        for k, v in zip(("tp", "fp", "fn", "tn"), (tp, fp, fn, tn)):
            tot[k] += v
        per[name] = {
            "tau": float(taus[c]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "ppv": tp / (tp + fp) if tp + fp else float("nan"),
            "fnr": fn / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        }
    ev = [v for v in per.values() if not np.isnan(v["sensitivity"])]
    macro = {
        k: float(np.nanmean([v[k] for v in ev]))
        for k in ("sensitivity", "specificity", "ppv", "fnr", "fpr")
    }
    silent = [n for n, v in per.items() if v["tp"] + v["fp"] == 0]
    return {
        "per_class": per,
        "macro": macro,
        "totals": tot,
        "silent_classes": silent,
        "alerts_per_image": tot["tp"] + tot["fp"] and (tot["tp"] + tot["fp"]) / y.shape[0],
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calibrated",
        action="store_true",
        help="use the Platt-recalibrated probabilities from ncc2026_calibration.py",
    )
    args = ap.parse_args()
    tag = "calibrated" if args.calibrated else "fp32"

    val = np.load(OUT_DIR / f"probs_val_{tag}.npz", allow_pickle=True)
    test = np.load(OUT_DIR / f"probs_test_{tag}.npz", allow_pickle=True)
    classes = [str(c) for c in test["classes"]]
    yv, pv = val["labels"], val["probs"]
    yt, pt = test["labels"], test["probs"]

    policies = {
        "uniform": np.full(len(classes), 0.5),
        "precfloor": fit_prec_floor(yv, pv),
        "sens90": fit_sens_target(yv, pv),
        "deployed": test["thresholds"].astype(float),
    }

    report = {"n_test": int(yt.shape[0]), "n_val": int(yv.shape[0]), "policies": {}}
    for name, taus in policies.items():
        report["policies"][name] = score(yt, pt, taus, classes)

    suffix = "_calibrated" if args.calibrated else ""
    (OUT_DIR / f"operating_points{suffix}.json").write_text(
        json.dumps(report, indent=2, default=float)
    )

    print(f"{'policy':10s} {'sens':>6s} {'spec':>6s} {'PPV':>6s} {'FN':>5s} {'FP':>6s} "
          f"{'alerts/img':>10s}  silent")
    for name in policies:
        r = report["policies"][name]
        m, t = r["macro"], r["totals"]
        print(
            f"{name:10s} {m['sensitivity']:6.3f} {m['specificity']:6.3f} {m['ppv']:6.3f} "
            f"{t['fn']:5d} {t['fp']:6d} {r['alerts_per_image']:10.2f}  "
            f"{len(r['silent_classes'])}"
        )


if __name__ == "__main__":
    main()
