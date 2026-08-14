#!/usr/bin/env python3
"""Score the decision a primary-care screener actually makes: refer or not.

Per-disease multi-label metrics understate a screening tool, because the
operator's output is a single referral decision. RFMiD ships a Disease_Risk
flag (any pathology present), which is exactly that target. We also score the
knowledge graph's three-level referral priority, which is the component the
graph genuinely drives.

Thresholds are fitted on validation and applied unchanged to test.

Usage:
    python3 scripts/ncc2026_referral.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("USE_PRETRAINED", "0")

OUT_DIR = REPO / "outputs/ncc2026"
CACHE = REPO / "outputs/ncc2026/cache"

SENS_TARGETS = (0.90, 0.95)


def bootstrap(fn, y, s, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(fn(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def op_at(y: np.ndarray, s: np.ndarray, tau: float) -> dict:
    pred = s >= tau
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "tau": float(tau),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "referral_rate": (tp + fp) / len(y),
    }


def fit_tau(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Largest threshold still meeting the sensitivity target on validation."""
    best = float(np.min(s))
    for t in np.unique(np.round(s, 4)):
        o = op_at(y, s, t)
        if o["sensitivity"] >= target:
            best = float(t)
    return best


def kg_priority(probs: np.ndarray, taus: np.ndarray, classes: list[str]) -> list[str]:
    from src.models.vignn import ClinicalKnowledgeGraph

    kg = ClinicalKnowledgeGraph(classes)
    out = []
    for row in probs:
        detected = [c for i, c in enumerate(classes) if row[i] >= taus[i]]
        out.append(kg.get_referral_priority(detected))
    return out


def main() -> None:
    val = np.load(OUT_DIR / "probs_val_fp32.npz", allow_pickle=True)
    test = np.load(OUT_DIR / "probs_test_fp32.npz", allow_pickle=True)
    classes = [str(c) for c in test["classes"]]
    taus = test["thresholds"].astype(float)

    dfv = pd.read_csv(CACHE / "val_labels.csv", encoding="utf-8-sig")
    dft = pd.read_csv(CACHE / "test_labels.csv", encoding="utf-8-sig")
    yv = dfv["Disease_Risk"].to_numpy()
    yt = dft["Disease_Risk"].to_numpy()

    sv = val["probs"].max(axis=1)
    st = test["probs"].max(axis=1)

    report = {
        "target": "Disease_Risk (any pathology present)",
        "n_val": int(len(yv)),
        "n_test": int(len(yt)),
        "test_positive_rate": float(yt.mean()),
        "auc": float(roc_auc_score(yt, st)),
        "auc_ci": bootstrap(roc_auc_score, yt, st),
        "auprc": float(average_precision_score(yt, st)),
        "auprc_ci": bootstrap(average_precision_score, yt, st),
        "operating_points": {},
    }
    for target in SENS_TARGETS:
        tau = fit_tau(yv, sv, target)
        report["operating_points"][f"sens{int(target * 100)}"] = {
            "fitted_on_val": op_at(yv, sv, tau),
            "held_out_test": op_at(yt, st, tau),
        }

    # What the knowledge graph adds: a three-level referral priority. Scored
    # under both threshold policies, because the policy decides what it sees.
    sens90_taus = (
        np.load(OUT_DIR / "sens90_thresholds.npy")
        if (OUT_DIR / "sens90_thresholds.npy").exists()
        else taus
    )
    report["kg_referral_priority"] = {}
    for pol, tt in (("deployed", taus), ("sens90", sens90_taus)):
        prio = kg_priority(test["probs"], tt, classes)
        stats = {}
        for lv in sorted(set(prio)):
            mask = np.array([p == lv for p in prio])
            stats[lv] = {
                "n": int(mask.sum()),
                "share": float(mask.mean()),
                "disease_risk_rate": float(yt[mask].mean()) if mask.sum() else float("nan"),
            }
        report["kg_referral_priority"][pol] = stats

    # RFMiD is an enriched research corpus (79% positive). Project the same
    # operating point onto plausible primary-care prevalences via Bayes, which
    # is what actually decides referral burden in a Ugandan health centre.
    proj = {}
    for k, v in report["operating_points"].items():
        t = v["held_out_test"]
        sens, spec = t["sensitivity"], t["specificity"]
        proj[k] = {
            f"prev_{int(p * 100)}pct": {
                "ppv": sens * p / (sens * p + (1 - spec) * (1 - p)),
                "npv": spec * (1 - p) / (spec * (1 - p) + (1 - sens) * p),
                "referral_rate": sens * p + (1 - spec) * (1 - p),
            }
            for p in (0.05, 0.10, 0.20, 0.30)
        }
    report["prevalence_projection"] = proj

    (OUT_DIR / "referral.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"Disease_Risk positives in test: {yt.sum()}/{len(yt)} ({yt.mean():.1%})")
    print(f"referral AUC {report['auc']:.3f} CI {tuple(round(v, 3) for v in report['auc_ci'])}")
    print(f"referral AUPRC {report['auprc']:.3f}")
    for k, v in report["operating_points"].items():
        t = v["held_out_test"]
        print(
            f"{k}: tau={t['tau']:.3f} sens {t['sensitivity']:.3f} spec {t['specificity']:.3f} "
            f"PPV {t['ppv']:.3f} NPV {t['npv']:.3f} refer {t['referral_rate']:.1%} "
            f"(missed {t['fn']})"
        )
    print("KG priority:", json.dumps(report["kg_referral_priority"]))
    print("prevalence projection (sens95):", json.dumps(proj.get("sens95", {}), default=float))


if __name__ == "__main__":
    main()
