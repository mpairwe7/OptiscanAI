#!/usr/bin/env python3
"""Measure what the clinical knowledge graph changes, rather than asserting it.

The graph is applied after classification, so its effect on the benchmark can be
isolated exactly: run the same test probabilities through
``apply_clinical_reasoning`` and re-score. This also counts how often each rule
actually fires, which is the part the previous draft never checked.

Usage:
    python3 scripts/ncc2026_kg_ablation.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("USE_PRETRAINED", "0")

OUT = REPO / "outputs/ncc2026"


def macro(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    auc = [
        roc_auc_score(y[:, c], p[:, c]) for c in range(y.shape[1]) if 0 < y[:, c].sum() < len(y)
    ]
    ap = [average_precision_score(y[:, c], p[:, c]) for c in range(y.shape[1]) if y[:, c].sum() > 0]
    return float(np.mean(auc)), float(np.mean(ap))


def main() -> None:
    from src.models.vignn import ClinicalKnowledgeGraph

    d = np.load(OUT / "probs_test_fp32.npz", allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    probs, labels = d["probs"], d["labels"]
    kg = ClinicalKnowledgeGraph(classes)

    refined = np.array(probs, copy=True)
    n_changed_rows = 0
    changed_per_class = {c: 0 for c in classes}

    for i, row in enumerate(probs):
        pred = {c: float(row[j]) for j, c in enumerate(classes)}
        out = kg.apply_clinical_reasoning(pred)
        changed = False
        for j, c in enumerate(classes):
            if abs(out[c] - pred[c]) > 1e-9:
                refined[i, j] = out[c]
                changed_per_class[c] += 1
                changed = True
        n_changed_rows += int(changed)

    auc_b, ap_b = macro(labels, probs)
    auc_a, ap_a = macro(labels, refined)

    report = {
        "n_images": int(len(probs)),
        "n_classes": len(classes),
        "graph_nodes": len(classes),
        "graph_edges": int(kg.get_edge_count()),
        "images_changed_by_reasoning": n_changed_rows,
        "share_images_changed": n_changed_rows / len(probs),
        "classes_ever_adjusted": {c: n for c, n in changed_per_class.items() if n},
        "before": {"macro_auc": auc_b, "macro_auprc": ap_b},
        "after": {"macro_auc": auc_a, "macro_auprc": ap_a},
        "delta": {"macro_auc": auc_a - auc_b, "macro_auprc": ap_a - ap_b},
    }
    (OUT / "kg_ablation.json").write_text(json.dumps(report, indent=2))

    print(f"graph: {report['graph_nodes']} nodes, {report['graph_edges']} typed edges")
    print(
        f"reasoning altered {n_changed_rows}/{len(probs)} images "
        f"({report['share_images_changed']:.1%})"
    )
    print(f"classes ever adjusted: {report['classes_ever_adjusted'] or 'none'}")
    print(f"macro AUC   {auc_b:.4f} -> {auc_a:.4f}  ({auc_a - auc_b:+.4f})")
    print(f"macro AUPRC {ap_b:.4f} -> {ap_a:.4f}  ({ap_a - ap_b:+.4f})")


if __name__ == "__main__":
    main()
