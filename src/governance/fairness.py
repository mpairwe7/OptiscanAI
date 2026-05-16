"""Fairness evaluation for medical AI models.
Checks for demographic parity and performance equity across subgroups."""

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class SubgroupMetrics:
    name: str
    size: int
    f1_macro: float
    auc_roc: float
    precision: float
    recall: float


@dataclass
class FairnessReport:
    overall_f1: float = 0.0
    subgroup_metrics: list[SubgroupMetrics] = field(default_factory=list)
    max_f1_disparity: float = 0.0
    max_auc_disparity: float = 0.0
    equalized_odds_satisfied: bool = False
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_f1": self.overall_f1,
            "subgroups": [
                {
                    "name": s.name,
                    "size": s.size,
                    "f1_macro": s.f1_macro,
                    "auc_roc": s.auc_roc,
                    "precision": s.precision,
                    "recall": s.recall,
                }
                for s in self.subgroup_metrics
            ],
            "max_f1_disparity": self.max_f1_disparity,
            "max_auc_disparity": self.max_auc_disparity,
            "equalized_odds_satisfied": self.equalized_odds_satisfied,
            "recommendations": self.recommendations,
        }


class FairnessEvaluator:
    """Evaluates model fairness across disease categories and severity groups."""

    def __init__(self, disease_names: list[str], threshold: float = 0.5):
        self.disease_names = disease_names
        self.threshold = threshold

        # Define disease category groupings for subgroup analysis
        self.disease_categories = {
            "VASCULAR": ["DR", "BRVO", "CRVO", "CRAO", "BRAO", "HR", "PRH", "VH", "MCA", "VS"],
            "DEGENERATIVE": ["ARMD", "MH", "DN", "MYA", "ERM", "MHL", "RP"],
            "GLAUCOMATOUS": ["ODC", "ODP", "ODE", "ODPM"],
            "INFLAMMATORY": ["RS", "CRS", "CWS", "CB", "RPEC"],
            "OTHER": [
                "TSLN",
                "LS",
                "MS",
                "CSR",
                "TV",
                "AH",
                "ST",
                "AION",
                "PT",
                "RT",
                "EDN",
                "MNF",
                "TD",
                "CME",
                "PTCR",
                "CF",
                "PLQ",
                "HPED",
                "CL",
            ],
        }

    def evaluate_category_fairness(
        self, targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
    ) -> FairnessReport:
        """Evaluate performance parity across disease categories."""
        preds = (probabilities > self.threshold).astype(np.float32)

        overall_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        report = FairnessReport(overall_f1=overall_f1)

        for category, diseases in self.disease_categories.items():
            col_indices = [i for i, d in enumerate(self.disease_names) if d in diseases]
            if not col_indices:
                continue

            cat_targets = targets[:, col_indices]
            cat_preds = preds[:, col_indices]
            cat_probs = probabilities[:, col_indices]

            # Only compute if there are positive samples
            valid = cat_targets.sum(axis=0) > 0
            if valid.sum() < 1:
                continue

            cat_f1 = float(
                f1_score(
                    cat_targets[:, valid], cat_preds[:, valid], average="macro", zero_division=0
                )
            )

            try:
                cat_auc = float(
                    roc_auc_score(cat_targets[:, valid], cat_probs[:, valid], average="macro")
                )
            except ValueError:
                cat_auc = 0.0

            from sklearn.metrics import precision_score, recall_score

            cat_prec = float(
                precision_score(
                    cat_targets[:, valid], cat_preds[:, valid], average="macro", zero_division=0
                )
            )
            cat_rec = float(
                recall_score(
                    cat_targets[:, valid], cat_preds[:, valid], average="macro", zero_division=0
                )
            )

            report.subgroup_metrics.append(
                SubgroupMetrics(
                    name=category,
                    size=int(cat_targets[:, valid].sum()),
                    f1_macro=cat_f1,
                    auc_roc=cat_auc,
                    precision=cat_prec,
                    recall=cat_rec,
                )
            )

        # Compute disparities
        if len(report.subgroup_metrics) >= 2:
            f1_scores = [s.f1_macro for s in report.subgroup_metrics]
            auc_scores = [s.auc_roc for s in report.subgroup_metrics if s.auc_roc > 0]
            report.max_f1_disparity = max(f1_scores) - min(f1_scores)
            report.max_auc_disparity = (max(auc_scores) - min(auc_scores)) if auc_scores else 0.0
            report.equalized_odds_satisfied = report.max_f1_disparity < 0.1

        # Recommendations
        if report.max_f1_disparity > 0.15:
            report.recommendations.append(
                f"HIGH: F1 disparity of {report.max_f1_disparity:.3f} exceeds 0.15 threshold. "
                "Consider oversampling underperforming categories or adjusting per-category thresholds."
            )
        if report.max_f1_disparity > 0.05:
            report.recommendations.append(
                "MEDIUM: Some disease categories show performance gaps. "
                "Review per-class metrics and consider targeted data augmentation."
            )

        worst = (
            min(report.subgroup_metrics, key=lambda s: s.f1_macro)
            if report.subgroup_metrics
            else None
        )
        if worst and worst.f1_macro < 0.3:
            report.recommendations.append(
                f"HIGH: Category '{worst.name}' has F1={worst.f1_macro:.3f}. "
                "This group likely has insufficient training samples."
            )

        if not report.recommendations:
            report.recommendations.append(
                "Performance is relatively balanced across disease categories."
            )

        return report

    def evaluate_prevalence_fairness(
        self, targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
    ) -> FairnessReport:
        """Evaluate performance equity by disease prevalence (common vs rare)."""
        preds = (probabilities > self.threshold).astype(np.float32)
        prevalence = targets.sum(axis=0)

        # Split into prevalence buckets
        buckets = {"COMMON": [], "MODERATE": [], "RARE": []}
        for i, count in enumerate(prevalence):
            if count > len(targets) * 0.05:
                buckets["COMMON"].append(i)
            elif count > len(targets) * 0.01:
                buckets["MODERATE"].append(i)
            else:
                buckets["RARE"].append(i)

        overall_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        report = FairnessReport(overall_f1=overall_f1)

        for bucket_name, indices in buckets.items():
            if not indices:
                continue
            b_targets = targets[:, indices]
            b_preds = preds[:, indices]
            b_probs = probabilities[:, indices]
            valid = b_targets.sum(axis=0) > 0
            if valid.sum() < 1:
                continue

            b_f1 = float(
                f1_score(b_targets[:, valid], b_preds[:, valid], average="macro", zero_division=0)
            )
            try:
                b_auc = float(
                    roc_auc_score(b_targets[:, valid], b_probs[:, valid], average="macro")
                )
            except ValueError:
                b_auc = 0.0

            report.subgroup_metrics.append(
                SubgroupMetrics(
                    name=bucket_name,
                    size=int(b_targets.sum()),
                    f1_macro=b_f1,
                    auc_roc=b_auc,
                    precision=0.0,
                    recall=0.0,
                )
            )

        return report
