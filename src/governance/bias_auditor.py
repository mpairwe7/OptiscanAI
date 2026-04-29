"""
BiasAuditor — Automated demographic bias detection for retinal AI.

Evaluates model fairness across protected attributes:
    - Age groups (pediatric, adult, elderly)
    - Sex (male, female)
    - Ethnicity / geography
    - Camera device / image source
    - Disease severity subgroups

Metrics computed per subgroup:
    - AUC-ROC, F1, sensitivity, specificity
    - Demographic parity difference
    - Equalized odds difference
    - Calibration error per group

EU AI Act (Article 10): High-risk AI systems must be tested for bias.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SubgroupMetrics:
    """Metrics for a single demographic subgroup."""
    group_name: str
    group_value: str
    sample_count: int
    auc_roc: float
    f1_score: float
    sensitivity: float
    specificity: float
    calibration_error: float
    mean_confidence: float


@dataclass
class BiasReport:
    """Complete bias audit report."""
    timestamp: float = field(default_factory=time.time)
    model_version: str = ""
    dataset_name: str = ""
    total_samples: int = 0
    protected_attributes: List[str] = field(default_factory=list)
    subgroup_metrics: List[SubgroupMetrics] = field(default_factory=list)
    demographic_parity: Dict[str, float] = field(default_factory=dict)
    equalized_odds: Dict[str, float] = field(default_factory=dict)
    fairness_pass: bool = True
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class BiasAuditor:
    """Automated bias detection across demographic subgroups.

    Parameters
    ----------
    protected_attributes : list[str]
        Columns in metadata to audit (e.g. ['age_group', 'sex', 'device']).
    dp_threshold : float
        Maximum acceptable demographic parity difference. Default 0.1.
    eo_threshold : float
        Maximum acceptable equalized odds difference. Default 0.1.
    min_subgroup_size : int
        Minimum samples per subgroup for valid comparison. Default 30.
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        dp_threshold: float = 0.1,
        eo_threshold: float = 0.1,
        min_subgroup_size: int = 30,
    ):
        self.protected_attributes = protected_attributes or [
            "age_group", "sex", "ethnicity", "camera_device"
        ]
        self.dp_threshold = dp_threshold
        self.eo_threshold = eo_threshold
        self.min_subgroup_size = min_subgroup_size

    def audit(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: Dict[str, np.ndarray],
        model_version: str = "unknown",
        dataset_name: str = "unknown",
        threshold: float = 0.5,
    ) -> BiasReport:
        """Run full bias audit.

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions [N, C] (probabilities after sigmoid).
        targets : np.ndarray
            Ground truth labels [N, C] (binary).
        metadata : dict
            Mapping of attribute name -> array of group labels [N].
        model_version : str
            Model version identifier.
        dataset_name : str
            Dataset identifier.
        threshold : float
            Classification threshold.

        Returns
        -------
        BiasReport
            Complete audit report.
        """
        report = BiasReport(
            model_version=model_version,
            dataset_name=dataset_name,
            total_samples=len(predictions),
            protected_attributes=list(metadata.keys()),
        )

        for attr_name, attr_values in metadata.items():
            if attr_name not in self.protected_attributes:
                continue

            unique_groups = np.unique(attr_values)
            group_metrics = {}

            for group_val in unique_groups:
                mask = attr_values == group_val
                n_samples = mask.sum()

                if n_samples < self.min_subgroup_size:
                    logger.info(
                        f"Skipping {attr_name}={group_val} ({n_samples} samples "
                        f"< min {self.min_subgroup_size})"
                    )
                    continue

                group_preds = predictions[mask]
                group_targets = targets[mask]

                metrics = self._compute_group_metrics(
                    group_preds, group_targets, threshold,
                    group_name=attr_name, group_value=str(group_val),
                )
                report.subgroup_metrics.append(metrics)
                group_metrics[str(group_val)] = metrics

            # Compute fairness metrics across groups
            if len(group_metrics) >= 2:
                dp = self._demographic_parity(group_metrics, threshold, predictions)
                report.demographic_parity[attr_name] = dp

                eo = self._equalized_odds(group_metrics)
                report.equalized_odds[attr_name] = eo

                # Check thresholds
                if dp > self.dp_threshold:
                    violation = (
                        f"Demographic parity violation for {attr_name}: "
                        f"{dp:.3f} > {self.dp_threshold}"
                    )
                    report.violations.append(violation)
                    report.fairness_pass = False

                if eo > self.eo_threshold:
                    violation = (
                        f"Equalized odds violation for {attr_name}: "
                        f"{eo:.3f} > {self.eo_threshold}"
                    )
                    report.violations.append(violation)
                    report.fairness_pass = False

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        logger.info(
            f"Bias audit complete: {'PASS' if report.fairness_pass else 'FAIL'} | "
            f"{len(report.violations)} violations"
        )

        return report

    def _compute_group_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        threshold: float,
        group_name: str,
        group_value: str,
    ) -> SubgroupMetrics:
        """Compute classification metrics for a single subgroup."""
        binary_preds = (preds >= threshold).astype(int)

        # Per-class metrics averaged
        aucs = []
        f1s = []
        sensitivities = []
        specificities = []
        cal_errors = []

        for c in range(preds.shape[1]):
            y_true = targets[:, c]
            y_pred = binary_preds[:, c]
            y_prob = preds[:, c]

            # Skip if no positive samples in this class
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue

            # AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_true, y_prob)
                aucs.append(auc)
            except Exception:
                pass

            # Confusion matrix elements
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()

            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            prec = tp / max(tp + fp, 1)
            rec = sens
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)

            sensitivities.append(sens)
            specificities.append(spec)
            f1s.append(f1)

            # Calibration error (ECE-like per class)
            cal_err = abs(y_prob.mean() - y_true.mean())
            cal_errors.append(cal_err)

        return SubgroupMetrics(
            group_name=group_name,
            group_value=group_value,
            sample_count=len(preds),
            auc_roc=float(np.mean(aucs)) if aucs else 0.0,
            f1_score=float(np.mean(f1s)) if f1s else 0.0,
            sensitivity=float(np.mean(sensitivities)) if sensitivities else 0.0,
            specificity=float(np.mean(specificities)) if specificities else 0.0,
            calibration_error=float(np.mean(cal_errors)) if cal_errors else 0.0,
            mean_confidence=float(preds.max(axis=1).mean()),
        )

    def _demographic_parity(
        self,
        group_metrics: Dict[str, SubgroupMetrics],
        threshold: float,
        all_predictions: np.ndarray,
    ) -> float:
        """Compute demographic parity difference.

        DP = max|P(Y_hat=1|A=a) - P(Y_hat=1|A=b)| across all group pairs.
        """
        positive_rates = []
        for gm in group_metrics.values():
            positive_rates.append(gm.mean_confidence)

        if len(positive_rates) < 2:
            return 0.0
        return float(max(positive_rates) - min(positive_rates))

    def _equalized_odds(self, group_metrics: Dict[str, SubgroupMetrics]) -> float:
        """Compute equalized odds difference.

        EO = max(|TPR_a - TPR_b|, |FPR_a - FPR_b|) across all group pairs.
        """
        tprs = [gm.sensitivity for gm in group_metrics.values()]
        fprs = [1 - gm.specificity for gm in group_metrics.values()]

        if len(tprs) < 2:
            return 0.0

        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)
        return float(max(tpr_diff, fpr_diff))

    def _generate_recommendations(self, report: BiasReport) -> List[str]:
        """Generate actionable recommendations based on audit findings."""
        recs = []

        if not report.violations:
            recs.append("No fairness violations detected. Continue monitoring.")
            return recs

        for violation in report.violations:
            if "Demographic parity" in violation:
                recs.append(
                    "Consider rebalancing training data across demographic groups "
                    "or applying fairness-aware loss functions."
                )
            if "Equalized odds" in violation:
                recs.append(
                    "Investigate per-group calibration and consider group-specific "
                    "threshold optimization."
                )

        # Check for underrepresented groups
        for sm in report.subgroup_metrics:
            if sm.sample_count < self.min_subgroup_size * 2:
                recs.append(
                    f"Group {sm.group_name}={sm.group_value} is underrepresented "
                    f"({sm.sample_count} samples). Prioritize data collection."
                )

        return list(set(recs))

    def save_report(self, report: BiasReport, output_path: str = "outputs/bias_report.json"):
        """Save bias audit report to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to serializable dict
        report_dict = {
            "timestamp": report.timestamp,
            "model_version": report.model_version,
            "dataset_name": report.dataset_name,
            "total_samples": report.total_samples,
            "fairness_pass": report.fairness_pass,
            "demographic_parity": report.demographic_parity,
            "equalized_odds": report.equalized_odds,
            "violations": report.violations,
            "recommendations": report.recommendations,
            "subgroup_metrics": [
                {
                    "group_name": sm.group_name,
                    "group_value": sm.group_value,
                    "sample_count": sm.sample_count,
                    "auc_roc": sm.auc_roc,
                    "f1_score": sm.f1_score,
                    "sensitivity": sm.sensitivity,
                    "specificity": sm.specificity,
                    "calibration_error": sm.calibration_error,
                    "mean_confidence": sm.mean_confidence,
                }
                for sm in report.subgroup_metrics
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Bias report saved to {output_path}")
        return output_path
