"""
Data and model drift detection for production monitoring.
Detects distribution shifts in inputs and prediction patterns.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    drift_detected: bool = False
    severity: str = "none"  # none | warning | critical
    psi_scores: dict[str, float] = field(default_factory=dict)
    ks_p_values: dict[str, float] = field(default_factory=dict)
    details: str = ""


class DataDriftDetector:
    """Detects distribution shift in input images vs training baseline."""

    def __init__(
        self,
        reference_stats: dict,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
    ):
        self.reference = reference_stats
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    @staticmethod
    def compute_reference_stats(pixel_values: np.ndarray) -> dict:
        """Compute baseline statistics from training data pixel values.
        pixel_values: (N, C) or (N,) array of channel means per image."""
        return {
            "mean": float(pixel_values.mean()),
            "std": float(pixel_values.std()),
            "histogram": np.histogram(pixel_values, bins=50, density=True)[0].tolist(),
            "bin_edges": np.histogram(pixel_values, bins=50, density=True)[1].tolist(),
        }

    def detect(self, incoming_values: np.ndarray) -> DriftReport:
        """Compare incoming batch stats against reference."""
        report = DriftReport()

        # KS test
        ref_mean, ref_std = self.reference["mean"], self.reference["std"]
        ref_samples = np.random.normal(ref_mean, ref_std, size=len(incoming_values))
        ks_stat, ks_p = scipy_stats.ks_2samp(ref_samples, incoming_values)
        report.ks_p_values["intensity"] = float(ks_p)

        # PSI
        ref_hist = np.array(self.reference["histogram"]).clip(1e-10)
        inc_hist, _ = np.histogram(incoming_values, bins=len(ref_hist), density=True)
        inc_hist = inc_hist.clip(1e-10)
        psi = float(np.sum((inc_hist - ref_hist) * np.log(inc_hist / ref_hist)))
        report.psi_scores["intensity"] = psi

        # Evaluate
        if psi > self.psi_threshold * 2 or ks_p < self.ks_threshold / 10:
            report.drift_detected = True
            report.severity = "critical"
        elif psi > self.psi_threshold or ks_p < self.ks_threshold:
            report.drift_detected = True
            report.severity = "warning"

        if report.drift_detected:
            report.details = f"PSI={psi:.4f} (threshold={self.psi_threshold}), KS p={ks_p:.4f}"
            logger.warning(f"Data drift detected: {report.details}")

        return report


class ModelDriftDetector:
    """Detects shifts in model prediction distribution."""

    def __init__(self, reference_predictions: np.ndarray, window_size: int = 200):
        self.ref_mean = reference_predictions.mean(axis=0)
        self.ref_std = reference_predictions.std(axis=0).clip(1e-8)
        self.ref_confidence = reference_predictions.max(axis=1).mean()
        self.window = deque(maxlen=window_size)

    def record(self, predictions: np.ndarray):
        """Record a batch of predictions for rolling analysis."""
        for p in predictions:
            self.window.append(p)

    def detect(self) -> DriftReport:
        """Check if recent predictions have drifted."""
        if len(self.window) < 50:
            return DriftReport(details="Not enough data")

        recent = np.array(list(self.window))
        report = DriftReport()

        # Confidence drop
        recent_confidence = recent.max(axis=1).mean()
        conf_drop = self.ref_confidence - recent_confidence
        if conf_drop > 0.15:
            report.drift_detected = True
            report.severity = "critical"
            report.details = f"Confidence dropped {conf_drop:.3f}"
        elif conf_drop > 0.05:
            report.drift_detected = True
            report.severity = "warning"
            report.details = f"Confidence dropped {conf_drop:.3f}"

        # Per-class distribution shift
        recent_mean = recent.mean(axis=0)
        z_scores = np.abs((recent_mean - self.ref_mean) / self.ref_std)
        shifted_classes = (z_scores > 3).sum()
        if shifted_classes > len(self.ref_mean) * 0.2:
            report.drift_detected = True
            report.severity = "critical"
            report.psi_scores["shifted_classes"] = int(shifted_classes)

        return report
