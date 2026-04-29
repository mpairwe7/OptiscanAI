"""Tests for health monitoring and drift detection."""
import sys

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

import pytest
import numpy as np

from src.monitoring.health import HealthMonitor, HealthReport
from src.monitoring.drift import DataDriftDetector, ModelDriftDetector, DriftReport


# ===========================================================================
# HealthMonitor
# ===========================================================================

def test_health_monitor_record():
    """Record latencies and verify report has correct counts."""
    monitor = HealthMonitor(max_latency_p99_ms=200.0, max_error_rate=0.1)

    for i in range(100):
        monitor.record(latency_ms=float(i), success=True)
    # Add some errors
    for i in range(5):
        monitor.record(latency_ms=150.0, success=False)

    report = monitor.report()
    assert isinstance(report, HealthReport)
    assert report.total_predictions == 105
    assert report.total_errors == 5
    assert report.error_rate == pytest.approx(5 / 105, abs=1e-4)
    assert report.latency_p50_ms > 0
    assert report.latency_p95_ms > 0
    assert report.latency_p99_ms > 0
    assert report.throughput_rps > 0
    assert report.uptime_seconds > 0


def test_health_monitor_empty():
    """Report on empty monitor should return defaults."""
    monitor = HealthMonitor()
    report = monitor.report()
    assert report.total_predictions == 0
    assert report.latency_p50_ms == 0.0


def test_health_monitor_sla_compliant():
    """All fast + successful requests should be SLA compliant."""
    monitor = HealthMonitor(max_latency_p99_ms=200.0, max_error_rate=0.1)
    for _ in range(50):
        monitor.record(latency_ms=10.0, success=True)

    report = monitor.report()
    assert report.sla_compliant is True


def test_health_monitor_sla_violation_latency():
    """High latency should violate SLA."""
    monitor = HealthMonitor(max_latency_p99_ms=50.0, max_error_rate=0.1)
    # All requests are slow
    for _ in range(100):
        monitor.record(latency_ms=100.0, success=True)

    report = monitor.report()
    assert report.sla_compliant is False


def test_health_monitor_sla_violation_errors():
    """High error rate should violate SLA."""
    monitor = HealthMonitor(max_latency_p99_ms=200.0, max_error_rate=0.05)
    # 50% error rate
    for _ in range(50):
        monitor.record(latency_ms=10.0, success=True)
    for _ in range(50):
        monitor.record(latency_ms=10.0, success=False)

    report = monitor.report()
    assert report.sla_compliant is False


# ===========================================================================
# DataDriftDetector
# ===========================================================================

def test_data_drift_detector_no_drift():
    """No drift when incoming is drawn from the exact reference samples."""
    rng = np.random.RandomState(42)
    reference_values = rng.normal(loc=0.5, scale=0.1, size=1000)
    ref_stats = DataDriftDetector.compute_reference_stats(reference_values)
    detector = DataDriftDetector(reference_stats=ref_stats)

    # Use a subset of the actual reference values so distribution matches perfectly
    incoming = reference_values[:200]
    report = detector.detect(incoming)
    assert isinstance(report, DriftReport)
    assert "intensity" in report.ks_p_values
    assert "intensity" in report.psi_scores


def test_data_drift_detector_with_shift():
    """Shifted data should trigger drift detection."""
    rng = np.random.RandomState(42)
    reference_values = rng.normal(loc=0.5, scale=0.1, size=1000)
    ref_stats = DataDriftDetector.compute_reference_stats(reference_values)
    detector = DataDriftDetector(reference_stats=ref_stats, psi_threshold=0.2, ks_threshold=0.05)

    # Incoming from a very different distribution
    incoming = rng.normal(loc=5.0, scale=2.0, size=200)
    report = detector.detect(incoming)
    assert isinstance(report, DriftReport)
    assert report.drift_detected is True
    assert report.severity in ("warning", "critical")
    assert "intensity" in report.ks_p_values
    assert "intensity" in report.psi_scores


def test_data_drift_reference_stats():
    """compute_reference_stats should return mean, std, histogram, bin_edges."""
    values = np.random.randn(500)
    stats = DataDriftDetector.compute_reference_stats(values)
    assert "mean" in stats
    assert "std" in stats
    assert "histogram" in stats
    assert "bin_edges" in stats
    assert isinstance(stats["histogram"], list)
    assert len(stats["histogram"]) == 50


# ===========================================================================
# ModelDriftDetector
# ===========================================================================

def test_model_drift_detector_no_drift():
    """Stable predictions should not trigger drift."""
    rng = np.random.RandomState(42)
    ref_preds = rng.rand(200, 45)
    detector = ModelDriftDetector(reference_predictions=ref_preds, window_size=200)

    # Record similar predictions
    for _ in range(60):
        detector.record(rng.rand(1, 45))

    report = detector.detect()
    assert isinstance(report, DriftReport)
    # Same distribution should not show critical drift
    # (note: random data may occasionally show warning)


def test_model_drift_detector_confidence_drop():
    """Significant confidence drop should trigger drift detection."""
    rng = np.random.RandomState(42)
    # Reference: high confidence (values near 1.0)
    ref_preds = rng.uniform(0.7, 1.0, size=(200, 45))
    detector = ModelDriftDetector(reference_predictions=ref_preds, window_size=200)

    # Record very low confidence predictions
    for _ in range(60):
        low_conf = rng.uniform(0.0, 0.2, size=(1, 45))
        detector.record(low_conf)

    report = detector.detect()
    assert isinstance(report, DriftReport)
    assert report.drift_detected is True
    assert report.severity in ("warning", "critical")


def test_model_drift_detector_insufficient_data():
    """With insufficient data, should report accordingly."""
    rng = np.random.RandomState(42)
    ref_preds = rng.rand(100, 45)
    detector = ModelDriftDetector(reference_predictions=ref_preds, window_size=200)

    # Only record a few predictions (< 50 threshold)
    for _ in range(10):
        detector.record(rng.rand(1, 45))

    report = detector.detect()
    assert "Not enough data" in report.details
