"""Tests for MetricTracker: update, compute, reset, value ranges."""
import sys

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

import pytest
import torch
import numpy as np

from src.training.metrics import (
    MetricTracker,
    compute_multilabel_metrics,
    find_optimal_thresholds,
)

NUM_CLASSES = 45

EXPECTED_KEYS = {
    "f1_macro",
    "f1_micro",
    "f1_samples",
    "precision_macro",
    "recall_macro",
    "hamming_loss",
    "auc_roc",
    "mAP",
}


# ---------------------------------------------------------------------------
# Basic update and compute
# ---------------------------------------------------------------------------

def test_metric_tracker_update():
    """MetricTracker.compute() should return all expected metric keys."""
    tracker = MetricTracker(threshold=0.5)

    # Simulate 3 batches
    for _ in range(3):
        logits = torch.randn(8, NUM_CLASSES)
        targets = torch.randint(0, 2, (8, NUM_CLASSES)).float()
        tracker.update(logits, targets)

    metrics = tracker.compute()
    assert isinstance(metrics, dict)
    for key in EXPECTED_KEYS:
        assert key in metrics, f"Missing metric key: {key}"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_metric_tracker_reset():
    """After reset(), compute() should return an empty dict."""
    tracker = MetricTracker()
    tracker.update(torch.randn(4, NUM_CLASSES), torch.randint(0, 2, (4, NUM_CLASSES)).float())
    assert len(tracker.all_logits) > 0

    tracker.reset()
    assert len(tracker.all_logits) == 0
    assert len(tracker.all_targets) == 0
    metrics = tracker.compute()
    assert metrics == {}


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------

def test_metric_values_range():
    """F1, precision, recall should be in [0, 1]."""
    tracker = MetricTracker(threshold=0.5)
    for _ in range(5):
        logits = torch.randn(16, NUM_CLASSES)
        targets = torch.randint(0, 2, (16, NUM_CLASSES)).float()
        tracker.update(logits, targets)

    metrics = tracker.compute()
    for key in ["f1_macro", "f1_micro", "f1_samples", "precision_macro", "recall_macro"]:
        val = metrics[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"


# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------

def test_perfect_predictions():
    """Logits perfectly matching targets should yield F1 near 1.0."""
    tracker = MetricTracker(threshold=0.5)
    # Create known targets with reasonable class distribution
    rng = np.random.RandomState(42)
    targets_np = rng.choice([0, 1], size=(32, NUM_CLASSES), p=[0.7, 0.3]).astype(np.float32)
    targets = torch.from_numpy(targets_np)

    # Create logits that will match targets after sigmoid > 0.5
    # +5 where target is 1, -5 where target is 0
    logits = targets * 10.0 - 5.0  # 1 -> +5, 0 -> -5

    tracker.update(logits, targets)
    metrics = tracker.compute()

    assert metrics["f1_macro"] > 0.95, f"F1 macro {metrics['f1_macro']} should be near 1.0"
    assert metrics["f1_micro"] > 0.95, f"F1 micro {metrics['f1_micro']} should be near 1.0"
    assert metrics["precision_macro"] > 0.95
    assert metrics["recall_macro"] > 0.95


# ---------------------------------------------------------------------------
# Edge case: single batch
# ---------------------------------------------------------------------------

def test_single_batch_compute():
    """compute() should work with just a single update call."""
    tracker = MetricTracker()
    tracker.update(torch.randn(4, NUM_CLASSES), torch.randint(0, 2, (4, NUM_CLASSES)).float())
    metrics = tracker.compute()
    assert isinstance(metrics, dict)
    assert "f1_macro" in metrics


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

def test_threshold_sensitivity():
    """Different thresholds should produce different metric values."""
    logits = torch.randn(32, NUM_CLASSES)
    targets = torch.randint(0, 2, (32, NUM_CLASSES)).float()

    tracker_low = MetricTracker(threshold=0.3)
    tracker_low.update(logits, targets)
    metrics_low = tracker_low.compute()

    tracker_high = MetricTracker(threshold=0.7)
    tracker_high.update(logits, targets)
    metrics_high = tracker_high.compute()

    # With a lower threshold, recall should be >= higher threshold's recall
    assert metrics_low["recall_macro"] >= metrics_high["recall_macro"] - 0.01


def test_compute_metrics_with_per_class_thresholds():
    """Metric computation should accept a threshold vector."""
    y_true = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    y_prob = np.array([[0.8, 0.4], [0.4, 0.9], [0.7, 0.6]], dtype=np.float32)
    thresholds = np.array([0.75, 0.5], dtype=np.float32)

    metrics = compute_multilabel_metrics(y_true, y_prob, threshold=thresholds)

    assert metrics["f1_macro"] > 0.6
    assert metrics["threshold_mean"] == pytest.approx(0.625, abs=1e-6)


def test_find_optimal_thresholds_returns_vector():
    """Threshold search should return one threshold per class."""
    y_true = np.array(
        [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]],
        dtype=np.float32,
    )
    y_prob = np.array(
        [[0.9, 0.2], [0.8, 0.3], [0.4, 0.8], [0.3, 0.9], [0.7, 0.65]],
        dtype=np.float32,
    )

    thresholds = find_optimal_thresholds(
        y_true,
        y_prob,
        search_space=np.array([0.3, 0.5, 0.7], dtype=np.float32),
    )

    assert thresholds.shape == (2,)
    assert all(0.0 <= t <= 1.0 for t in thresholds)
