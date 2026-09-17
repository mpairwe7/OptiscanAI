"""Tests for Split Conformal Prediction and Risk-Controlling Triage."""


import numpy as np
import pytest

from src.evaluation.conformal_triage import ConformalRetinalTriage


@pytest.fixture
def sample_data():
    np.random.seed(42)
    classes = ["DR", "GLC", "ARMD", "HTR", "MYA"]
    n_cal = 200
    n_test = 50

    # Synthetic calibrated probabilities
    cal_probs = np.random.beta(0.5, 5.0, size=(n_cal, len(classes)))
    cal_labels = (np.random.rand(n_cal, len(classes)) < 0.25).astype(int)

    test_probs = np.random.beta(0.5, 5.0, size=(n_test, len(classes)))
    test_labels = (np.random.rand(n_test, len(classes)) < 0.25).astype(int)

    return classes, cal_probs, cal_labels, test_probs, test_labels


def test_conformal_fit_and_predict_sets(sample_data):
    classes, cal_probs, cal_labels, test_probs, _ = sample_data
    engine = ConformalRetinalTriage(class_names=classes, alpha=0.10)

    assert not engine.is_calibrated
    q_hat = engine.fit(cal_probs, cal_labels, method="joint")

    assert engine.is_calibrated
    assert 0.0 <= q_hat <= 1.0

    sets = engine.predict_sets(test_probs)
    assert len(sets) == len(test_probs)
    assert all(isinstance(s, list) for s in sets)


def test_conformal_class_stratified(sample_data):
    classes, cal_probs, cal_labels, test_probs, _ = sample_data
    engine = ConformalRetinalTriage(class_names=classes, alpha=0.05)
    engine.fit(cal_probs, cal_labels, method="class_stratified")

    assert engine.class_q_hats is not None
    assert len(engine.class_q_hats) == len(classes)
    sets = engine.predict_sets(test_probs)
    assert len(sets) == len(test_probs)


def test_conformal_triage_decisions_and_abstention(sample_data):
    classes, cal_probs, cal_labels, _, _ = sample_data
    engine = ConformalRetinalTriage(class_names=classes, alpha=0.05, max_set_size_for_auto_triage=2)
    engine.fit(cal_probs, cal_labels)

    # 1. Clear negative case (zeros)
    zero_prob = np.zeros((1, len(classes)))
    decisions = engine.evaluate_triage(zero_prob)
    assert len(decisions) == 1
    assert decisions[0].triage_level == "ROUTINE"
    assert not decisions[0].abstain

    # 2. Extreme ambiguous case (all high probabilities -> large set)
    high_prob = np.ones((1, len(classes)))
    decisions_high = engine.evaluate_triage(high_prob)
    assert len(decisions_high) == 1
    assert decisions_high[0].set_size == len(classes)
    assert decisions_high[0].triage_level == "ABSTAIN_MANDATORY_REVIEW"
    assert decisions_high[0].abstain


def test_conformal_coverage_evaluation(sample_data):
    classes, cal_probs, cal_labels, test_probs, test_labels = sample_data
    engine = ConformalRetinalTriage(class_names=classes, alpha=0.10)
    engine.fit(cal_probs, cal_labels)

    metrics = engine.evaluate_coverage(test_probs, test_labels)
    assert "empirical_coverage" in metrics
    assert "average_set_size" in metrics
    assert 0.0 <= metrics["empirical_coverage"] <= 1.0


def test_conformal_save_and_load(tmp_path, sample_data):
    classes, cal_probs, cal_labels, test_probs, _ = sample_data
    engine = ConformalRetinalTriage(class_names=classes, alpha=0.05)
    engine.fit(cal_probs, cal_labels)

    save_path = tmp_path / "conformal_calibration.json"
    engine.save(save_path)
    assert save_path.exists()

    loaded = ConformalRetinalTriage.load(save_path)
    assert loaded.is_calibrated
    assert loaded.alpha == engine.alpha
    assert np.isclose(loaded.q_hat, engine.q_hat)

    # Verify predict sets match
    orig_sets = engine.predict_sets(test_probs)
    loaded_sets = loaded.predict_sets(test_probs)
    assert orig_sets == loaded_sets
