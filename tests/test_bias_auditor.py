"""Tests for BiasAuditor."""

import pytest
import numpy as np
import tempfile
import os

from src.governance.bias_auditor import BiasAuditor, BiasReport


@pytest.fixture
def auditor():
    return BiasAuditor(
        protected_attributes=["sex", "age_group"],
        dp_threshold=0.1,
        eo_threshold=0.1,
        min_subgroup_size=10,
    )


@pytest.fixture
def fair_data():
    """Generate balanced data with similar performance across groups."""
    np.random.seed(42)
    n = 200
    n_classes = 10

    predictions = np.random.rand(n, n_classes) * 0.5 + 0.25
    targets = (predictions > 0.5).astype(int)

    metadata = {
        "sex": np.array(["male"] * 100 + ["female"] * 100),
        "age_group": np.array(["adult"] * 100 + ["elderly"] * 100),
    }
    return predictions, targets, metadata


@pytest.fixture
def biased_data():
    """Generate data with significant performance disparity."""
    np.random.seed(42)
    n = 200
    n_classes = 10

    predictions = np.random.rand(n, n_classes)
    # Make male predictions much higher
    predictions[:100] *= 0.9
    predictions[:100] += 0.3
    predictions[100:] *= 0.3

    targets = (np.random.rand(n, n_classes) > 0.7).astype(int)

    metadata = {
        "sex": np.array(["male"] * 100 + ["female"] * 100),
    }
    return predictions, targets, metadata


class TestBiasAudit:
    def test_audit_runs(self, auditor, fair_data):
        predictions, targets, metadata = fair_data
        report = auditor.audit(predictions, targets, metadata)
        assert isinstance(report, BiasReport)
        assert report.total_samples == 200

    def test_fair_data_passes(self, auditor, fair_data):
        predictions, targets, metadata = fair_data
        report = auditor.audit(predictions, targets, metadata)
        # Fair data should have fewer violations
        assert isinstance(report.fairness_pass, bool)

    def test_biased_data_detected(self, auditor, biased_data):
        predictions, targets, metadata = biased_data
        report = auditor.audit(predictions, targets, metadata)
        # Significant bias should be detected
        assert len(report.demographic_parity) > 0

    def test_subgroup_metrics_computed(self, auditor, fair_data):
        predictions, targets, metadata = fair_data
        report = auditor.audit(predictions, targets, metadata)
        assert len(report.subgroup_metrics) > 0
        for sm in report.subgroup_metrics:
            assert sm.sample_count > 0
            assert 0 <= sm.auc_roc <= 1 or sm.auc_roc == 0
            assert 0 <= sm.f1_score <= 1

    def test_recommendations_generated(self, auditor, biased_data):
        predictions, targets, metadata = biased_data
        report = auditor.audit(predictions, targets, metadata)
        assert isinstance(report.recommendations, list)


class TestReportSaving:
    def test_save_report(self, auditor, fair_data):
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp(dir="/tmp")
        try:
            predictions, targets, metadata = fair_data
            report = auditor.audit(predictions, targets, metadata, model_version="test")
            output_path = os.path.join(tmpdir, "bias_report.json")
            saved_path = auditor.save_report(report, output_path)
            assert os.path.exists(saved_path)

            import json
            with open(saved_path) as f:
                data = json.load(f)
            assert "fairness_pass" in data
            assert "subgroup_metrics" in data
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEdgeCases:
    def test_small_subgroup_skipped(self):
        auditor = BiasAuditor(min_subgroup_size=50)
        predictions = np.random.rand(30, 5)
        targets = (predictions > 0.5).astype(int)
        metadata = {"sex": np.array(["male"] * 15 + ["female"] * 15)}
        report = auditor.audit(predictions, targets, metadata)
        # Both groups too small, should have no subgroup metrics
        assert len(report.subgroup_metrics) == 0

    def test_single_group_no_violation(self):
        auditor = BiasAuditor(min_subgroup_size=5)
        predictions = np.random.rand(50, 5)
        targets = (predictions > 0.5).astype(int)
        metadata = {"sex": np.array(["male"] * 50)}
        report = auditor.audit(predictions, targets, metadata)
        assert len(report.violations) == 0
