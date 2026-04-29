"""Tests for data validation: schema, label range, null checks, class distribution."""
import sys

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.validation import DataValidator, ValidationResult, ValidationReport
from src.data.datamodule import DISEASE_COLUMNS


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_validate_schema(sample_labels_df, disease_columns):
    """DataFrame should have ID column and all 45 disease columns."""
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_schema(sample_labels_df)
    assert isinstance(result, ValidationResult)
    assert result.passed is True
    assert result.check_name == "schema_validation"


def test_validate_schema_missing_column(disease_columns):
    """Validation should detect missing required columns."""
    df = pd.DataFrame({"ID": [1, 2], "DR": [0, 1]})
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_schema(df)
    assert result.passed is False
    assert "Missing" in result.details


# ---------------------------------------------------------------------------
# Label range validation
# ---------------------------------------------------------------------------

def test_validate_label_range(sample_labels_df, disease_columns):
    """All label values should be 0 or 1."""
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_label_range(sample_labels_df)
    assert isinstance(result, ValidationResult)
    assert result.passed is True


def test_validate_label_range_invalid(disease_columns):
    """Should detect labels outside {0, 1}."""
    data = {"ID": [1, 2, 3]}
    for col in disease_columns:
        data[col] = [0, 1, 0]
    df = pd.DataFrame(data)
    # Inject invalid value
    df.loc[0, disease_columns[0]] = 2
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_label_range(df)
    assert result.passed is False
    assert "Invalid" in result.details


# ---------------------------------------------------------------------------
# Null image ID validation
# ---------------------------------------------------------------------------

def test_validate_no_null_images(sample_labels_df, disease_columns):
    """No image IDs should be null."""
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_no_null_ids(sample_labels_df)
    assert bool(result.passed) is True


def test_validate_null_images_detected(disease_columns):
    """Should detect null image IDs."""
    data = {"ID": ["img_0", None, "img_2"]}
    for col in disease_columns:
        data[col] = [0, 1, 0]
    df = pd.DataFrame(data)
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_no_null_ids(df)
    assert bool(result.passed) is False
    assert "null" in result.details.lower()


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def test_validate_class_distribution(sample_labels_df, disease_columns):
    """No disease class should have 0 total samples (conftest ensures >= 1)."""
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_class_distribution(sample_labels_df, min_samples=1)
    assert isinstance(result, ValidationResult)
    assert result.passed is True


def test_validate_class_distribution_empty_class(disease_columns):
    """Should detect classes with zero samples."""
    data = {"ID": list(range(10))}
    for col in disease_columns:
        data[col] = [0] * 10  # All zeros
    df = pd.DataFrame(data)
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_class_distribution(df, min_samples=1)
    assert result.passed is False


# ---------------------------------------------------------------------------
# Duplicate ID validation
# ---------------------------------------------------------------------------

def test_validate_no_duplicate_ids(sample_labels_df, disease_columns):
    """No duplicate IDs should exist."""
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_no_duplicate_ids(sample_labels_df)
    assert bool(result.passed) is True


def test_validate_duplicate_ids_detected(disease_columns):
    """Should detect duplicate IDs."""
    data = {"ID": ["img_0", "img_0", "img_2"]}
    for col in disease_columns:
        data[col] = [0, 1, 0]
    df = pd.DataFrame(data)
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_no_duplicate_ids(df)
    assert bool(result.passed) is False


# ---------------------------------------------------------------------------
# Full validation report
# ---------------------------------------------------------------------------

def test_validation_report(sample_labels_df, sample_img_dir, disease_columns):
    """Full validate_all should produce a valid report dict."""
    validator = DataValidator(disease_columns=disease_columns)
    report = validator.validate_all(sample_labels_df, img_dir=Path(sample_img_dir))

    assert isinstance(report, ValidationReport)
    assert report.passed is True
    summary = report.summary
    assert "total" in summary
    assert "passed" in summary
    assert "failed" in summary
    assert "pass_rate" in summary
    assert summary["total"] > 0
    assert summary["pass_rate"] > 0.0


def test_validation_report_to_dict(sample_labels_df, disease_columns):
    """to_dict should produce a serializable dict with all check results."""
    validator = DataValidator(disease_columns=disease_columns)
    report = validator.validate_all(sample_labels_df)
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "passed" in d
    assert "summary" in d
    assert "checks" in d
    assert isinstance(d["checks"], list)
    assert len(d["checks"]) > 0


def test_validation_report_without_img_dir(sample_labels_df, disease_columns):
    """validate_all without img_dir should skip image existence check."""
    validator = DataValidator(disease_columns=disease_columns)
    report = validator.validate_all(sample_labels_df, img_dir=None)
    check_names = [r.check_name for r in report.results]
    assert "image_existence" not in check_names


def test_label_correlation_check(sample_labels_df, disease_columns):
    """Label correlation check should run without error."""
    validator = DataValidator(disease_columns=disease_columns)
    result = validator.validate_label_correlations(sample_labels_df)
    assert isinstance(result, ValidationResult)
    assert result.check_name == "label_correlation"
