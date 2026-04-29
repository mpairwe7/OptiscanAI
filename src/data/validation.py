"""Data validation for retinal disease classification pipeline.
Validates schema, distributions, and quality of training data.
Inspired by Great Expectations patterns."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    passed: bool
    check_name: str
    details: str
    severity: str = "error"  # error | warning | info

@dataclass
class ValidationReport:
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        return {"total": total, "passed": passed, "failed": failed, "pass_rate": passed/max(total,1)}

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "checks": [{"check": r.check_name, "passed": r.passed, "details": r.details, "severity": r.severity} for r in self.results]
        }

class DataValidator:
    """Validates training data quality for retinal disease classification."""

    def __init__(self, disease_columns: list[str]):
        self.disease_columns = disease_columns

    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Check that DataFrame has required columns."""
        missing = [c for c in ["ID"] + self.disease_columns if c not in df.columns]
        passed = len(missing) == 0
        return ValidationResult(
            passed=passed,
            check_name="schema_validation",
            details=f"Missing columns: {missing}" if not passed else "All required columns present"
        )

    def validate_label_range(self, df: pd.DataFrame) -> ValidationResult:
        """Check labels are binary (0 or 1)."""
        label_data = df[self.disease_columns]
        unique_vals = set()
        for col in self.disease_columns:
            unique_vals.update(df[col].dropna().unique().tolist())
        invalid = unique_vals - {0, 1, 0.0, 1.0}
        passed = len(invalid) == 0
        return ValidationResult(
            passed=passed,
            check_name="label_range",
            details=f"Invalid values found: {invalid}" if not passed else "All labels are binary"
        )

    def validate_no_null_ids(self, df: pd.DataFrame) -> ValidationResult:
        """Check no null image IDs."""
        null_count = df["ID"].isna().sum()
        passed = null_count == 0
        return ValidationResult(
            passed=passed,
            check_name="null_id_check",
            details=f"{null_count} null IDs found" if not passed else "No null IDs"
        )

    def validate_class_distribution(self, df: pd.DataFrame, min_samples: int = 1) -> ValidationResult:
        """Check each class has minimum samples."""
        class_counts = df[self.disease_columns].sum()
        empty_classes = class_counts[class_counts < min_samples].index.tolist()
        passed = len(empty_classes) == 0
        return ValidationResult(
            passed=passed,
            check_name="class_distribution",
            details=f"Classes with <{min_samples} samples: {empty_classes}" if not passed else "All classes have sufficient samples",
            severity="warning"
        )

    def validate_no_duplicate_ids(self, df: pd.DataFrame) -> ValidationResult:
        """Check for duplicate image IDs."""
        dup_count = df["ID"].duplicated().sum()
        passed = dup_count == 0
        return ValidationResult(
            passed=passed,
            check_name="duplicate_id_check",
            details=f"{dup_count} duplicate IDs" if not passed else "No duplicate IDs"
        )

    def validate_label_correlations(self, df: pd.DataFrame, max_correlation: float = 0.99) -> ValidationResult:
        """Check for suspiciously high label correlations (potential leakage)."""
        corr_matrix = df[self.disease_columns].corr()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.abs().max().max()
        passed = max_corr < max_correlation
        return ValidationResult(
            passed=passed,
            check_name="label_correlation",
            details=f"Max label correlation: {max_corr:.4f}" if not passed else f"Max correlation {max_corr:.4f} within bounds",
            severity="warning"
        )

    def validate_image_directory(self, img_dir: Path, df: pd.DataFrame, extensions: list[str] = None) -> ValidationResult:
        """Check that images exist for all IDs."""
        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg"]
        missing = []
        for img_id in df["ID"].values[:100]:  # Check first 100 for speed
            found = any((img_dir / f"{img_id}{ext}").exists() for ext in extensions)
            if not found:
                missing.append(img_id)
        passed = len(missing) == 0
        return ValidationResult(
            passed=passed,
            check_name="image_existence",
            details=f"{len(missing)} images missing (checked first 100)" if not passed else "All checked images exist"
        )

    def validate_all(self, df: pd.DataFrame, img_dir: Path = None) -> ValidationReport:
        """Run all validation checks."""
        report = ValidationReport()
        report.results.append(self.validate_schema(df))
        report.results.append(self.validate_label_range(df))
        report.results.append(self.validate_no_null_ids(df))
        report.results.append(self.validate_class_distribution(df))
        report.results.append(self.validate_no_duplicate_ids(df))
        report.results.append(self.validate_label_correlations(df))
        if img_dir:
            report.results.append(self.validate_image_directory(img_dir, df))

        logger.info(f"Data validation: {report.summary}")
        return report
