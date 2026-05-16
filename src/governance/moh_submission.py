"""Uganda Ministry of Health regulatory submission package generator.

Generates the complete documentation package required for MoH technical
review and national scale-up consideration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ClinicalValidationSummary:
    """Clinical performance metrics for MoH review."""

    dataset: str = "RFMiD v2 (1920 samples)"
    num_classes: int = 28
    total_samples: int = 1920
    sensitivity_macro: float = 0.0
    specificity_macro: float = 0.0
    precision_macro: float = 0.0
    f1_macro: float = 0.0
    auc_macro: float = 0.0
    per_class_metrics: dict = field(default_factory=dict)


@dataclass
class DeviceCompatibility:
    """Device compatibility matrix."""

    tested_devices: list[str] = field(default_factory=list)
    min_android_version: str = "12"
    min_ram_gb: int = 4
    requires_gps: bool = False
    requires_play_services: bool = False
    offline_capable: bool = True


@dataclass
class MoHSubmissionPackage:
    """Complete MoH regulatory submission package."""

    submission_date: str = ""
    product_name: str = "RetinalAI Clinical Screening Platform"
    product_version: str = "1.0.0"
    manufacturer: str = ""
    intended_use: str = (
        "AI-assisted screening for retinal diseases in primary healthcare "
        "settings by trained community health workers. Not a diagnostic device."
    )
    target_population: str = "Adults in Uganda requiring retinal screening"
    classification: str = "IEC 62304 Class B (non-serious injury potential with mitigation)"
    clinical_validation: ClinicalValidationSummary = field(
        default_factory=ClinicalValidationSummary
    )
    device_compatibility: DeviceCompatibility = field(default_factory=DeviceCompatibility)
    language_support: list[str] = field(default_factory=lambda: ["English", "Luganda"])
    bias_audit_passed: bool = False
    max_f1_disparity: float = 0.0
    pdp_act_compliant: bool = True
    data_governance: dict = field(
        default_factory=lambda: {
            "consent_required": True,
            "data_retention_days": 730,
            "cross_border_restrictions": "Data stored in Uganda. Transfer only to EAC countries.",
            "anonymization": "PII stripped for aggregate reporting and research",
            "audit_trail": "SHA-256 hash-chain audit for all predictions",
        }
    )
    post_market_plan: dict = field(
        default_factory=lambda: {
            "monitoring": "Continuous drift detection, fairness monitoring, and bias auditing",
            "update_mechanism": "Delta sync for model and threshold updates",
            "incident_reporting": "Automated alerts for performance degradation",
            "re_validation_schedule": "Quarterly clinical validation review",
        }
    )

    def __post_init__(self):
        if not self.submission_date:
            self.submission_date = date.today().isoformat()


def generate_submission_package(
    output_dir: str = "outputs/moh_submission",
    clinical_metrics: Optional[dict] = None,
    bias_report: Optional[dict] = None,
) -> Path:
    """Generate the full MoH submission package."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    package = MoHSubmissionPackage()

    if clinical_metrics:
        package.clinical_validation = ClinicalValidationSummary(
            **{
                k: v
                for k, v in clinical_metrics.items()
                if k in ClinicalValidationSummary.__dataclass_fields__
            }
        )

    if bias_report:
        package.bias_audit_passed = bias_report.get("max_f1_disparity", 1.0) < 0.08
        package.max_f1_disparity = bias_report.get("max_f1_disparity", 0.0)

    package.device_compatibility = DeviceCompatibility(
        tested_devices=[
            "Tecno Spark 10",
            "Tecno Camon 20",
            "Infinix Hot 30",
            "Samsung Galaxy A14",
            "Clinical fundus camera",
        ],
    )

    # Write package
    pkg_path = output_path / "moh_submission_package.json"
    with open(pkg_path, "w") as f:
        json.dump(asdict(package), f, indent=2, default=str)

    logger.info("MoH submission package saved to %s", pkg_path)
    return pkg_path
