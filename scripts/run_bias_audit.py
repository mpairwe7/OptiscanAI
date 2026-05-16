#!/usr/bin/env python3
"""
Automated bias audit report generator.

Usage:
    python scripts/run_bias_audit.py --checkpoint best.pth --data-dir data/rfmid
    python scripts/run_bias_audit.py --checkpoint best.pth --output outputs/bias_report.json
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.governance.audit_logger import ImmutableAuditLogger
from src.governance.bias_auditor import BiasAuditor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run bias audit")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/bias_report.json")
    parser.add_argument("--dp-threshold", type=float, default=0.1)
    parser.add_argument("--eo-threshold", type=float, default=0.1)
    parser.add_argument(
        "--uganda",
        action="store_true",
        help="Run Uganda-specific bias audit with device/lighting/geographic subgroups",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.08,
        help="Max F1 disparity across subgroups (Uganda audit, default 0.08)",
    )
    args = parser.parse_args()

    if args.uganda:
        from src.governance.bias_auditor import UgandaBiasAuditor

        auditor = UgandaBiasAuditor(f1_disparity_threshold=args.f1_threshold)
        logger.info("Running Uganda-specific bias audit (F1 threshold=%.2f)", args.f1_threshold)
    else:
        auditor = BiasAuditor(
            dp_threshold=args.dp_threshold,
            eo_threshold=args.eo_threshold,
        )

    # Generate synthetic metadata for demonstration
    # In production, this comes from patient records / image metadata
    n_samples = 1000
    n_classes = 45

    logger.info("Generating synthetic audit data for demonstration...")
    predictions = np.random.rand(n_samples, n_classes) * 0.8
    targets = (np.random.rand(n_samples, n_classes) > 0.85).astype(int)

    metadata = {
        "age_group": np.random.choice(["pediatric", "adult", "elderly"], n_samples),
        "sex": np.random.choice(["male", "female"], n_samples),
        "camera_device": np.random.choice(["Topcon", "Canon", "Zeiss", "Mobile"], n_samples),
    }

    # Run audit
    report = auditor.audit(
        predictions=predictions,
        targets=targets,
        metadata=metadata,
        model_version=args.checkpoint or "demo",
        dataset_name="RFMiD-synthetic",
    )

    # Save report
    auditor.save_report(report, args.output)

    # Log to audit trail
    audit_logger = ImmutableAuditLogger()
    audit_logger.log_bias_audit(
        model_version=report.model_version,
        passed=report.fairness_pass,
        violations=report.violations,
        metrics_summary={
            "demographic_parity": report.demographic_parity,
            "equalized_odds": report.equalized_odds,
        },
    )

    # Print summary
    print("\nBias Audit Report")
    print(f"{'='*50}")
    print(f"Result: {'PASS' if report.fairness_pass else 'FAIL'}")
    print(f"Violations: {len(report.violations)}")
    for v in report.violations:
        print(f"  - {v}")
    print("Recommendations:")
    for r in report.recommendations:
        print(f"  - {r}")
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
