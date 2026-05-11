#!/usr/bin/env python3
"""Generate Uganda Ministry of Health regulatory submission package.

Usage:
    PYTHONPATH=. python scripts/generate_moh_package.py
    PYTHONPATH=. python scripts/generate_moh_package.py --output-dir outputs/moh_submission

Produces:
    outputs/moh_submission/
        moh_submission_package.json    # Complete regulatory package
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Uganda MoH regulatory submission package"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/moh_submission",
        help="Output directory for submission package",
    )
    parser.add_argument(
        "--clinical-metrics",
        type=str,
        default=None,
        help="Path to clinical validation metrics JSON",
    )
    parser.add_argument(
        "--bias-report",
        type=str,
        default=None,
        help="Path to bias audit report JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Load optional inputs
    import json

    clinical_metrics = None
    if args.clinical_metrics and Path(args.clinical_metrics).exists():
        with open(args.clinical_metrics) as f:
            clinical_metrics = json.load(f)
        logger.info("Loaded clinical metrics from %s", args.clinical_metrics)

    bias_report = None
    if args.bias_report and Path(args.bias_report).exists():
        with open(args.bias_report) as f:
            bias_report = json.load(f)
        logger.info("Loaded bias report from %s", args.bias_report)

    from src.governance.moh_submission import generate_submission_package

    output_path = generate_submission_package(
        output_dir=args.output_dir,
        clinical_metrics=clinical_metrics,
        bias_report=bias_report,
    )

    print(f"\nMoH submission package generated: {output_path}")
    print(f"  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
