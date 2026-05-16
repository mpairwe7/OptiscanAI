#!/usr/bin/env python3
"""Run data validation checks before training."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.datamodule import DISEASE_COLUMNS, RetinalDataModule
from src.data.validation import DataValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Initialize data module to get paths
    dm = RetinalDataModule(cfg)
    dm.prepare_data()
    dm.setup(stage="fit")

    validator = DataValidator(disease_columns=DISEASE_COLUMNS)

    # Validate training data
    report = validator.validate_all(
        df=dm.train_dataset.labels_df,
        img_dir=Path(dm.train_dataset.img_dir),
    )

    # Save report
    out_dir = Path("outputs/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "data_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info(f"Validation report saved to {report_path}")

    if not report.passed:
        logger.error("Data validation FAILED — check report for details")
        sys.exit(1)
    else:
        logger.info("Data validation PASSED")


if __name__ == "__main__":
    main()
