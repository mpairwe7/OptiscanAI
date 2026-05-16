#!/usr/bin/env python3
"""Check if model retraining is needed based on drift and other triggers."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.retraining import RetrainingTrigger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=None, help="Path to current metrics JSON")
    parser.add_argument("--output", default="outputs/retraining_decision.json")
    args = parser.parse_args()

    trigger = RetrainingTrigger()

    current_metrics = None
    if args.metrics and Path(args.metrics).exists():
        current_metrics = json.loads(Path(args.metrics).read_text())

    decision = trigger.evaluate(current_metrics=current_metrics)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision.to_dict(), indent=2))

    if decision.should_retrain:
        logger.warning(f"RETRAINING RECOMMENDED ({decision.priority}): {decision.reason}")
        sys.exit(2)  # Exit code 2 = retrain needed
    else:
        logger.info("No retraining needed")
        sys.exit(0)


if __name__ == "__main__":
    main()
