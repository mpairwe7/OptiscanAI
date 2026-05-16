#!/usr/bin/env python3
"""Run hyperparameter optimization."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.training.hpo import run_hpo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--output", default="configs/train_optimized.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    best_cfg = run_hpo(cfg, n_trials=args.n_trials, timeout_seconds=args.timeout)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Optimized config saved to {out_path}")


if __name__ == "__main__":
    main()
