#!/usr/bin/env python3
"""Generate model card and dataset card for the trained model."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.datamodule import DISEASE_COLUMNS, RetinalDataModule
from src.governance.dataset_card import DatasetCard
from src.governance.model_card import generate_model_card
from train import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--metrics", default="outputs/evaluation_metrics.json")
    parser.add_argument("--output-dir", default="outputs/governance")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)

    # Load metrics if available
    metrics = {}
    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    # Count parameters
    model = build_model(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    del model

    # Generate model card
    model_name = cfg["model"]["name"]
    card = generate_model_card(
        model_name=f"Retinal-{model_name.upper()}",
        version=cfg.get("version", "2.0.0"),
        architecture=f"{model_name} with clinical knowledge graph",
        num_parameters=num_params,
        metrics=metrics,
    )
    card.to_json(out_dir / "model_card.json")
    card.to_markdown(out_dir / "MODEL_CARD.md")

    # Generate dataset card
    dataset_card = DatasetCard()
    try:
        dm = RetinalDataModule(cfg)
        dm.prepare_data()
        dm.setup(stage="fit")
        dataset_card.populate_from_dataframe(dm.train_dataset.labels_df, DISEASE_COLUMNS)
    except Exception as e:
        logger.warning(f"Could not populate dataset stats: {e}")

    dataset_card.to_json(out_dir / "dataset_card.json")
    dataset_card.to_markdown(out_dir / "DATASET_CARD.md")

    logger.info(f"Governance documents saved to {out_dir}")


if __name__ == "__main__":
    main()
