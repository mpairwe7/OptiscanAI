#!/usr/bin/env python3
"""Evaluate a trained model and output metrics JSON for DVC tracking."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.data.datamodule import RetinalDataModule
from src.evaluation.calibration import (
    TemperatureScaler,
    bootstrap_confidence_interval,
    compute_ece,
)
from src.evaluation.evaluator import ModelEvaluator
from src.training.metrics import compute_multilabel_metrics
from train import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dm = RetinalDataModule(cfg)
    dm.prepare_data()
    dm.setup(None)
    cfg["model"]["num_classes"] = len(dm.disease_columns)
    cfg["model"]["disease_names"] = dm.disease_columns

    # Model
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Evaluate
    thresholds = ckpt.get(
        "decision_thresholds",
        cfg.get("evaluation", {}).get("threshold", 0.5),
    )
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        disease_names=dm.disease_columns,
        threshold=thresholds,
    )
    eval_loader = dm.test_dataloader() if dm.test_dataset is not None else dm.val_dataloader()
    results = evaluator.evaluate(eval_loader)

    cal_cfg = cfg.get("calibration", {})
    ece, _ = compute_ece(
        results["y_prob"],
        results["y_true"],
        n_bins=cal_cfg.get("ece_bins", 15),
    )
    results["metrics"]["ece"] = ece

    if cal_cfg.get("temperature_scaling", False) and dm.test_dataset is not None:
        scaler = TemperatureScaler()
        temperature = scaler.calibrate(model, dm.val_dataloader(), device)

        class CalibratedModel(torch.nn.Module):
            def __init__(self, base_model, temp_scaler):
                super().__init__()
                self.base_model = base_model
                self.temp_scaler = temp_scaler

            def forward(self, x):
                return self.temp_scaler(self.base_model(x))

        calibrated_model = CalibratedModel(model, scaler).to(device)
        results = ModelEvaluator(
            model=calibrated_model,
            device=device,
            disease_names=dm.disease_columns,
            threshold=thresholds,
        ).evaluate(eval_loader)
        calibrated_ece, _ = compute_ece(
            results["y_prob"],
            results["y_true"],
            n_bins=cal_cfg.get("ece_bins", 15),
        )
        results["metrics"]["ece"] = calibrated_ece
        results["metrics"]["temperature"] = temperature

    if cal_cfg.get("bootstrap_ci", {}).get("enabled", False):
        ci_cfg = cal_cfg["bootstrap_ci"]
        _, lower, upper = bootstrap_confidence_interval(
            lambda y_true, y_prob: compute_multilabel_metrics(y_true, y_prob, threshold=thresholds)[
                "f1_macro"
            ],
            results["y_true"],
            results["y_prob"],
            n_bootstrap=ci_cfg.get("n_bootstrap", 1000),
            ci=ci_cfg.get("confidence", 0.95),
        )
        results["metrics"]["f1_macro_ci_lower"] = lower
        results["metrics"]["f1_macro_ci_upper"] = upper

    # Save metrics
    out_path = Path("outputs/evaluation_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results["metrics"], indent=2))
    logger.info(f"Evaluation metrics saved to {out_path}")
    logger.info(f"Results: {json.dumps(results['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
