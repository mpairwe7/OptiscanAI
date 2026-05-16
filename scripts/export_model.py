#!/usr/bin/env python3
"""Export trained model to ONNX and TorchScript formats."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

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

    device = torch.device("cpu")  # Export on CPU for compatibility

    # Load model
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    export_cfg = cfg.get("export", {})
    out_dir = Path("outputs/export")
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 224, 224)
    manifest = {
        "model": cfg["model"]["name"],
        "num_classes": cfg["model"]["num_classes"],
        "formats": [],
        "decision_thresholds": ckpt.get("decision_thresholds"),
    }

    # ONNX export
    if export_cfg.get("onnx", {}).get("enabled", False):
        onnx_path = out_dir / "model.onnx"
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                opset_version=export_cfg["onnx"].get("opset_version", 17),
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            )
            manifest["formats"].append("onnx")
            logger.info(f"ONNX exported to {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    # TorchScript export
    if export_cfg.get("torchscript", {}).get("enabled", False):
        ts_path = out_dir / "model.pt"
        try:
            scripted = torch.jit.trace(model, dummy_input)
            scripted.save(str(ts_path))
            manifest["formats"].append("torchscript")
            logger.info(f"TorchScript exported to {ts_path}")
        except Exception as e:
            logger.warning(f"TorchScript export failed: {e}")

    # Save manifest
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    if ckpt.get("decision_thresholds") is not None:
        (out_dir / "thresholds.json").write_text(json.dumps(ckpt["decision_thresholds"], indent=2))
    logger.info(f"Export complete: {manifest['formats']}")


if __name__ == "__main__":
    main()
