#!/usr/bin/env python3
"""Export Fundus Gate V2 learned component (MobileNetV3-Small) to ONNX INT8.

Usage:
    PYTHONPATH=. python scripts/export_fundus_gate_onnx.py \
        --weights weights/fundus_gate.pth \
        --output-dir outputs/mobile_export

Produces:
    outputs/mobile_export/
        gate_mobilenetv3_fp32.onnx    # Full precision gate model
        gate_mobilenetv3.onnx         # INT8 quantized gate model
        gate_export_report.json       # Export validation report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def build_gate_model(weights_path: str | None = None) -> nn.Module:
    """Build the MobileNetV3-Small binary classifier for fundus gating."""
    import timm

    model = timm.create_model(
        "mobilenetv3_small_100", pretrained=True, num_classes=1
    )

    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        # Handle prefix variations
        cleaned = {}
        for k, v in state.items():
            k = k.replace("backbone.", "").replace("model.", "")
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)
        logger.info("Loaded gate weights from %s", weights_path)
    else:
        logger.warning(
            "No gate weights found at %s — using pretrained ImageNet init",
            weights_path,
        )

    model.eval()
    return model


def export_gate_onnx(
    model: nn.Module, output_path: Path, img_size: int = 224
) -> Path:
    """Export gate model to ONNX."""
    dummy = torch.randn(1, 3, img_size, img_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["gate_logit"],
        dynamic_axes={"input": {0: "batch"}, "gate_logit": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
    )

    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / 1e6
    logger.info("Exported gate FP32 ONNX: %s (%.1f MB)", output_path, size_mb)
    return output_path


def quantize_gate_int8(fp32_path: Path, int8_path: Path) -> Path:
    """Dynamic INT8 quantization."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)

    size_mb = int8_path.stat().st_size / 1e6
    logger.info("Quantized gate INT8 ONNX: %s (%.1f MB)", int8_path, size_mb)
    return int8_path


def validate_gate_parity(
    torch_model: nn.Module,
    fp32_path: Path,
    int8_path: Path,
    n_samples: int = 100,
) -> dict:
    """Validate gate model parity."""
    import onnxruntime as ort

    torch_model.eval()

    fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    max_diff = 0.0
    int8_latencies = []

    for _ in range(n_samples):
        dummy = torch.randn(1, 3, 224, 224)
        dummy_np = dummy.numpy()

        with torch.no_grad():
            torch_out = torch.sigmoid(torch_model(dummy)).numpy()

        1.0 / (1.0 + np.exp(-fp32_sess.run(None, {"input": dummy_np})[0]))

        t0 = time.perf_counter()
        int8_out = 1.0 / (1.0 + np.exp(-int8_sess.run(None, {"input": dummy_np})[0]))
        int8_latencies.append((time.perf_counter() - t0) * 1000)

        max_diff = max(max_diff, np.abs(torch_out - int8_out).max())

    return {
        "n_samples": n_samples,
        "max_probability_diff": float(max_diff),
        "parity_ok": bool(max_diff < 0.05),
        "fp32_size_mb": fp32_path.stat().st_size / 1e6,
        "int8_size_mb": int8_path.stat().st_size / 1e6,
        "int8_latency_p50_ms": float(np.percentile(int8_latencies, 50)),
        "int8_latency_p95_ms": float(np.percentile(int8_latencies, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Export Fundus Gate to ONNX")
    parser.add_argument(
        "--weights", type=str, default="weights/fundus_gate.pth"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/mobile_export"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_dir = Path(args.output_dir)

    # Build and export
    model = build_gate_model(args.weights)
    fp32_path = export_gate_onnx(model, output_dir / "gate_mobilenetv3_fp32.onnx")
    int8_path = quantize_gate_int8(fp32_path, output_dir / "gate_mobilenetv3.onnx")

    # Validate
    report = validate_gate_parity(model, fp32_path, int8_path)

    report_path = output_dir / "gate_export_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nGate Export {'PASSED' if report['parity_ok'] else 'FAILED'}")
    print(f"  FP32: {report['fp32_size_mb']:.1f} MB")
    print(f"  INT8: {report['int8_size_mb']:.1f} MB")
    print(f"  Max prob diff: {report['max_probability_diff']:.4f}")
    print(f"  INT8 latency p95: {report['int8_latency_p95_ms']:.1f} ms")

    if not report["parity_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
