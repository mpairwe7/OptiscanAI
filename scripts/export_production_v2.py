#!/usr/bin/env python3
"""
Production export for RetinalFoundationHybridV2 with dtype safety.

Fixes the float/half mismatch that caused quantization failures in v1.
All tensors are explicitly cast to float32 before export.

Usage:
    python scripts/export_production_v2.py --checkpoint outputs/checkpoints/v2/final_with_thresholds.pth
    python scripts/export_production_v2.py --checkpoint best.pth --formats onnx torchscript --quantize
"""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def ensure_float32(model: nn.Module) -> nn.Module:
    """Ensure all model parameters and buffers are float32.

    This fixes the float/half dtype mismatch that causes ONNX and
    quantization failures.
    """
    model = model.float()
    for name, buf in model.named_buffers():
        if buf.dtype == torch.float16 or buf.dtype == torch.bfloat16:
            buf.data = buf.data.float()
    return model


def export_onnx_safe(
    model: nn.Module,
    output_path: str,
    opset_version: int = 18,
    input_shape: tuple = (1, 3, 224, 224),
) -> str:
    """Export to ONNX with proper dtype handling."""
    model = ensure_float32(model).cpu().eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32)

    # Get reference output
    with torch.no_grad():
        ref_output = model(dummy)
        if isinstance(ref_output, dict):
            ref_output = ref_output.get("logits", list(ref_output.values())[0])

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"ONNX exported: {output_path} ({size_mb:.1f} MB, opset {opset_version})")

    # Verify
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path)
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
        max_diff = np.abs(ref_output.numpy() - onnx_out).max()
        logger.info(f"ONNX verification: max diff = {max_diff:.6f}")
        if max_diff > 1e-4:
            logger.warning(f"ONNX output diverges (max_diff={max_diff:.4f})")
    except ImportError:
        logger.info("onnxruntime not installed; skipping verification")

    return output_path


def export_torchscript_safe(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
) -> str:
    """Export to TorchScript with dtype safety."""
    model = ensure_float32(model).cpu().eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32)

    with torch.no_grad():
        ref_output = model(dummy)
        if isinstance(ref_output, dict):
            ref_output = ref_output.get("logits", list(ref_output.values())[0])

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    traced = torch.jit.trace(model, dummy)
    traced.save(output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"TorchScript exported: {output_path} ({size_mb:.1f} MB)")

    # Verify
    loaded = torch.jit.load(output_path)
    with torch.no_grad():
        ts_out = loaded(dummy)
        if isinstance(ts_out, dict):
            ts_out = ts_out.get("logits", list(ts_out.values())[0])
        max_diff = (ref_output - ts_out).abs().max().item()
        logger.info(f"TorchScript verification: max diff = {max_diff:.6f}")

    return output_path


def quantize_int8_safe(model: nn.Module, output_path: str) -> str:
    """Dynamic INT8 quantization with float32 pre-casting."""
    model = ensure_float32(model).cpu().eval()

    quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={nn.Linear}, dtype=torch.qint8
    )

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    torch.save(quantized.state_dict(), output_path)

    # Measure size
    import io

    buf = io.BytesIO()
    torch.save(quantized.state_dict(), buf)
    size_mb = buf.tell() / (1024 * 1024)
    logger.info(f"INT8 quantized: {output_path} ({size_mb:.1f} MB)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export HybridV2 for production")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--formats", nargs="+", default=["onnx", "torchscript"])
    parser.add_argument("--output-dir", type=str, default="outputs/export_v2")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--backbone", type=str, default="vit_large_patch16_224")
    parser.add_argument("--num-classes", type=int, default=28)
    args = parser.parse_args()

    from src.models.retinal_foundation_hybrid_v2 import create_hybrid_v2
    from src.models.vignn import ClinicalKnowledgeGraph

    # Build model
    # Use a minimal disease list for the factory
    disease_names = [
        "DR",
        "ARMD",
        "MH",
        "DN",
        "MYA",
        "BRVO",
        "TSLN",
        "ERM",
        "LS",
        "MS",
        "CSR",
        "ODC",
        "CRVO",
        "TV",
        "AH",
        "ODP",
        "ODE",
        "ST",
        "AION",
        "PT",
        "RT",
        "RS",
        "CRS",
        "EDN",
        "RPEC",
        "MHL",
        "RP",
        "CWS",
    ][: args.num_classes]
    kg = ClinicalKnowledgeGraph(disease_names=disease_names)

    model = create_hybrid_v2(
        num_classes=args.num_classes,
        clinical_knowledge_graph=kg,
        backbone=args.backbone,
        use_lora=True,
        checkpoint_path=args.checkpoint,
    )

    # Merge LoRA for export
    model.prepare_for_export()

    os.makedirs(args.output_dir, exist_ok=True)

    # Export
    if "onnx" in args.formats:
        export_onnx_safe(model, os.path.join(args.output_dir, "model.onnx"))

    if "torchscript" in args.formats:
        export_torchscript_safe(model, os.path.join(args.output_dir, "model.pt"))

    if args.quantize:
        quantize_int8_safe(
            copy.deepcopy(model),
            os.path.join(args.output_dir, "model_int8.pth"),
        )

    logger.info("Export complete.")


if __name__ == "__main__":
    main()
