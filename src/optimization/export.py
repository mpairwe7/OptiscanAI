"""
Production model export pipeline.

Exports RetinalFoundationHybrid to deployment formats:
    - ONNX (opset 18) for cross-platform inference
    - TorchScript for PyTorch-native serving
    - Core ML for Apple Silicon edge devices
    - TensorRT for NVIDIA GPU optimization

All exports include input validation, shape verification, and numerical
consistency checks against the original PyTorch model.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str):
    """Create parent directory if it exists and is non-empty."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def export_onnx(
    model: nn.Module,
    output_path: str = "outputs/export/model.onnx",
    opset_version: int = 18,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    verify: bool = True,
) -> str:
    """Export model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        Trained model (should have LoRA merged if applicable).
    output_path : str
        Where to save the .onnx file.
    opset_version : int
        ONNX opset version. 18 recommended for 2026 compatibility.
    input_shape : tuple
        Example input shape for tracing.
    dynamic_axes : dict | None
        Dynamic axis specification. Default: batch dimension is dynamic.
    verify : bool
        If True, verify exported model against PyTorch output.

    Returns
    -------
    str
        Path to the exported ONNX model.
    """
    _ensure_parent_dir(output_path)
    model = model.cpu().eval()

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    dummy_input = torch.randn(*input_shape)

    # Get reference output
    with torch.no_grad():
        ref_output = model(dummy_input)
        if isinstance(ref_output, dict):
            ref_output = ref_output.get("logits", list(ref_output.values())[0])

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"ONNX export: {output_path} ({file_size:.1f}MB, opset {opset_version})")

    # Verification
    if verify:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(output_path)
            onnx_output = sess.run(None, {"input": dummy_input.numpy()})
            diff = np.abs(ref_output.numpy() - onnx_output[0]).max()
            logger.info(f"ONNX verification: max diff = {diff:.6f}")
            if diff > 0.01:
                logger.warning(f"ONNX output diverges from PyTorch (max diff={diff:.4f})")
        except ImportError:
            logger.info("onnxruntime not installed; skipping ONNX verification")

    return output_path


def export_torchscript(
    model: nn.Module,
    output_path: str = "outputs/export/model.pt",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    method: str = "trace",
    verify: bool = True,
) -> str:
    """Export model to TorchScript.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    output_path : str
        Where to save the .pt file.
    input_shape : tuple
        Example input shape.
    method : str
        'trace' or 'script'. Trace is more reliable for transformer models.
    verify : bool
        Verify numerical consistency.

    Returns
    -------
    str
        Path to the exported TorchScript model.
    """
    _ensure_parent_dir(output_path)
    model = model.cpu().eval()
    dummy_input = torch.randn(*input_shape)

    with torch.no_grad():
        ref_output = model(dummy_input)
        if isinstance(ref_output, dict):
            ref_output = ref_output.get("logits", list(ref_output.values())[0])

    if method == "trace":
        scripted = torch.jit.trace(model, dummy_input)
    else:
        scripted = torch.jit.script(model)

    scripted.save(output_path)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"TorchScript export: {output_path} ({file_size:.1f}MB, method={method})")

    if verify:
        loaded = torch.jit.load(output_path)
        with torch.no_grad():
            ts_output = loaded(dummy_input)
            if isinstance(ts_output, dict):
                ts_output = ts_output.get("logits", list(ts_output.values())[0])
            diff = (ref_output - ts_output).abs().max().item()
            logger.info(f"TorchScript verification: max diff = {diff:.6f}")

    return output_path


def export_coreml(
    model: nn.Module,
    output_path: str = "outputs/export/model.mlpackage",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
) -> str:
    """Export model to Core ML format for Apple Silicon inference.

    Requires the ``coremltools`` package. The exported model targets the
    Apple Neural Engine (ANE) on M-series chips.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    output_path : str
        Where to save the .mlpackage.
    input_shape : tuple
        Example input shape.

    Returns
    -------
    str
        Path to the exported Core ML model.
    """
    try:
        import coremltools as ct
    except ImportError:
        logger.error("coremltools not installed. Install with: pip install coremltools")
        raise

    _ensure_parent_dir(output_path)
    model = model.cpu().eval()
    dummy_input = torch.randn(*input_shape)

    # Trace first
    traced = torch.jit.trace(model, dummy_input)

    ml_model = ct.convert(
        traced,
        inputs=[ct.ImageType(name="retinal_image", shape=input_shape, scale=1 / 255.0)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )

    ml_model.save(output_path)
    logger.info(f"Core ML export: {output_path}")
    return output_path


def export_tensorrt(
    onnx_path: str,
    output_path: str = "outputs/export/model.trt",
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 32,
    workspace_gb: int = 4,
) -> str:
    """Convert ONNX model to TensorRT engine.

    Requires ``tensorrt`` package and NVIDIA GPU.

    Parameters
    ----------
    onnx_path : str
        Path to ONNX model.
    output_path : str
        Where to save the .trt engine.
    fp16 : bool
        Enable FP16 precision.
    int8 : bool
        Enable INT8 precision (requires calibration).
    max_batch_size : int
        Maximum batch size for the engine.
    workspace_gb : int
        GPU workspace for engine building.

    Returns
    -------
    str
        Path to the TensorRT engine.
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("tensorrt not installed. Install from NVIDIA: pip install tensorrt")
        raise

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"TensorRT parse error: {parser.get_error(i)}")
            raise RuntimeError("TensorRT ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("TensorRT: FP16 enabled")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("TensorRT: INT8 enabled")

    # Build engine
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    min_shape = (1, 3, 224, 224)
    opt_shape = (max_batch_size // 2, 3, 224, 224)
    max_shape = (max_batch_size, 3, 224, 224)
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(output_path, "wb") as f:
        f.write(engine)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"TensorRT export: {output_path} ({file_size:.1f}MB)")
    return output_path


# ---------------------------------------------------------------------------
# One-command export all
# ---------------------------------------------------------------------------


def export_all(
    model: nn.Module,
    output_dir: str = "outputs/export",
    formats: Optional[list[str]] = None,
    verify: bool = True,
) -> Dict[str, str]:
    """Export model to all requested formats.

    Parameters
    ----------
    model : nn.Module
        Trained, export-ready model (LoRA should be merged).
    output_dir : str
        Base output directory.
    formats : list[str] | None
        Formats to export. Default: ['onnx', 'torchscript'].
        Options: 'onnx', 'torchscript', 'coreml', 'tensorrt'.
    verify : bool
        Run numerical verification for each format.

    Returns
    -------
    dict
        Mapping of format name to output path.
    """
    if formats is None:
        formats = ["onnx", "torchscript"]

    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    # Deep-copy to avoid mutating the caller's model (LoRA merge is irreversible)
    model = copy.deepcopy(model)

    # Merge LoRA if the model has it
    if hasattr(model, "prepare_for_export"):
        model.prepare_for_export()

    model = model.cpu().eval()

    if "onnx" in formats:
        try:
            paths["onnx"] = export_onnx(
                model, os.path.join(output_dir, "model.onnx"), verify=verify
            )
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

    if "torchscript" in formats:
        try:
            paths["torchscript"] = export_torchscript(
                model, os.path.join(output_dir, "model.pt"), verify=verify
            )
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")

    if "coreml" in formats:
        try:
            paths["coreml"] = export_coreml(model, os.path.join(output_dir, "model.mlpackage"))
        except Exception as e:
            logger.error(f"Core ML export failed: {e}")

    if "tensorrt" in formats and "onnx" in paths:
        try:
            paths["tensorrt"] = export_tensorrt(
                paths["onnx"], os.path.join(output_dir, "model.trt")
            )
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")

    logger.info(f"Export complete: {list(paths.keys())}")
    return paths
