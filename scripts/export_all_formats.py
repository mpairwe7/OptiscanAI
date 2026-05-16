#!/usr/bin/env python3
"""Export RetinalAI model to all serving formats with parity validation.

Supports ONNX, TorchScript, CoreML, and INT8/FP16 quantization in a
single invocation.  Each exported artefact is validated for numerical
parity against the original PyTorch FP32 model and the results are
printed as a summary table.

Usage
-----
    # Export to all default formats (ONNX + TorchScript + INT8 quantized)
    python scripts/export_all_formats.py --model-path models/model_vignn_rank1.pth

    # Specific formats only
    python scripts/export_all_formats.py \\
        --model-path outputs/checkpoints/best.pth \\
        --output-dir outputs/export \\
        --formats onnx torchscript int8

    # All formats including CoreML (macOS only)
    python scripts/export_all_formats.py \\
        --model-path best.pth \\
        --formats onnx torchscript coreml int8 fp16
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Available export formats
# ---------------------------------------------------------------------------
ALL_FORMATS = ["onnx", "torchscript", "coreml", "int8", "fp16"]
DEFAULT_FORMATS = ["onnx", "torchscript", "int8"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    """Load a RetinalAI model from a checkpoint.

    Tries multiple model architectures in order of likelihood:
    vignn -> retinal_foundation_hybrid -> scene_graph_transformer -> graphclip.

    Returns
    -------
    tuple
        (model, checkpoint_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Determine metadata from checkpoint
    model_name = ckpt.get("model_name", "vignn") if isinstance(ckpt, dict) else "vignn"
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    from src.data.datamodule import DISEASE_COLUMNS
    from src.models.vignn import ClinicalKnowledgeGraph, create_vignn_model

    kg = ClinicalKnowledgeGraph(disease_names=DISEASE_COLUMNS)
    num_classes = len(DISEASE_COLUMNS)

    if model_name == "retinal_foundation_hybrid":
        try:
            from src.models.retinal_foundation_hybrid import create_hybrid_model
            from src.models.vignn import create_knowledge_graph

            kg_full = create_knowledge_graph()
            model = create_hybrid_model(
                clinical_knowledge_graph=kg_full,
                checkpoint_path=None,
            )
        except ImportError:
            logger.warning("retinal_foundation_hybrid not available -- " "falling back to vignn")
            model = create_vignn_model(
                num_classes=num_classes,
                clinical_knowledge_graph=kg,
            )
    elif model_name == "scene_graph_transformer":
        try:
            from src.models.scene_graph_transformer import SceneGraphTransformer

            model = SceneGraphTransformer(
                num_classes=num_classes,
                hidden_dim=384,
                num_layers=3,
                num_heads=4,
                dropout=0.1,
                clinical_knowledge_graph=kg,
            )
        except ImportError:
            model = create_vignn_model(
                num_classes=num_classes,
                clinical_knowledge_graph=kg,
            )
    elif model_name == "graphclip":
        try:
            from src.models.graphclip import GraphCLIP

            model = GraphCLIP(
                num_classes=num_classes,
                hidden_dim=384,
                num_graph_layers=3,
                num_heads=4,
                dropout=0.1,
                clinical_knowledge_graph=kg,
            )
        except ImportError:
            model = create_vignn_model(
                num_classes=num_classes,
                clinical_knowledge_graph=kg,
            )
    else:
        model = create_vignn_model(
            num_classes=num_classes,
            clinical_knowledge_graph=kg,
        )

    # Load weights
    if isinstance(state_dict, dict) and not isinstance(state_dict, torch.nn.Module):
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            logger.warning("State dict load had issues: %s", exc)

    # Merge LoRA if applicable
    if hasattr(model, "prepare_for_export"):
        model.prepare_for_export()

    model.cpu().eval()
    logger.info(
        "Model loaded: %s (%d parameters)", model_name, sum(p.numel() for p in model.parameters())
    )

    return model, ckpt if isinstance(ckpt, dict) else {"model_state_dict": ckpt}


# ---------------------------------------------------------------------------
# Reference output for parity checks
# ---------------------------------------------------------------------------


def get_reference_output(model: torch.nn.Module) -> tuple[torch.Tensor, np.ndarray]:
    """Run a dummy forward pass and return (input_tensor, output_probs)."""
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
        if isinstance(output, dict):
            output = output.get("logits", list(output.values())[0])
        probs = torch.sigmoid(output).numpy()
    return dummy, probs


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_onnx_format(
    model: torch.nn.Module,
    output_dir: Path,
    dummy: torch.Tensor,
    ref_probs: np.ndarray,
) -> dict:
    """Export to ONNX and validate parity."""
    from src.optimization.export import export_onnx

    onnx_path = str(output_dir / "model.onnx")
    t0 = time.perf_counter()
    try:
        export_onnx(model, onnx_path, verify=False)
        export_time = time.perf_counter() - t0
    except Exception as exc:
        return {
            "format": "onnx",
            "status": "FAILED",
            "error": str(exc),
            "path": onnx_path,
            "size_mb": 0.0,
            "export_time_s": time.perf_counter() - t0,
            "parity_max_diff": None,
            "parity_passed": False,
        }

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    # Parity check
    max_diff = None
    parity_passed = False
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        ort_out = sess.run(None, {input_name: dummy.numpy()})
        ort_probs = 1.0 / (1.0 + np.exp(-ort_out[0].astype(np.float64)))
        max_diff = float(np.abs(ref_probs - ort_probs).max())
        parity_passed = max_diff < 1e-3
    except ImportError:
        logger.info("onnxruntime not installed -- skipping ONNX parity check")
    except Exception as exc:
        logger.warning("ONNX parity check failed: %s", exc)

    return {
        "format": "onnx",
        "status": "OK",
        "path": onnx_path,
        "size_mb": round(size_mb, 2),
        "export_time_s": round(export_time, 2),
        "parity_max_diff": round(max_diff, 6) if max_diff is not None else None,
        "parity_passed": parity_passed,
    }


def export_torchscript_format(
    model: torch.nn.Module,
    output_dir: Path,
    dummy: torch.Tensor,
    ref_probs: np.ndarray,
) -> dict:
    """Export to TorchScript and validate parity."""
    from src.optimization.export import export_torchscript

    ts_path = str(output_dir / "model.pt")
    t0 = time.perf_counter()
    try:
        export_torchscript(model, ts_path, verify=False)
        export_time = time.perf_counter() - t0
    except Exception as exc:
        return {
            "format": "torchscript",
            "status": "FAILED",
            "error": str(exc),
            "path": ts_path,
            "size_mb": 0.0,
            "export_time_s": time.perf_counter() - t0,
            "parity_max_diff": None,
            "parity_passed": False,
        }

    size_mb = os.path.getsize(ts_path) / (1024 * 1024)

    # Parity check
    max_diff = None
    parity_passed = False
    try:
        loaded = torch.jit.load(ts_path)
        with torch.no_grad():
            ts_out = loaded(dummy)
            if isinstance(ts_out, dict):
                ts_out = ts_out.get("logits", list(ts_out.values())[0])
            ts_probs = torch.sigmoid(ts_out).numpy()
            max_diff = float(np.abs(ref_probs - ts_probs).max())
            parity_passed = max_diff < 1e-4
    except Exception as exc:
        logger.warning("TorchScript parity check failed: %s", exc)

    return {
        "format": "torchscript",
        "status": "OK",
        "path": ts_path,
        "size_mb": round(size_mb, 2),
        "export_time_s": round(export_time, 2),
        "parity_max_diff": round(max_diff, 6) if max_diff is not None else None,
        "parity_passed": parity_passed,
    }


def export_coreml_format(
    model: torch.nn.Module,
    output_dir: Path,
) -> dict:
    """Export to CoreML. Parity is not checked (requires macOS)."""
    t0 = time.perf_counter()
    try:
        from src.optimization.export import export_coreml

        coreml_path = str(output_dir / "model.mlpackage")
        export_coreml(model, coreml_path)
        export_time = time.perf_counter() - t0

        # CoreML packages are directories; measure total size
        total_size = 0
        coreml_dir = Path(coreml_path)
        if coreml_dir.is_dir():
            for f in coreml_dir.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
        else:
            total_size = os.path.getsize(coreml_path)

        size_mb = total_size / (1024 * 1024)

        return {
            "format": "coreml",
            "status": "OK",
            "path": coreml_path,
            "size_mb": round(size_mb, 2),
            "export_time_s": round(export_time, 2),
            "parity_max_diff": None,
            "parity_passed": None,  # requires macOS to validate
        }
    except ImportError:
        return {
            "format": "coreml",
            "status": "SKIPPED",
            "error": "coremltools not installed (pip install coremltools)",
            "path": "",
            "size_mb": 0.0,
            "export_time_s": time.perf_counter() - t0,
            "parity_max_diff": None,
            "parity_passed": None,
        }
    except Exception as exc:
        return {
            "format": "coreml",
            "status": "FAILED",
            "error": str(exc),
            "path": "",
            "size_mb": 0.0,
            "export_time_s": time.perf_counter() - t0,
            "parity_max_diff": None,
            "parity_passed": False,
        }


def export_quantized_format(
    model: torch.nn.Module,
    output_dir: Path,
    dummy: torch.Tensor,
    ref_probs: np.ndarray,
    precision: str = "int8",
) -> dict:
    """Quantize and save a PyTorch model with parity validation."""
    import copy

    t0 = time.perf_counter()
    try:
        model_copy = copy.deepcopy(model).cpu().eval()

        if precision == "int8":
            from src.optimization.quantization import quantize_dynamic_int8

            quantized = quantize_dynamic_int8(model_copy)
            save_path = output_dir / "model_int8.pth"
        elif precision == "fp16":
            from src.optimization.quantization import convert_to_fp16

            quantized = convert_to_fp16(model_copy)
            save_path = output_dir / "model_fp16.pth"
        else:
            return {
                "format": f"quantized_{precision}",
                "status": "FAILED",
                "error": f"Unknown precision: {precision}",
                "path": "",
                "size_mb": 0.0,
                "export_time_s": time.perf_counter() - t0,
                "parity_max_diff": None,
                "parity_passed": False,
            }

        torch.save(quantized.state_dict(), str(save_path))
        export_time = time.perf_counter() - t0
        size_mb = os.path.getsize(str(save_path)) / (1024 * 1024)

        # Parity check
        max_diff = None
        parity_passed = False
        try:
            test_input = dummy.clone()
            if precision == "fp16":
                test_input = test_input.half()

            with torch.no_grad():
                q_out = quantized(test_input)
                if isinstance(q_out, dict):
                    q_out = q_out.get("logits", list(q_out.values())[0])
                q_probs = torch.sigmoid(q_out.float()).numpy()
                max_diff = float(np.abs(ref_probs - q_probs).max())
                # Quantized models get a more relaxed tolerance
                tolerance = 0.05 if precision == "int8" else 1e-3
                parity_passed = max_diff < tolerance
        except Exception as exc:
            logger.warning("Quantized parity check failed (%s): %s", precision, exc)

        return {
            "format": f"quantized_{precision}",
            "status": "OK",
            "path": str(save_path),
            "size_mb": round(size_mb, 2),
            "export_time_s": round(export_time, 2),
            "parity_max_diff": round(max_diff, 6) if max_diff is not None else None,
            "parity_passed": parity_passed,
        }

    except Exception as exc:
        return {
            "format": f"quantized_{precision}",
            "status": "FAILED",
            "error": str(exc),
            "path": "",
            "size_mb": 0.0,
            "export_time_s": time.perf_counter() - t0,
            "parity_max_diff": None,
            "parity_passed": False,
        }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], fp32_size_mb: float) -> None:
    """Print a formatted summary table of export results."""
    # Column widths
    fmt_w = 18
    status_w = 8
    size_w = 12
    parity_w = 14
    diff_w = 14
    time_w = 10

    sep = "-" * (fmt_w + status_w + size_w + parity_w + diff_w + time_w + 17)

    print()
    print(sep)
    print(
        f"{'Format':<{fmt_w}} | "
        f"{'Status':<{status_w}} | "
        f"{'Size (MB)':<{size_w}} | "
        f"{'Parity':<{parity_w}} | "
        f"{'Max Diff':<{diff_w}} | "
        f"{'Time (s)':<{time_w}}"
    )
    print(sep)

    # FP32 baseline
    print(
        f"{'fp32 (baseline)':<{fmt_w}} | "
        f"{'--':<{status_w}} | "
        f"{fp32_size_mb:<{size_w}.2f} | "
        f"{'reference':<{parity_w}} | "
        f"{'--':<{diff_w}} | "
        f"{'--':<{time_w}}"
    )

    for r in results:
        status = r["status"]
        size_str = f"{r['size_mb']:.2f}" if r["size_mb"] > 0 else "--"
        if r["parity_passed"] is None:
            parity_str = "N/A"
        elif r["parity_passed"]:
            parity_str = "PASS"
        else:
            parity_str = "FAIL"

        diff_str = f"{r['parity_max_diff']:.6f}" if r["parity_max_diff"] is not None else "--"
        time_str = f"{r['export_time_s']:.2f}" if r.get("export_time_s") else "--"

        print(
            f"{r['format']:<{fmt_w}} | "
            f"{status:<{status_w}} | "
            f"{size_str:<{size_w}} | "
            f"{parity_str:<{parity_w}} | "
            f"{diff_str:<{diff_w}} | "
            f"{time_str:<{time_w}}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=("Export RetinalAI model to all serving formats with parity validation."),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model_vignn_rank1.pth",
        help="Path to PyTorch checkpoint (default: models/model_vignn_rank1.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/export",
        help="Directory for exported artefacts (default: outputs/export)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=DEFAULT_FORMATS,
        choices=ALL_FORMATS,
        help=f"Formats to export (default: {DEFAULT_FORMATS})",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip parity validation (faster but less safe)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = _PROJECT_ROOT / model_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n[1/3] Loading model from {model_path} ...")
    model, checkpoint = load_model(str(model_path))

    # FP32 baseline size
    import io as _io

    buf = _io.BytesIO()
    torch.save(model.state_dict(), buf)
    fp32_size_mb = buf.tell() / (1024 * 1024)

    # Reference output for parity
    print("[2/3] Computing reference output for parity validation ...")
    dummy, ref_probs = get_reference_output(model)

    # Export each format
    print(f"[3/3] Exporting to: {args.formats} ...")
    results: list[dict] = []

    if "onnx" in args.formats:
        logger.info("Exporting ONNX ...")
        results.append(export_onnx_format(model, output_dir, dummy, ref_probs))

    if "torchscript" in args.formats:
        logger.info("Exporting TorchScript ...")
        results.append(export_torchscript_format(model, output_dir, dummy, ref_probs))

    if "coreml" in args.formats:
        logger.info("Exporting CoreML ...")
        results.append(export_coreml_format(model, output_dir))

    if "int8" in args.formats:
        logger.info("Exporting INT8 quantized ...")
        results.append(export_quantized_format(model, output_dir, dummy, ref_probs, "int8"))

    if "fp16" in args.formats:
        logger.info("Exporting FP16 ...")
        results.append(export_quantized_format(model, output_dir, dummy, ref_probs, "fp16"))

    # Summary
    print_summary(results, fp32_size_mb)

    # Save manifest
    manifest = {
        "source_checkpoint": str(model_path),
        "output_dir": str(output_dir),
        "fp32_size_mb": round(fp32_size_mb, 2),
        "formats": results,
        "decision_thresholds": checkpoint.get("decision_thresholds"),
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Manifest saved: %s", manifest_path)

    # Exit with non-zero if any export failed
    failed = [r for r in results if r["status"] == "FAILED"]
    if failed:
        logger.warning(
            "%d format(s) failed to export: %s",
            len(failed),
            [r["format"] for r in failed],
        )
        sys.exit(1)

    print(f"\nAll exports complete. Artefacts in: {output_dir}")


if __name__ == "__main__":
    main()
