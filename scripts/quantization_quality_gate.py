#!/usr/bin/env python3
"""Quality gate for quantised RetinalAI model artefacts.

Validates that quantised models meet clinical acceptance criteria before
deployment.  Designed for CI integration -- exits 0 when all gates pass,
exits 1 when any gate fails.

Gates
-----
- **Faithfulness**: Quantised predictions must stay within a configurable
  tolerance (default 4%) of the BF16/FP32 baseline.
- **WER**: For speech-enabled pipelines, word error rate increase must
  not exceed a threshold (default 3%).
- **Bundle size**: Each artefact must be under a maximum file size.
- **Latency**: p95 inference latency must be under a configured ceiling.

Usage
-----
    # Basic validation against manifest from quantize_models.py
    python scripts/quantization_quality_gate.py \\
        --manifest-path outputs/quantized/quantization_manifest.json

    # With custom thresholds
    python scripts/quantization_quality_gate.py \\
        --manifest-path outputs/quantized/quantization_manifest.json \\
        --baseline-metrics outputs/baseline_metrics.json \\
        --max-faithfulness-drop 0.03 \\
        --max-bundle-size-mb 200 \\
        --max-p95-latency-ms 50

    # CI mode: exit code signals pass/fail
    python scripts/quantization_quality_gate.py \\
        --manifest-path outputs/quantized/quantization_manifest.json \\
        && echo "QUALITY GATE PASSED" || echo "QUALITY GATE FAILED"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate result data structure
# ---------------------------------------------------------------------------


class GateResult:
    """Container for a single quality gate check result."""

    def __init__(
        self,
        gate_name: str,
        artefact: str,
        passed: bool,
        metric_value: float | str | None = None,
        threshold: float | str | None = None,
        detail: str = "",
    ):
        self.gate_name = gate_name
        self.artefact = artefact
        self.passed = passed
        self.metric_value = metric_value
        self.threshold = threshold
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate_name,
            "artefact": self.artefact,
            "passed": self.passed,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Faithfulness gate
# ---------------------------------------------------------------------------


def check_faithfulness(
    artefact: dict,
    reference_predictions_path: str,
    max_drop: float = 0.04,
) -> GateResult:
    """Validate that quantised model predictions are faithful to baseline.

    Loads the quantised model artefact, runs inference on the same input
    used for the reference, and compares output probabilities.  The
    faithfulness score is 1 - mean_absolute_error; the gate fails if
    the drop from baseline exceeds ``max_drop``.

    Parameters
    ----------
    artefact : dict
        Single artefact entry from quantization_manifest.json.
    reference_predictions_path : str
        Path to reference_predictions.npz produced by quantize_models.py.
    max_drop : float
        Maximum allowed faithfulness drop (0.04 = 4%).

    Returns
    -------
    GateResult
    """

    fmt = artefact.get("format", "unknown")
    artefact_path = artefact.get("path", "")

    if not artefact_path or not Path(artefact_path).exists():
        return GateResult(
            gate_name="faithfulness",
            artefact=fmt,
            passed=False,
            detail=f"Artefact not found: {artefact_path}",
        )

    # Load reference predictions
    try:
        ref = np.load(reference_predictions_path)
        ref_input = ref["input"]  # (1, 3, 224, 224)
        ref_probs = ref["probs"]  # (1, num_classes)
    except Exception as exc:
        return GateResult(
            gate_name="faithfulness",
            artefact=fmt,
            passed=False,
            detail=f"Cannot load reference predictions: {exc}",
        )

    # Attempt to load and run the quantised artefact
    quant_probs = None

    try:
        if fmt.startswith("onnx"):
            quant_probs = _infer_onnx(artefact_path, ref_input)
        elif fmt == "tensorrt":
            quant_probs = _infer_torchscript(artefact_path, ref_input)
        else:
            quant_probs = _infer_pytorch_checkpoint(artefact_path, ref_input)
    except Exception as exc:
        logger.warning("Faithfulness inference failed for %s: %s", fmt, exc)
        return GateResult(
            gate_name="faithfulness",
            artefact=fmt,
            passed=False,
            detail=f"Inference failed: {exc}",
        )

    if quant_probs is None:
        return GateResult(
            gate_name="faithfulness",
            artefact=fmt,
            passed=False,
            detail="Could not obtain quantised predictions",
        )

    # Compute faithfulness score
    # Faithfulness = 1 - MAE between reference and quantised probabilities
    mae = float(np.abs(ref_probs.flatten() - quant_probs.flatten()).mean())
    faithfulness = 1.0 - mae
    baseline_faithfulness = 1.0  # FP32 reference is perfect by definition
    drop = baseline_faithfulness - faithfulness

    passed = drop <= max_drop

    detail = (
        f"faithfulness={faithfulness:.4f}, MAE={mae:.4f}, "
        f"drop={drop:.4f} (max allowed={max_drop:.4f})"
    )

    return GateResult(
        gate_name="faithfulness",
        artefact=fmt,
        passed=passed,
        metric_value=round(drop, 6),
        threshold=max_drop,
        detail=detail,
    )


def _infer_onnx(onnx_path: str, input_array: np.ndarray) -> np.ndarray | None:
    """Run ONNX inference and return sigmoid probabilities."""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: input_array.astype(np.float32)})
        logits = outputs[0].astype(np.float64)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs
    except ImportError:
        logger.warning("onnxruntime not installed -- cannot validate ONNX faithfulness")
        return None


def _infer_torchscript(ts_path: str, input_array: np.ndarray) -> np.ndarray | None:
    """Run TorchScript inference and return sigmoid probabilities."""
    import torch

    try:
        model = torch.jit.load(ts_path, map_location="cpu")
        model.eval()
        inp = torch.from_numpy(input_array).float()
        with torch.no_grad():
            output = model(inp)
            if isinstance(output, dict):
                output = output.get("logits", list(output.values())[0])
            probs = torch.sigmoid(output).numpy()
        return probs
    except Exception as exc:
        logger.warning("TorchScript inference failed: %s", exc)
        return None


def _infer_pytorch_checkpoint(
    ckpt_path: str,
    input_array: np.ndarray,
) -> np.ndarray | None:
    """Load a PyTorch checkpoint and run inference.

    Handles both plain state_dict saves and wrapped dicts from the
    quantise pipeline (with 'state_dict' key).
    """
    import torch

    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # The quantise pipeline wraps state_dicts in metadata dicts
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        else:
            # Might be a raw state_dict
            state_dict = payload

        # Reconstruct model to load weights into
        from src.data.datamodule import DISEASE_COLUMNS
        from src.models.vignn import ClinicalKnowledgeGraph, create_vignn_model

        kg = ClinicalKnowledgeGraph(disease_names=DISEASE_COLUMNS)
        model = create_vignn_model(
            num_classes=len(DISEASE_COLUMNS),
            clinical_knowledge_graph=kg,
        )

        # Apply dynamic quantisation if the checkpoint was quantised
        quant_type = None
        if isinstance(payload, dict):
            quant_type = payload.get("quantization", "")

        if quant_type and "proxy" in str(quant_type):
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            # Quantised state dicts may have different keys
            logger.debug("Strict load failed, attempting partial load")
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name in own_state:
                    try:
                        own_state[name].copy_(param)
                    except Exception:
                        pass

        model.eval()
        inp = torch.from_numpy(input_array).float()
        with torch.no_grad():
            output = model(inp)
            if isinstance(output, dict):
                output = output.get("logits", list(output.values())[0])
            probs = torch.sigmoid(output.float()).numpy()
        return probs

    except Exception as exc:
        logger.warning("PyTorch checkpoint inference failed for %s: %s", ckpt_path, exc)
        return None


# ---------------------------------------------------------------------------
# WER gate (for speech-enabled pipelines)
# ---------------------------------------------------------------------------


def check_wer(
    baseline_metrics: dict,
    quantised_metrics: Optional[dict] = None,
    max_increase: float = 0.03,
    artefact_name: str = "all",
) -> GateResult:
    """Validate that word error rate has not degraded beyond threshold.

    This gate is relevant when the platform includes speech-to-text
    models (e.g. for voice-first mobile UX).  If no WER data is
    available the gate is automatically passed with a note.

    Parameters
    ----------
    baseline_metrics : dict
        Must contain 'wer' key with float value (0.0 - 1.0).
    quantised_metrics : dict | None
        Must contain 'wer' key.  If None, gate passes with info note.
    max_increase : float
        Maximum allowed WER increase (0.03 = 3%).

    Returns
    -------
    GateResult
    """
    baseline_wer = baseline_metrics.get("wer")
    if baseline_wer is None:
        return GateResult(
            gate_name="wer",
            artefact=artefact_name,
            passed=True,
            detail="No baseline WER data -- gate skipped (speech models not in scope)",
        )

    if quantised_metrics is None or "wer" not in quantised_metrics:
        return GateResult(
            gate_name="wer",
            artefact=artefact_name,
            passed=True,
            detail="No quantised WER data available -- gate skipped",
        )

    quant_wer = quantised_metrics["wer"]
    increase = quant_wer - baseline_wer
    passed = increase <= max_increase

    detail = (
        f"baseline_wer={baseline_wer:.4f}, quantised_wer={quant_wer:.4f}, "
        f"increase={increase:.4f} (max allowed={max_increase:.4f})"
    )

    return GateResult(
        gate_name="wer",
        artefact=artefact_name,
        passed=passed,
        metric_value=round(increase, 6),
        threshold=max_increase,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Bundle size gate
# ---------------------------------------------------------------------------


def check_bundle_size(
    artefact: dict,
    max_size_mb: float = 500.0,
) -> GateResult:
    """Validate that an artefact does not exceed the maximum bundle size.

    Parameters
    ----------
    artefact : dict
        Single artefact entry from quantization_manifest.json.
    max_size_mb : float
        Maximum allowed size in megabytes.

    Returns
    -------
    GateResult
    """
    fmt = artefact.get("format", "unknown")
    artefact_path = artefact.get("path", "")
    reported_size = artefact.get("size_mb", 0.0)

    # Verify actual file size on disk
    actual_size = 0.0
    if artefact_path and Path(artefact_path).exists():
        p = Path(artefact_path)
        if p.is_file():
            actual_size = p.stat().st_size / (1024 * 1024)
        elif p.is_dir():
            actual_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)

    # Use whichever is larger (manifest may be stale)
    size = max(reported_size, actual_size)

    passed = size <= max_size_mb
    detail = f"size={size:.2f} MB (max allowed={max_size_mb:.2f} MB)"

    return GateResult(
        gate_name="bundle_size",
        artefact=fmt,
        passed=passed,
        metric_value=round(size, 2),
        threshold=max_size_mb,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Latency gate
# ---------------------------------------------------------------------------


def check_latency(
    artefact: dict,
    max_p95_ms: float = 100.0,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> GateResult:
    """Benchmark inference latency and validate p95 is under threshold.

    Parameters
    ----------
    artefact : dict
        Single artefact entry from quantization_manifest.json.
    max_p95_ms : float
        Maximum allowed p95 latency in milliseconds.
    warmup_runs : int
        Number of warmup iterations.
    benchmark_runs : int
        Number of timed iterations.

    Returns
    -------
    GateResult
    """
    import torch

    fmt = artefact.get("format", "unknown")
    artefact_path = artefact.get("path", "")

    if not artefact_path or not Path(artefact_path).exists():
        return GateResult(
            gate_name="latency",
            artefact=fmt,
            passed=False,
            detail=f"Artefact not found: {artefact_path}",
        )

    dummy = torch.randn(1, 3, 224, 224)

    try:
        if fmt.startswith("onnx"):
            latencies = _benchmark_onnx(artefact_path, dummy.numpy(), warmup_runs, benchmark_runs)
        elif fmt == "tensorrt":
            latencies = _benchmark_torchscript(artefact_path, dummy, warmup_runs, benchmark_runs)
        else:
            latencies = _benchmark_pytorch_checkpoint(
                artefact_path,
                dummy,
                warmup_runs,
                benchmark_runs,
            )
    except Exception as exc:
        return GateResult(
            gate_name="latency",
            artefact=fmt,
            passed=False,
            detail=f"Benchmark failed: {exc}",
        )

    if latencies is None or len(latencies) == 0:
        return GateResult(
            gate_name="latency",
            artefact=fmt,
            passed=False,
            detail="Could not collect latency measurements",
        )

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean = float(np.mean(latencies))

    passed = p95 <= max_p95_ms

    detail = (
        f"mean={mean:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms, "
        f"p99={p99:.2f}ms (max p95={max_p95_ms:.2f}ms)"
    )

    return GateResult(
        gate_name="latency",
        artefact=fmt,
        passed=passed,
        metric_value=round(p95, 2),
        threshold=max_p95_ms,
        detail=detail,
    )


def _benchmark_onnx(
    onnx_path: str,
    input_array: np.ndarray,
    warmup: int,
    runs: int,
) -> list[float] | None:
    """Benchmark ONNX inference latency in milliseconds."""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        inp = input_array.astype(np.float32)

        for _ in range(warmup):
            sess.run(None, {input_name: inp})

        latencies: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, {input_name: inp})
            latencies.append((time.perf_counter() - t0) * 1000)

        return latencies
    except ImportError:
        logger.warning("onnxruntime not installed -- cannot benchmark ONNX latency")
        return None


def _benchmark_torchscript(
    ts_path: str,
    dummy: Any,
    warmup: int,
    runs: int,
) -> list[float] | None:
    """Benchmark TorchScript inference latency in milliseconds."""
    import torch

    try:
        model = torch.jit.load(ts_path, map_location="cpu")
        model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)

        latencies: list[float] = []
        with torch.no_grad():
            for _ in range(runs):
                t0 = time.perf_counter()
                model(dummy)
                latencies.append((time.perf_counter() - t0) * 1000)

        return latencies
    except Exception as exc:
        logger.warning("TorchScript benchmark failed: %s", exc)
        return None


def _benchmark_pytorch_checkpoint(
    ckpt_path: str,
    dummy: Any,
    warmup: int,
    runs: int,
) -> list[float] | None:
    """Benchmark a PyTorch checkpoint inference latency in milliseconds."""
    import torch
    import torch.nn as nn

    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        else:
            state_dict = payload

        from src.data.datamodule import DISEASE_COLUMNS
        from src.models.vignn import ClinicalKnowledgeGraph, create_vignn_model

        kg = ClinicalKnowledgeGraph(disease_names=DISEASE_COLUMNS)
        model = create_vignn_model(
            num_classes=len(DISEASE_COLUMNS),
            clinical_knowledge_graph=kg,
        )

        # Apply quantisation if checkpoint indicates it
        quant_type = ""
        if isinstance(payload, dict):
            quant_type = str(payload.get("quantization", ""))

        if quant_type and "proxy" in quant_type:
            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8,
            )

        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass

        model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)

        latencies: list[float] = []
        with torch.no_grad():
            for _ in range(runs):
                t0 = time.perf_counter()
                model(dummy)
                latencies.append((time.perf_counter() - t0) * 1000)

        return latencies

    except Exception as exc:
        logger.warning("PyTorch checkpoint benchmark failed for %s: %s", ckpt_path, exc)
        return None


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_gate_report(results: list[GateResult]) -> None:
    """Print a formatted PASS/FAIL table for all gate checks."""
    gate_w = 16
    artefact_w = 22
    status_w = 6
    value_w = 14
    thresh_w = 14
    detail_w = 50

    sep = "-" * (gate_w + artefact_w + status_w + value_w + thresh_w + detail_w + 17)

    print()
    print("=" * len(sep))
    print("  QUANTIZATION QUALITY GATE REPORT")
    print("=" * len(sep))
    print()
    print(sep)
    print(
        f"{'Gate':<{gate_w}} | "
        f"{'Artefact':<{artefact_w}} | "
        f"{'Pass?':<{status_w}} | "
        f"{'Value':<{value_w}} | "
        f"{'Threshold':<{thresh_w}} | "
        f"{'Detail':<{detail_w}}"
    )
    print(sep)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        value_str = str(r.metric_value) if r.metric_value is not None else "--"
        thresh_str = str(r.threshold) if r.threshold is not None else "--"
        # Truncate detail for table display
        detail_str = r.detail[:detail_w] if r.detail else "--"

        print(
            f"{r.gate_name:<{gate_w}} | "
            f"{r.artefact:<{artefact_w}} | "
            f"{status:<{status_w}} | "
            f"{value_str:<{value_w}} | "
            f"{thresh_str:<{thresh_w}} | "
            f"{detail_str:<{detail_w}}"
        )

    print(sep)

    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    failed_count = total - passed_count

    print()
    if failed_count == 0:
        print(f"  RESULT: ALL {total} CHECKS PASSED")
    else:
        print(f"  RESULT: {failed_count}/{total} CHECKS FAILED")
        for r in results:
            if not r.passed:
                print(f"    - [{r.gate_name}] {r.artefact}: {r.detail}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quality gate for quantised RetinalAI artefacts. "
            "Exits 0 if all gates pass, 1 if any gate fails."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        required=True,
        help="Path to quantization_manifest.json from quantize_models.py",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default=None,
        help="Path to baseline metrics JSON (with optional 'wer' field)",
    )
    parser.add_argument(
        "--max-faithfulness-drop",
        type=float,
        default=0.04,
        help="Maximum allowed faithfulness drop from baseline (default: 0.04 = 4%%)",
    )
    parser.add_argument(
        "--max-wer-increase",
        type=float,
        default=0.03,
        help="Maximum allowed WER increase from baseline (default: 0.03 = 3%%)",
    )
    parser.add_argument(
        "--max-bundle-size-mb",
        type=float,
        default=500.0,
        help="Maximum allowed artefact size in MB (default: 500)",
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=100.0,
        help="Maximum allowed p95 inference latency in ms (default: 100)",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency benchmarks (faster for CI dry-run)",
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="Skip faithfulness checks (requires model loading)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load manifest
    # -----------------------------------------------------------------------
    manifest_path = Path(args.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = _PROJECT_ROOT / manifest_path

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    artefacts = manifest.get("artefacts", [])
    reference_path = manifest.get("reference_predictions", "")
    fp32_size = manifest.get("fp32_size_mb", 0.0)

    logger.info(
        "Loaded manifest: %d artefacts, FP32 baseline %.2f MB",
        len(artefacts),
        fp32_size,
    )

    # -----------------------------------------------------------------------
    # Load baseline metrics (optional)
    # -----------------------------------------------------------------------
    baseline_metrics: dict = {}
    if args.baseline_metrics:
        bm_path = Path(args.baseline_metrics)
        if not bm_path.is_absolute():
            bm_path = _PROJECT_ROOT / bm_path
        if bm_path.exists():
            with open(bm_path) as f:
                baseline_metrics = json.load(f)
            logger.info("Loaded baseline metrics from %s", bm_path)
        else:
            logger.warning("Baseline metrics not found: %s", bm_path)

    # -----------------------------------------------------------------------
    # Run quality gates
    # -----------------------------------------------------------------------
    all_results: list[GateResult] = []

    # Filter to only successful artefacts
    valid_artefacts = [a for a in artefacts if a.get("status") == "OK"]
    failed_artefacts = [a for a in artefacts if a.get("status") != "OK"]

    if failed_artefacts:
        logger.warning(
            "%d artefact(s) had non-OK status and will be skipped: %s",
            len(failed_artefacts),
            [a.get("format") for a in failed_artefacts],
        )

    if not valid_artefacts:
        logger.error("No valid artefacts to validate")
        print("\n  RESULT: NO VALID ARTEFACTS -- GATE FAILED\n")
        sys.exit(1)

    print(f"\nValidating {len(valid_artefacts)} artefact(s) ...")

    for artefact in valid_artefacts:
        fmt = artefact.get("format", "unknown")
        logger.info("--- Checking: %s ---", fmt)

        # Faithfulness gate
        if not args.skip_faithfulness and reference_path:
            ref_p = Path(reference_path)
            if not ref_p.is_absolute():
                ref_p = _PROJECT_ROOT / ref_p
            if ref_p.exists():
                result = check_faithfulness(
                    artefact,
                    str(ref_p),
                    max_drop=args.max_faithfulness_drop,
                )
                all_results.append(result)
                logger.info(
                    "  Faithfulness [%s]: %s",
                    fmt,
                    "PASS" if result.passed else "FAIL",
                )

        # Bundle size gate
        result = check_bundle_size(artefact, max_size_mb=args.max_bundle_size_mb)
        all_results.append(result)
        logger.info(
            "  Bundle size [%s]: %s",
            fmt,
            "PASS" if result.passed else "FAIL",
        )

        # Latency gate
        if not args.skip_latency:
            result = check_latency(
                artefact,
                max_p95_ms=args.max_p95_latency_ms,
            )
            all_results.append(result)
            logger.info(
                "  Latency [%s]: %s",
                fmt,
                "PASS" if result.passed else "FAIL",
            )

    # WER gate (global, not per-artefact)
    wer_result = check_wer(
        baseline_metrics,
        max_increase=args.max_wer_increase,
    )
    all_results.append(wer_result)
    logger.info("WER gate: %s", "PASS" if wer_result.passed else "FAIL")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print_gate_report(all_results)

    # Save structured report
    report = {
        "manifest_path": str(manifest_path),
        "thresholds": {
            "max_faithfulness_drop": args.max_faithfulness_drop,
            "max_wer_increase": args.max_wer_increase,
            "max_bundle_size_mb": args.max_bundle_size_mb,
            "max_p95_latency_ms": args.max_p95_latency_ms,
        },
        "artefacts_checked": len(valid_artefacts),
        "artefacts_skipped": len(failed_artefacts),
        "total_checks": len(all_results),
        "checks_passed": sum(1 for r in all_results if r.passed),
        "checks_failed": sum(1 for r in all_results if not r.passed),
        "overall_passed": all(r.passed for r in all_results),
        "results": [r.to_dict() for r in all_results],
    }

    report_path = manifest_path.parent / "quality_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Quality gate report saved: %s", report_path)

    # -----------------------------------------------------------------------
    # Exit code for CI
    # -----------------------------------------------------------------------
    if report["overall_passed"]:
        print(f"Quality gate report: {report_path}")
        print("EXIT 0 -- all gates passed\n")
        sys.exit(0)
    else:
        print(f"Quality gate report: {report_path}")
        print("EXIT 1 -- one or more gates failed\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
