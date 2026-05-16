#!/usr/bin/env python3
"""Export MobileStudentV1 to ONNX INT8 with parity validation.

Usage:
    PYTHONPATH=. python scripts/export_mobile_student.py \
        --checkpoint outputs/distillation/student_best.pth \
        --output-dir outputs/mobile_export

Produces:
    outputs/mobile_export/
        student_fp32.onnx         # Full precision ONNX
        student_int8.onnx         # Dynamic INT8 quantized ONNX
        parity_report.json        # Parity validation report
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mobile_student import MobileStudentV1

logger = logging.getLogger(__name__)


def load_student(checkpoint_path: str, device: torch.device) -> MobileStudentV1:
    """Load student model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = ckpt.get("config", {})
    sc = config.get("student", {})
    num_classes = config.get("class_filtering", {}).get("num_classes", 28)

    student = MobileStudentV1(
        num_classes=num_classes,
        hidden_dim=sc.get("hidden_dim", 512),
        dropout1=sc.get("dropout1", 0.4),
        dropout2=sc.get("dropout2", 0.25),
        pretrained=False,
    )

    state = ckpt.get("state_dict", ckpt)
    student.load_state_dict(state, strict=False)

    if "thresholds" in ckpt:
        student.thresholds.copy_(ckpt["thresholds"])

    student.prepare_for_export()
    return student.to(device)


def export_onnx_fp32(
    model: MobileStudentV1, output_path: Path, img_size: int = 224
) -> Path:
    """Export model to ONNX FP32."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
    )

    # Verify
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / 1e6
    logger.info("Exported FP32 ONNX: %s (%.1f MB)", output_path, size_mb)
    return output_path


def quantize_onnx_int8(fp32_path: Path, int8_path: Path) -> Path:
    """Dynamic INT8 quantization of ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QUInt8,
    )

    size_mb = int8_path.stat().st_size / 1e6
    logger.info("Quantized INT8 ONNX: %s (%.1f MB)", int8_path, size_mb)
    return int8_path


def validate_onnx_parity(
    torch_model: MobileStudentV1,
    onnx_fp32_path: Path,
    onnx_int8_path: Path,
    n_samples: int = 50,
    img_size: int = 224,
) -> dict:
    """Validate parity between PyTorch, FP32 ONNX, and INT8 ONNX."""
    import onnxruntime as ort

    device = next(torch_model.parameters()).device
    torch_model.eval()

    # Create ONNX sessions
    fp32_session = ort.InferenceSession(
        str(onnx_fp32_path), providers=["CPUExecutionProvider"]
    )
    int8_session = ort.InferenceSession(
        str(onnx_int8_path), providers=["CPUExecutionProvider"]
    )

    max_fp32_diff = 0.0
    max_int8_diff = 0.0
    fp32_diffs = []
    int8_diffs = []
    max_fp32_prob_diff = 0.0
    max_int8_prob_diff = 0.0
    fp32_prob_diffs = []
    int8_prob_diffs = []
    int8_binary_agreements = []
    latencies_torch = []
    latencies_int8 = []

    for i in range(n_samples):
        dummy = torch.randn(1, 3, img_size, img_size)
        dummy_np = dummy.numpy()

        # PyTorch
        with torch.no_grad():
            t0 = time.perf_counter()
            torch_out = torch_model(dummy.to(device)).cpu().numpy()
            latencies_torch.append((time.perf_counter() - t0) * 1000)
            torch_prob = 1.0 / (1.0 + np.exp(-torch_out))

        # FP32 ONNX
        fp32_out = fp32_session.run(None, {"input": dummy_np})[0]

        # INT8 ONNX
        t0 = time.perf_counter()
        int8_out = int8_session.run(None, {"input": dummy_np})[0]
        latencies_int8.append((time.perf_counter() - t0) * 1000)

        fp32_diff = np.abs(torch_out - fp32_out).max()
        int8_diff = np.abs(torch_out - int8_out).max()
        fp32_prob = 1.0 / (1.0 + np.exp(-fp32_out))
        int8_prob = 1.0 / (1.0 + np.exp(-int8_out))
        fp32_prob_diff = np.abs(torch_prob - fp32_prob).max()
        int8_prob_diff = np.abs(torch_prob - int8_prob).max()
        int8_binary_agreement = np.mean((torch_prob >= 0.5) == (int8_prob >= 0.5))

        max_fp32_diff = max(max_fp32_diff, fp32_diff)
        max_int8_diff = max(max_int8_diff, int8_diff)
        max_fp32_prob_diff = max(max_fp32_prob_diff, fp32_prob_diff)
        max_int8_prob_diff = max(max_int8_prob_diff, int8_prob_diff)
        fp32_diffs.append(fp32_diff)
        int8_diffs.append(int8_diff)
        fp32_prob_diffs.append(fp32_prob_diff)
        int8_prob_diffs.append(int8_prob_diff)
        int8_binary_agreements.append(int8_binary_agreement)

    report = {
        "n_samples": n_samples,
        "fp32_onnx": {
            "max_logit_diff": float(max_fp32_diff),
            "mean_logit_diff": float(np.mean(fp32_diffs)),
            "max_probability_diff": float(max_fp32_prob_diff),
            "mean_probability_diff": float(np.mean(fp32_prob_diffs)),
            "size_mb": onnx_fp32_path.stat().st_size / 1e6,
        },
        "int8_onnx": {
            "max_logit_diff": float(max_int8_diff),
            "mean_logit_diff": float(np.mean(int8_diffs)),
            "max_probability_diff": float(max_int8_prob_diff),
            "mean_probability_diff": float(np.mean(int8_prob_diffs)),
            "mean_binary_agreement_at_0_5": float(np.mean(int8_binary_agreements)),
            "size_mb": onnx_int8_path.stat().st_size / 1e6,
            "latency_p50_ms": float(np.percentile(latencies_int8, 50)),
            "latency_p95_ms": float(np.percentile(latencies_int8, 95)),
            "latency_p99_ms": float(np.percentile(latencies_int8, 99)),
        },
        "torch": {
            "latency_p50_ms": float(np.percentile(latencies_torch, 50)),
            "latency_p95_ms": float(np.percentile(latencies_torch, 95)),
        },
        "parity_checks": {
            "fp32_parity_ok": bool(max_fp32_diff < 1e-4),
            "int8_parity_ok": bool(
                max_int8_prob_diff < 0.20
                and np.mean(int8_binary_agreements) >= 0.99
            ),
        },
    }
    return report


def validate_model_size(onnx_path: Path, max_size_mb: float = 50.0) -> bool:
    """Check ONNX model is within size budget."""
    size_mb = onnx_path.stat().st_size / 1e6
    ok = size_mb <= max_size_mb
    if not ok:
        logger.error("Model size %.1f MB exceeds limit %.1f MB", size_mb, max_size_mb)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Export MobileStudentV1 to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/distillation/student_best.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mobile_export",
    )
    parser.add_argument("--max-size-mb", type=float, default=50.0)
    parser.add_argument("--n-parity-samples", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load student
    student = load_student(args.checkpoint, device)
    logger.info("Student params: %s", student.get_param_summary())

    # Export FP32
    fp32_path = export_onnx_fp32(student, output_dir / "student_fp32.onnx")

    # Quantize INT8
    int8_path = quantize_onnx_int8(fp32_path, output_dir / "student_int8.onnx")

    # Validate size
    size_ok = validate_model_size(int8_path, args.max_size_mb)

    # Parity validation (run on CPU for fair comparison)
    student_cpu = student.cpu()
    report = validate_onnx_parity(
        student_cpu, fp32_path, int8_path, n_samples=args.n_parity_samples
    )
    report["size_check"] = {
        "int8_size_mb": int8_path.stat().st_size / 1e6,
        "max_allowed_mb": args.max_size_mb,
        "passed": size_ok,
    }

    # Copy thresholds
    thresh_src = Path(args.checkpoint).parent / "thresholds_student.json"
    if thresh_src.exists():
        import shutil

        shutil.copy(thresh_src, output_dir / "thresholds.json")
        logger.info("Copied thresholds to %s", output_dir / "thresholds.json")

    # Save report
    report_path = output_dir / "parity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Parity report saved to %s", report_path)

    # Summary
    all_passed = (
        report["parity_checks"]["fp32_parity_ok"]
        and report["parity_checks"]["int8_parity_ok"]
        and size_ok
    )

    print(f"\n{'='*60}")
    print(f"Mobile Student Export {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*60}")
    print(f"  FP32 ONNX: {report['fp32_onnx']['size_mb']:.1f} MB")
    print(f"  INT8 ONNX: {report['int8_onnx']['size_mb']:.1f} MB (limit: {args.max_size_mb} MB)")
    print(f"  FP32 parity: max_diff={report['fp32_onnx']['max_logit_diff']:.6f}")
    print(
        "  INT8 parity: "
        f"max_prob_diff={report['int8_onnx']['max_probability_diff']:.6f}, "
        f"binary_agreement={report['int8_onnx']['mean_binary_agreement_at_0_5']:.4f}"
    )
    print(f"  INT8 latency p95: {report['int8_onnx']['latency_p95_ms']:.1f} ms (CPU)")
    print(f"{'='*60}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
