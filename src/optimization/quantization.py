"""
Quantization and optimization pipeline for production deployment.

Supports:
    - torch.compile (mode=max-autotune) for kernel fusion
    - Dynamic INT8 quantization via PyTorch
    - Static INT8 quantization with calibration
    - FP16 half-precision inference
    - Knowledge distillation (RETFound teacher -> lightweight student)

Target: <75MB model size with <12ms p99 latency on A100.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.quantization as tq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# torch.compile wrapper
# ---------------------------------------------------------------------------


def compile_model(
    model: nn.Module,
    mode: str = "max-autotune",
    backend: str = "inductor",
    fullgraph: bool = False,
) -> nn.Module:
    """Apply torch.compile for kernel fusion and optimization.

    Parameters
    ----------
    model : nn.Module
        Model to compile.
    mode : str
        Compilation mode: 'default', 'reduce-overhead', 'max-autotune'.
    backend : str
        Compiler backend.
    fullgraph : bool
        If True, require full-graph compilation (no graph breaks).

    Returns
    -------
    nn.Module
        Compiled model.
    """
    try:
        compiled = torch.compile(model, mode=mode, backend=backend, fullgraph=fullgraph)
        logger.info(f"torch.compile applied (mode={mode}, backend={backend})")
        return compiled
    except Exception as e:
        logger.warning(f"torch.compile failed ({e}), returning original model")
        return model


# ---------------------------------------------------------------------------
# Dynamic INT8 Quantization
# ---------------------------------------------------------------------------


def quantize_dynamic_int8(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Apply dynamic INT8 quantization (weights quantized, activations at runtime).

    Best for models dominated by nn.Linear layers (transformers, MLPs).
    Typically 2-4x smaller with minimal accuracy loss.

    Parameters
    ----------
    model : nn.Module
        Model to quantize. Must be on CPU.
    dtype : torch.dtype
        Quantization dtype (qint8 or float16).

    Returns
    -------
    nn.Module
        Quantized model.
    """
    model_cpu = model.cpu().eval()

    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        qconfig_spec={nn.Linear},
        dtype=dtype,
    )

    # Report size reduction
    orig_size = _model_size_mb(model_cpu)
    quant_size = _model_size_mb(quantized)
    logger.info(
        f"Dynamic INT8 quantization: {orig_size:.1f}MB -> {quant_size:.1f}MB "
        f"({(1 - quant_size/orig_size)*100:.1f}% reduction)"
    )

    return quantized


# ---------------------------------------------------------------------------
# Static INT8 Quantization with Calibration
# ---------------------------------------------------------------------------


def quantize_static_int8(
    model: nn.Module,
    calibration_loader: Any,
    num_calibration_batches: int = 50,
    backend: str = "x86",
) -> nn.Module:
    """Apply static INT8 quantization with calibration data.

    Quantizes both weights and activations for maximum compression.
    Requires representative calibration data for activation range estimation.

    Parameters
    ----------
    model : nn.Module
        Model to quantize. Must be on CPU.
    calibration_loader : DataLoader
        Representative data for calibration.
    num_calibration_batches : int
        How many batches to use for calibration.
    backend : str
        Quantization backend: 'x86', 'qnnpack', 'onednn'.

    Returns
    -------
    nn.Module
        Statically quantized model.
    """
    torch.backends.quantized.engine = backend
    model_cpu = model.cpu().eval()

    # Prepare for static quantization
    model_cpu.qconfig = tq.get_default_qconfig(backend)
    prepared = tq.prepare(model_cpu, inplace=False)

    # Calibration
    logger.info(f"Calibrating with {num_calibration_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            prepared(images.cpu())

    # Convert to quantized
    quantized = tq.convert(prepared, inplace=False)

    orig_size = _model_size_mb(model.cpu())
    quant_size = _model_size_mb(quantized)
    logger.info(
        f"Static INT8 quantization: {orig_size:.1f}MB -> {quant_size:.1f}MB "
        f"({(1 - quant_size/orig_size)*100:.1f}% reduction)"
    )

    return quantized


# ---------------------------------------------------------------------------
# FP16 Conversion
# ---------------------------------------------------------------------------


def convert_to_fp16(model: nn.Module) -> nn.Module:
    """Convert model to FP16 half precision for GPU inference."""
    model_fp16 = model.half()
    size = _model_size_mb(model_fp16)
    logger.info(f"FP16 conversion complete: {size:.1f}MB")
    return model_fp16


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------


class DistillationLoss(nn.Module):
    """Combined distillation + task loss for knowledge distillation.

    L = alpha * KD_loss(student, teacher) + (1 - alpha) * task_loss(student, targets)

    The teacher produces soft targets via temperature-scaled softmax;
    the student learns to match these soft distributions.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature

        # Soft targets from teacher
        soft_teacher = torch.sigmoid(teacher_logits / T)
        soft_student = torch.sigmoid(student_logits / T)

        # KL divergence on soft targets
        kd_loss = nn.functional.binary_cross_entropy(
            soft_student, soft_teacher.detach(), reduction="mean"
        ) * (T * T)

        # Hard target loss
        hard_loss = self.task_loss(student_logits, targets)

        return self.alpha * kd_loss + (1 - self.alpha) * hard_loss


class LightweightStudent(nn.Module):
    """Compact student model (~15-20M params) for edge deployment.

    Uses EfficientNet-B0 backbone + lightweight MLP head.
    Trained via knowledge distillation from RetinalFoundationHybrid teacher.
    """

    def __init__(self, num_classes: int = 48, dropout: float = 0.2):
        super().__init__()
        import timm

        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features  # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    device: str = "cuda",
    warmup_runs: int = 20,
    benchmark_runs: int = 100,
    use_fp16: bool = False,
) -> Dict[str, float]:
    """Benchmark model inference latency.

    Returns
    -------
    dict
        mean_ms, std_ms, p50_ms, p95_ms, p99_ms, throughput_fps
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)
    if use_fp16:
        model = model.half()
        dummy = dummy.half()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    latencies = sorted(latencies)
    n = len(latencies)

    results = {
        "mean_ms": sum(latencies) / n,
        "std_ms": (sum((x - sum(latencies) / n) ** 2 for x in latencies) / n) ** 0.5,
        "p50_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
        "throughput_fps": 1000.0 / (sum(latencies) / n),
        "device": device,
        "batch_size": input_shape[0],
        "fp16": use_fp16,
    }

    logger.info(
        f"Latency: mean={results['mean_ms']:.2f}ms, "
        f"p99={results['p99_ms']:.2f}ms, "
        f"throughput={results['throughput_fps']:.1f} FPS"
    )

    return results


# ---------------------------------------------------------------------------
# Full optimization pipeline
# ---------------------------------------------------------------------------


def optimize_for_production(
    model: nn.Module,
    calibration_loader: Optional[Any] = None,
    output_dir: str = "outputs/optimized",
    enable_compile: bool = True,
    enable_dynamic_int8: bool = True,
    enable_static_int8: bool = False,
    enable_fp16: bool = True,
    benchmark: bool = True,
) -> Dict[str, Any]:
    """Run full optimization pipeline and save all variants.

    Parameters
    ----------
    model : nn.Module
        Trained model to optimize.
    calibration_loader : DataLoader | None
        Required for static quantization.
    output_dir : str
        Directory to save optimized models.
    enable_compile : bool
        Apply torch.compile.
    enable_dynamic_int8 : bool
        Create dynamic INT8 variant.
    enable_static_int8 : bool
        Create static INT8 variant (needs calibration_loader).
    enable_fp16 : bool
        Create FP16 variant.
    benchmark : bool
        Run latency benchmarks.

    Returns
    -------
    dict
        Paths to saved models and benchmark results.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Any] = {"models": {}, "benchmarks": {}}

    # Save FP32 baseline
    fp32_path = os.path.join(output_dir, "model_fp32.pth")
    torch.save(model.state_dict(), fp32_path)
    results["models"]["fp32"] = fp32_path
    results["models"]["fp32_size_mb"] = _model_size_mb(model)

    # torch.compile
    if enable_compile and torch.cuda.is_available():
        compile_model(copy.deepcopy(model))
        results["models"]["compiled"] = "in-memory (torch.compile)"

    # Dynamic INT8
    if enable_dynamic_int8:
        int8_model = quantize_dynamic_int8(copy.deepcopy(model))
        int8_path = os.path.join(output_dir, "model_int8_dynamic.pth")
        torch.save(int8_model.state_dict(), int8_path)
        results["models"]["int8_dynamic"] = int8_path
        results["models"]["int8_dynamic_size_mb"] = _model_size_mb(int8_model)

    # Static INT8
    if enable_static_int8 and calibration_loader is not None:
        static_model = quantize_static_int8(copy.deepcopy(model), calibration_loader)
        static_path = os.path.join(output_dir, "model_int8_static.pth")
        torch.save(static_model.state_dict(), static_path)
        results["models"]["int8_static"] = static_path
        results["models"]["int8_static_size_mb"] = _model_size_mb(static_model)

    # FP16
    if enable_fp16:
        fp16_model = convert_to_fp16(copy.deepcopy(model))
        fp16_path = os.path.join(output_dir, "model_fp16.pth")
        torch.save(fp16_model.state_dict(), fp16_path)
        results["models"]["fp16"] = fp16_path
        results["models"]["fp16_size_mb"] = _model_size_mb(fp16_model)

    # Benchmarks
    if benchmark and torch.cuda.is_available():
        for batch_size in [1, 32]:
            shape = (batch_size, 3, 224, 224)
            key = f"batch_{batch_size}"
            results["benchmarks"][key] = benchmark_latency(
                copy.deepcopy(model), input_shape=shape, device="cuda", use_fp16=True
            )

    logger.info(f"Optimization pipeline complete. Outputs in {output_dir}")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB via state_dict serialization.

    Uses torch.save to a BytesIO buffer to correctly measure quantized models
    (whose packed parameters are not exposed via .parameters()).
    """
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)
