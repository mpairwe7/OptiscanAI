"""Quantized model management and server optimization endpoints.

Provides model listing, quantization status, and performance metrics
for GGUF/AWQ/GPTQ/ONNX quantized model variants.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["quantized-models"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class QuantizedModelInfo(BaseModel):
    """Information about a single quantized model variant."""

    format: str = Field(..., description="Quantization format (gguf_q4_k_m, awq_4bit, gptq_4bit, onnx, int8, fp16)")
    filename: str = Field(..., description="Model filename")
    size_mb: float = Field(..., description="File size in MB")
    path: str = Field(..., description="Relative path from models directory")
    available: bool = Field(True, description="Whether the model file exists and is loadable")
    quantization_bits: Optional[int] = Field(None, description="Quantization bit width")
    description: str = Field("", description="Human-readable description")


class QuantizedModelsResponse(BaseModel):
    """Response listing all available quantized model variants."""

    models: list[QuantizedModelInfo] = Field(default_factory=list)
    total_count: int = Field(0)
    baseline_model: str = Field("", description="Path to the baseline (full-precision) model")
    baseline_size_mb: float = Field(0.0)
    feature_flag_enabled: bool = Field(False, description="Whether FLAG_QUANTIZATION is active")


class ServerOptimizationStatus(BaseModel):
    """Status of server-side optimizations."""

    torch_compile_enabled: bool = False
    torch_compile_mode: str = ""
    prefix_cache_enabled: bool = False
    speculative_decoding_enabled: bool = False
    vllm_enabled: bool = False
    vllm_batch_size: int = 0
    quantized_model_loaded: str = ""
    quantized_embedder_loaded: str = ""
    memory_usage_mb: float = 0.0
    memory_reduction_pct: float = 0.0


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


QUANTIZED_DIRS = [
    "outputs/quantized",
    "models/export",
    "outputs/optimized",
]

FORMAT_PATTERNS = {
    "gguf_q4_k_m": ["*q4_k_m*", "*Q4_K_M*"],
    "gguf_q5_k_m": ["*q5_k_m*", "*Q5_K_M*"],
    "gguf_q8_0": ["*q8_0*", "*Q8_0*"],
    "awq_4bit": ["*awq*", "*AWQ*"],
    "gptq_4bit": ["*gptq*", "*GPTQ*"],
    "onnx": ["*.onnx"],
    "onnx_optimized": ["*optimized*.onnx"],
    "onnx_int8": ["*int8*.onnx"],
    "int8_dynamic": ["*int8_dynamic*"],
    "int8_static": ["*int8_static*"],
    "fp16": ["*fp16*"],
    "tensorrt": ["*tensorrt*", "*trt*"],
}


def _discover_quantized_models() -> list[QuantizedModelInfo]:
    """Scan known directories for quantized model artifacts."""
    models: list[QuantizedModelInfo] = []
    seen_paths: set[str] = set()
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    for rel_dir in QUANTIZED_DIRS:
        search_dir = project_root / rel_dir
        if not search_dir.exists():
            continue

        for fmt_name, patterns in FORMAT_PATTERNS.items():
            for pattern in patterns:
                for fpath in search_dir.glob(pattern):
                    if not fpath.is_file():
                        continue
                    abs_path = str(fpath.resolve())
                    if abs_path in seen_paths:
                        continue
                    seen_paths.add(abs_path)

                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    rel_path = str(fpath.relative_to(project_root))

                    bits = None
                    if "q4" in fmt_name or "4bit" in fmt_name:
                        bits = 4
                    elif "q5" in fmt_name:
                        bits = 5
                    elif "q8" in fmt_name or "int8" in fmt_name:
                        bits = 8
                    elif "fp16" in fmt_name:
                        bits = 16

                    desc = _format_description(fmt_name)

                    models.append(QuantizedModelInfo(
                        format=fmt_name,
                        filename=fpath.name,
                        size_mb=round(size_mb, 2),
                        path=rel_path,
                        available=True,
                        quantization_bits=bits,
                        description=desc,
                    ))

    return sorted(models, key=lambda m: m.size_mb)


def _format_description(fmt_name: str) -> str:
    """Human-readable description for a quantization format."""
    descriptions = {
        "gguf_q4_k_m": "GGUF 4-bit quantization (K-quants, medium quality) — best balance of size/quality",
        "gguf_q5_k_m": "GGUF 5-bit quantization (K-quants, medium) — higher quality, larger size",
        "gguf_q8_0": "GGUF 8-bit quantization — near-lossless quality",
        "awq_4bit": "AWQ 4-bit activation-aware quantization — optimized for GPU inference",
        "gptq_4bit": "GPTQ 4-bit with second-order error correction — high quality 4-bit",
        "onnx": "ONNX format — cross-platform inference via ONNX Runtime",
        "onnx_optimized": "ONNX with graph optimizations — faster inference",
        "onnx_int8": "ONNX with INT8 quantization — smallest ONNX variant",
        "int8_dynamic": "PyTorch dynamic INT8 quantization — CPU-optimized",
        "int8_static": "PyTorch static INT8 with calibration — maximum compression",
        "fp16": "FP16 half-precision — 50% size reduction, GPU-only",
        "tensorrt": "TensorRT-LLM optimized — maximum GPU throughput",
    }
    return descriptions.get(fmt_name, f"Quantized format: {fmt_name}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/quantized", response_model=QuantizedModelsResponse)
async def list_quantized_models() -> QuantizedModelsResponse:
    """List all available quantized model variants with sizes.

    Scans configured directories for GGUF, AWQ, GPTQ, ONNX, and other
    quantized model artifacts. Returns metadata including file size,
    quantization bit width, and availability.
    """
    flag_enabled = getattr(settings, "quantization", None) is not None and getattr(
        getattr(settings, "quantization", None), "enabled", False
    )

    models = _discover_quantized_models()

    # Baseline model info
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    baseline_path = project_root / settings.model_path
    baseline_size = 0.0
    if baseline_path.exists():
        baseline_size = round(baseline_path.stat().st_size / (1024 * 1024), 2)

    return QuantizedModelsResponse(
        models=models,
        total_count=len(models),
        baseline_model=settings.model_path,
        baseline_size_mb=baseline_size,
        feature_flag_enabled=flag_enabled,
    )


@router.get("/optimization/status", response_model=ServerOptimizationStatus)
async def get_optimization_status() -> ServerOptimizationStatus:
    """Get the current server-side optimization status.

    Reports on torch.compile, prefix caching, speculative decoding,
    vLLM integration, and memory usage.
    """
    import torch

    memory_mb = 0.0
    if torch.cuda.is_available():
        memory_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)

    quantization_cfg = getattr(settings, "quantization", None)

    return ServerOptimizationStatus(
        torch_compile_enabled=getattr(quantization_cfg, "torch_compile_enabled", False) if quantization_cfg else False,
        torch_compile_mode=getattr(quantization_cfg, "torch_compile_mode", "") if quantization_cfg else "",
        prefix_cache_enabled=getattr(quantization_cfg, "prefix_cache_enabled", False) if quantization_cfg else False,
        speculative_decoding_enabled=getattr(quantization_cfg, "speculative_decoding_enabled", False) if quantization_cfg else False,
        vllm_enabled=settings.ray.enabled,
        vllm_batch_size=settings.ray.batch_max_size,
        quantized_model_loaded=getattr(quantization_cfg, "active_format", "") if quantization_cfg else "",
        quantized_embedder_loaded=getattr(quantization_cfg, "embedder_format", "") if quantization_cfg else "",
        memory_usage_mb=memory_mb,
    )
