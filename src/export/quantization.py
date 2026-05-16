"""
FP16 and INT8 quantization with accuracy/latency comparison.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantizationBenchmark:
    """Benchmark FP32 vs FP16 vs INT8 quantized models."""

    def __init__(self, device: torch.device):
        self.device = device

    def dynamic_quantize_int8(self, model: nn.Module) -> nn.Module:
        """Apply dynamic INT8 quantization to Linear layers (CPU only)."""
        model_cpu = model.cpu().eval()
        quantized = torch.ao.quantization.quantize_dynamic(
            model_cpu, {nn.Linear}, dtype=torch.qint8
        )
        return quantized

    def fp16_convert(self, model: nn.Module) -> nn.Module:
        """Convert model to FP16."""
        return model.half()

    @torch.no_grad()
    def _benchmark_latency(self, model, dummy, n_runs=50, device=None):
        dev = device or self.device
        model.eval()
        # Warmup
        for _ in range(5):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return np.array(times)

    def compare_formats(
        self, model: nn.Module, model_name: str = "model", n_runs: int = 50
    ) -> dict:
        """Benchmark FP32, FP16, INT8 and return comparison."""
        dummy_gpu = torch.randn(1, 3, 224, 224, device=self.device)
        dummy_cpu = dummy_gpu.cpu()

        results = {}

        # FP32 GPU
        model_fp32 = model.to(self.device).eval()
        lat = self._benchmark_latency(model_fp32, dummy_gpu, n_runs)
        results["FP32_GPU"] = {
            "mean_ms": float(lat.mean()),
            "p95_ms": float(np.percentile(lat, 95)),
            "size_MB": sum(p.numel() * 4 for p in model_fp32.parameters()) / 1e6,
        }

        # FP16 GPU
        if self.device.type == "cuda":
            model_fp16 = self.fp16_convert(model.to(self.device))
            dummy_fp16 = dummy_gpu.half()
            lat = self._benchmark_latency(model_fp16, dummy_fp16, n_runs)
            results["FP16_GPU"] = {
                "mean_ms": float(lat.mean()),
                "p95_ms": float(np.percentile(lat, 95)),
                "size_MB": sum(p.numel() * 2 for p in model_fp16.parameters()) / 1e6,
            }

        # INT8 CPU (dynamic quantization is CPU-only)
        try:
            model_int8 = self.dynamic_quantize_int8(model)
            lat = self._benchmark_latency(model_int8, dummy_cpu, n_runs, device=torch.device("cpu"))
            results["INT8_CPU"] = {
                "mean_ms": float(lat.mean()),
                "p95_ms": float(np.percentile(lat, 95)),
                "size_MB": "dynamic",
            }
        except Exception as e:
            results["INT8_CPU"] = {"error": str(e)}

        logger.info(f"Quantization comparison for {model_name}:")
        for fmt, data in results.items():
            if "error" not in data:
                logger.info(f"  {fmt}: {data['mean_ms']:.1f}ms mean, {data.get('size_MB', '?')} MB")

        return results

    def save_report(self, results: dict, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "quantization_report.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
