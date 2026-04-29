"""
Latency & throughput benchmarking for all 4 models.
Extracted from notebook cell 56 (LatencyBenchmark).
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from src.visualization.ieee_style import (
    ieee_style, ieee_figure, save_ieee, add_watermark,
    IEEE_COLORS, MODEL_COLORS,
)


class LatencyBenchmark:
    """Benchmark inference latency/throughput across models and batch sizes."""

    def __init__(self, device: torch.device):
        self.device = device

    @torch.no_grad()
    def benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        batch_sizes: list[int] = None,
        warmup: int = 10,
        n_runs: int = 50,
    ) -> dict:
        """Benchmark a single model across batch sizes."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]

        model.eval()
        model.to(self.device)
        results = {"model": model_name, "batch_results": {}}

        params = sum(p.numel() for p in model.parameters()) / 1e6
        results["params_M"] = params

        for bs in batch_sizes:
            try:
                dummy = torch.randn(bs, 3, 224, 224, device=self.device)

                # Warmup
                for _ in range(warmup):
                    model(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                # Benchmark
                latencies = []
                for _ in range(n_runs):
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(dummy)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - t0) * 1000)

                lat = np.array(latencies)
                results["batch_results"][bs] = {
                    "mean_ms": float(lat.mean()),
                    "median_ms": float(np.median(lat)),
                    "p95_ms": float(np.percentile(lat, 95)),
                    "p99_ms": float(np.percentile(lat, 99)),
                    "std_ms": float(lat.std()),
                    "throughput_fps": float(1000 / lat.mean() * bs),
                }
            except RuntimeError:
                results["batch_results"][bs] = {"error": "OOM"}
                torch.cuda.empty_cache()

        # GPU memory
        if self.device.type == "cuda":
            results["gpu_mem_MB"] = float(torch.cuda.max_memory_allocated(self.device) / 1e6)
            torch.cuda.reset_peak_memory_stats(self.device)

        return results

    def benchmark_all_models(
        self, models: dict[str, nn.Module], **kwargs
    ) -> dict[str, dict]:
        """Benchmark all models sequentially."""
        all_results = {}
        for name, model in models.items():
            print(f"  Benchmarking {name}...")
            all_results[name] = self.benchmark_model(model, name, **kwargs)
            del model
            torch.cuda.empty_cache()
        return all_results


def plot_latency_benchmark(
    results: dict[str, dict],
    save_dir: str | Path = "outputs/plots/benchmarks",
):
    """Generate IEEE benchmark plots from benchmark results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with ieee_style():
        fig, axes = ieee_figure(2, 2, width="double", height_ratio=0.5)
        models = list(results.keys())

        # (a) Latency comparison (batch=1)
        lat_1 = []
        for m in models:
            bs1 = results[m].get("batch_results", {}).get(1, {})
            lat_1.append(bs1.get("mean_ms", 0))
        colors = [MODEL_COLORS.get(m, f"C{i}") for i, m in enumerate(models)]
        axes[0, 0].barh(models, lat_1, color=colors, edgecolor="black", lw=0.3)
        for i, v in enumerate(lat_1):
            axes[0, 0].text(v + 0.5, i, f"{v:.1f}ms", va="center", fontsize=7)
        axes[0, 0].set_xlabel("Latency (ms)")
        axes[0, 0].set_title("(a) Inference Latency (BS=1)")

        # (b) Throughput scaling
        batch_sizes = [1, 2, 4, 8, 16]
        for i, m in enumerate(models):
            fps_vals = []
            for bs in batch_sizes:
                d = results[m].get("batch_results", {}).get(bs, {})
                fps_vals.append(d.get("throughput_fps", 0))
            color = MODEL_COLORS.get(m, f"C{i}")
            axes[0, 1].plot(batch_sizes, fps_vals, "o-", ms=4, color=color, label=m)
        axes[0, 1].set_xlabel("Batch Size")
        axes[0, 1].set_ylabel("Throughput (FPS)")
        axes[0, 1].set_title("(b) Throughput Scaling")
        axes[0, 1].legend(fontsize=6)

        # (c) GPU memory
        mem_vals = [results[m].get("gpu_mem_MB", 0) for m in models]
        axes[1, 0].bar(models, mem_vals, color=colors, edgecolor="black", lw=0.3)
        for i, v in enumerate(mem_vals):
            axes[1, 0].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=7)
        axes[1, 0].set_ylabel("GPU Memory (MB)")
        axes[1, 0].set_title("(c) Peak GPU Memory")
        axes[1, 0].tick_params(axis="x", labelsize=7)

        # (d) Parameters
        param_vals = [results[m].get("params_M", 0) for m in models]
        axes[1, 1].bar(models, param_vals, color=colors, edgecolor="black", lw=0.3)
        for i, v in enumerate(param_vals):
            axes[1, 1].text(i, v + 0.5, f"{v:.1f}M", ha="center", fontsize=7)
        axes[1, 1].set_ylabel("Parameters (M)")
        axes[1, 1].set_title("(d) Model Size")
        axes[1, 1].tick_params(axis="x", labelsize=7)

        fig.suptitle("Multi-Model Latency Benchmark", fontsize=11, fontweight="bold", y=1.02)
        add_watermark(fig)
        save_ieee(fig, save_dir / "fig_latency_benchmark")

    # Save raw results
    with open(save_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Benchmark plots saved to {save_dir}")
