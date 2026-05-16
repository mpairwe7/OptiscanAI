#!/usr/bin/env python3
"""Benchmark fundus gate v2 latency and throughput.

Usage:
    python scripts/benchmark_gate.py
    python scripts/benchmark_gate.py --warmup 20 --runs 500
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image


def make_test_image(size: int = 224) -> Image.Image:
    """Create a synthetic fundus-like test image."""
    arr = np.zeros((size, size, 3), dtype=np.float32)
    cy, cx = size / 2.0, size / 2.0
    y, x = np.mgrid[0:size, 0:size]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / (size / 2.0)
    boundary = np.clip(1.0 - (dist - 0.82) * 20, 0, 1)
    arr[:, :, 0] = boundary * 0.55
    arr[:, :, 1] = boundary * 0.30
    arr[:, :, 2] = boundary * 0.12
    rng = np.random.RandomState(42)
    arr[:, :, 1] += rng.randn(size, size).astype(np.float32) * 0.015 * boundary
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def benchmark_statistical_only(image: Image.Image, warmup: int, runs: int) -> list[float]:
    """Benchmark statistical gate alone."""
    from src.data.fundus_gate import gate_image

    for _ in range(warmup):
        gate_image(image)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        gate_image(image)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def benchmark_v2_gate(image: Image.Image, warmup: int, runs: int) -> list[float]:
    """Benchmark full v2 fusion gate."""
    from src.data.fundus_gate_v2 import gate_image

    for _ in range(warmup):
        gate_image(image)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        gate_image(image)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def print_stats(name: str, latencies: list[float]):
    arr = np.array(latencies)
    print(
        f"  {name:30s}  p50={np.percentile(arr, 50):6.1f}ms  "
        f"p95={np.percentile(arr, 95):6.1f}ms  "
        f"p99={np.percentile(arr, 99):6.1f}ms  "
        f"mean={arr.mean():6.1f}ms  "
        f"throughput={1000/arr.mean():5.0f} img/s"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark fundus gate v2")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    image = make_test_image()
    print(f"\nBenchmarking fundus gate (warmup={args.warmup}, runs={args.runs})")
    print("=" * 90)

    # Memory baseline
    try:
        import psutil

        proc = psutil.Process()
        mem_before = proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        mem_before = None

    # Statistical only (v1)
    lat_stat = benchmark_statistical_only(image, args.warmup, args.runs)
    print_stats("Statistical gate (v1)", lat_stat)

    # V2 fusion gate
    lat_v2 = benchmark_v2_gate(image, args.warmup, args.runs)
    print_stats("Fusion gate (v2)", lat_v2)

    # Memory after
    if mem_before is not None:
        mem_after = proc.memory_info().rss / (1024 * 1024)
        print(
            f"\n  Memory: before={mem_before:.0f}MB, after={mem_after:.0f}MB, delta={mem_after - mem_before:.0f}MB"
        )

    # Targets
    print("\n  Targets:")
    p99_v2 = np.percentile(lat_v2, 99)
    status = "PASS" if p99_v2 < 12 else "FAIL"
    print(f"    v2 p99 < 12ms: {p99_v2:.1f}ms [{status}]")
    p99_stat = np.percentile(lat_stat, 99)
    status = "PASS" if p99_stat < 5 else "FAIL"
    print(f"    v1 p99 < 5ms:  {p99_stat:.1f}ms [{status}]")
    print()


if __name__ == "__main__":
    main()
