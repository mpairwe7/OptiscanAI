#!/usr/bin/env python3
"""
Latency and throughput benchmark for RetinalFoundationHybrid.

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --device cuda --batch-sizes 1 8 32 --fp16
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.models.retinal_foundation_hybrid import create_hybrid_model
from src.models.vignn import create_knowledge_graph
from src.optimization.quantization import _model_size_mb, benchmark_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RetinalFoundationHybrid")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--backbone", type=str, default="vit_large_patch16_224")
    args = parser.parse_args()

    # Build model
    kg = create_knowledge_graph()
    model = create_hybrid_model(
        clinical_knowledge_graph=kg,
        backbone=args.backbone,
        use_lora=False,
        enable_moe=True,
    )

    size_mb = _model_size_mb(model)
    params = model.get_param_summary()

    print("\nRetinalFoundationHybrid Benchmark")
    print(f"{'='*60}")
    print(f"Backbone: {args.backbone}")
    print(f"Total params: {params['total']/1e6:.1f}M")
    print(f"Trainable params: {params['trainable']/1e6:.1f}M")
    print(f"Model size (FP32): {size_mb:.1f}MB")
    print(f"Device: {args.device}")
    print(f"FP16: {args.fp16}")
    print(f"{'='*60}")

    results = []
    for bs in args.batch_sizes:
        result = benchmark_latency(
            model,
            input_shape=(bs, 3, 224, 224),
            device=args.device,
            warmup_runs=args.warmup,
            benchmark_runs=args.runs,
            use_fp16=args.fp16,
        )
        results.append(result)

        print(f"\nBatch size {bs}:")
        print(f"  Mean:       {result['mean_ms']:.2f} ms")
        print(f"  Std:        {result['std_ms']:.2f} ms")
        print(f"  P50:        {result['p50_ms']:.2f} ms")
        print(f"  P95:        {result['p95_ms']:.2f} ms")
        print(f"  P99:        {result['p99_ms']:.2f} ms")
        print(f"  Throughput: {result['throughput_fps']:.1f} FPS")

    print(f"\n{'='*60}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
