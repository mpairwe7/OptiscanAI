#!/usr/bin/env python3
"""
One-command production export for RetinalFoundationHybrid.

Usage:
    python scripts/export_production.py --checkpoint outputs/checkpoints/hybrid/best.pth
    python scripts/export_production.py --checkpoint best.pth --formats onnx torchscript --quantize
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.models.retinal_foundation_hybrid import create_hybrid_model
from src.models.vignn import create_knowledge_graph
from src.optimization.export import export_all
from src.optimization.quantization import (
    benchmark_latency,
    optimize_for_production,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export RetinalFoundationHybrid")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument(
        "--formats", nargs="+", default=["onnx", "torchscript"], help="Export formats"
    )
    parser.add_argument("--output-dir", type=str, default="outputs/export")
    parser.add_argument("--quantize", action="store_true", help="Run quantization pipeline")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmarks")
    parser.add_argument("--backbone", type=str, default="vit_large_patch16_224")
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    # Build model
    kg = create_knowledge_graph()
    model = create_hybrid_model(
        clinical_knowledge_graph=kg,
        backbone=args.backbone,
        use_lora=True,
        lora_rank=args.lora_rank,
        checkpoint_path=args.checkpoint,
    )

    # Merge LoRA for export
    model.prepare_for_export()
    model.eval()

    # Export
    logger.info(f"Exporting to formats: {args.formats}")
    paths = export_all(model, output_dir=args.output_dir, formats=args.formats)

    for fmt, path in paths.items():
        logger.info(f"  {fmt}: {path}")

    # Quantization
    if args.quantize:
        logger.info("Running quantization pipeline...")
        results = optimize_for_production(
            model, output_dir=args.output_dir, benchmark=args.benchmark
        )
        for key, value in results["models"].items():
            logger.info(f"  {key}: {value}")

    # Benchmark
    if args.benchmark and torch.cuda.is_available():
        logger.info("Running latency benchmarks...")
        for bs in [1, 32]:
            result = benchmark_latency(model, input_shape=(bs, 3, 224, 224), use_fp16=True)
            logger.info(
                f"  Batch={bs}: mean={result['mean_ms']:.2f}ms p99={result['p99_ms']:.2f}ms"
            )

    logger.info("Export complete.")


if __name__ == "__main__":
    main()
