"""
Full evaluation pipeline for trained models.
Computes all metrics, runs inference, and collects data for visualization.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_multilabel_metrics

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """End-to-end model evaluation with metrics, predictions, and benchmarks."""

    def __init__(
        self,
        model,
        device,
        disease_names: list[str],
        threshold: float | list[float] | np.ndarray = 0.5,
    ):
        self.model = model
        self.device = device
        self.disease_names = disease_names
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Run full evaluation. Returns metrics + raw predictions."""
        self.model.eval()
        all_logits, all_targets = [], []

        for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

        logits = torch.cat(all_logits).numpy()
        targets = torch.cat(all_targets).numpy()
        probs = 1 / (1 + np.exp(-logits))
        threshold_array = np.asarray(self.threshold) if not np.isscalar(self.threshold) else self.threshold
        if np.isscalar(threshold_array):
            preds = (probs > float(threshold_array)).astype(float)
        else:
            preds = (probs > threshold_array.reshape(1, -1)).astype(float)

        metrics = self._compute_metrics(targets, probs)
        per_class = self._compute_per_class(targets, preds, probs)

        return {
            "metrics": metrics,
            "per_class": per_class,
            "y_true": targets,
            "y_prob": probs,
            "y_pred": preds,
            "thresholds": (
                threshold_array.tolist()
                if not np.isscalar(threshold_array)
                else float(threshold_array)
            ),
        }

    def _compute_metrics(self, y_true, y_prob) -> dict:
        return compute_multilabel_metrics(y_true, y_prob, threshold=self.threshold)

    def _compute_per_class(self, y_true, y_pred, y_prob) -> list[dict]:
        results = []
        for i, name in enumerate(self.disease_names):
            d = {
                "disease": name,
                "support": int(y_true[:, i].sum()),
                "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            }
            try:
                if y_true[:, i].sum() > 0:
                    d["auc"] = roc_auc_score(y_true[:, i], y_prob[:, i])
                    d["ap"] = average_precision_score(y_true[:, i], y_prob[:, i])
                else:
                    d["auc"] = 0.0
                    d["ap"] = 0.0
            except ValueError:
                d["auc"] = 0.0
                d["ap"] = 0.0
            results.append(d)
        return results

    def benchmark_latency(self, batch_size: int = 1, warmup: int = 10, n_runs: int = 100) -> dict:
        """Benchmark inference latency."""
        self.model.eval()
        dummy = torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(dummy)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(n_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)

        # GPU memory
        gpu_mem = 0
        if self.device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(self.device) / 1e6

        params = sum(p.numel() for p in self.model.parameters()) / 1e6

        return {
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_median_ms": float(np.median(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "latency_std_ms": float(np.std(latencies)),
            "throughput_fps": float(1000 / np.mean(latencies) * batch_size),
            "gpu_mem_MB": float(gpu_mem),
            "params_M": float(params),
            "batch_size": batch_size,
        }

    def save_results(self, eval_results: dict, benchmark: dict, save_dir: Path):
        """Save evaluation results to JSON."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        with open(save_dir / "eval_metrics.json", "w") as f:
            json.dump(eval_results["metrics"], f, indent=2)

        # Per-class
        with open(save_dir / "per_class_metrics.json", "w") as f:
            json.dump(eval_results["per_class"], f, indent=2)

        # Benchmark
        with open(save_dir / "benchmark.json", "w") as f:
            json.dump(benchmark, f, indent=2)

        if "thresholds" in eval_results:
            with open(save_dir / "thresholds.json", "w") as f:
                json.dump(eval_results["thresholds"], f, indent=2)

        # Predictions (compact)
        np.savez_compressed(
            save_dir / "predictions.npz",
            y_true=eval_results["y_true"],
            y_prob=eval_results["y_prob"],
            y_pred=eval_results["y_pred"],
        )

        logger.info(f"Results saved to {save_dir}")
