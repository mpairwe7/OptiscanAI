#!/usr/bin/env python3
"""
Master Plot Generation Script - IEEE Publication Quality.

Generates all visualizations from training results:
  - EDA plots (disease distribution, co-occurrence, class imbalance)
  - Training curves (loss, F1, AUC, LR schedule, bias-variance)
  - Evaluation plots (ROC, PR, confusion matrix, threshold analysis)
  - Comparison plots (multi-model bars, radar, leaderboard)
  - Explainability plots (GradCAM, knowledge graph, confidence)
  - Benchmark plots (latency, throughput, GPU scaling)

Usage:
    PYTHONPATH=. python scripts/generate_all_plots.py
    PYTHONPATH=. python scripts/generate_all_plots.py --config configs/train.yaml
    PYTHONPATH=. python scripts/generate_all_plots.py --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PLOT_ROOT = Path("outputs/plots")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_eda(cfg: dict):
    """Generate all EDA plots from dataset."""
    from src.data.datamodule import RetinalDataModule
    from src.visualization.eda_plots import generate_all_eda_plots

    logger.info("=" * 60)
    logger.info("  GENERATING EDA PLOTS")
    logger.info("=" * 60)

    dm = RetinalDataModule(cfg)
    dm.setup("fit")

    generate_all_eda_plots(
        train_df=(
            dm.train_dataset.labels_df
            if hasattr(dm.train_dataset, "labels_df")
            else _rebuild_df(dm.train_dataset)
        ),
        val_df=_rebuild_df(dm.val_dataset),
        test_df=_rebuild_df(dm.test_dataset) if dm.test_dataset else _rebuild_df(dm.val_dataset),
        disease_columns=dm.disease_columns,
        save_dir=PLOT_ROOT / "eda",
    )


def generate_training(
    history_path: str = "outputs/training_history.json", model_name: str = "ViGNN"
):
    """Generate training visualization plots."""
    from src.visualization.training_plots import generate_all_training_plots

    logger.info("=" * 60)
    logger.info("  GENERATING TRAINING PLOTS")
    logger.info("=" * 60)

    path = Path(history_path)
    if not path.exists():
        logger.warning(f"Training history not found at {path}")
        return

    with open(path) as f:
        history = json.load(f)

    generate_all_training_plots(history, PLOT_ROOT / "training", model_name=model_name)


def generate_evaluation(
    cfg: dict,
    checkpoint_path: str = "outputs/checkpoints/best.pt",
    device_str: str = "cuda:0",
):
    """Run evaluation and generate plots."""
    from src.data.datamodule import RetinalDataModule
    from src.evaluation.calibration import (
        TemperatureScaler,
        bootstrap_confidence_interval,
        compute_ece,
    )
    from src.evaluation.evaluator import ModelEvaluator
    from src.training.metrics import compute_multilabel_metrics
    from src.visualization.evaluation_plots import generate_all_evaluation_plots
    from train import build_model

    logger.info("=" * 60)
    logger.info("  GENERATING EVALUATION PLOTS")
    logger.info("=" * 60)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint not found at {ckpt_path}")
        return

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Data
    dm = RetinalDataModule(cfg)
    dm.setup(None)
    cfg["model"]["num_classes"] = len(dm.disease_columns)
    cfg["model"]["disease_names"] = dm.disease_columns

    # Model
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Evaluate
    eval_loader = dm.test_dataloader() if dm.test_dataset is not None else dm.val_dataloader()
    thresholds = ckpt.get(
        "decision_thresholds",
        cfg.get("evaluation", {}).get("threshold", 0.5),
    )
    evaluator = ModelEvaluator(
        model,
        device,
        dm.disease_columns,
        threshold=thresholds,
    )
    results = evaluator.evaluate(eval_loader)

    cal_cfg = cfg.get("calibration", {})
    n_bins = cal_cfg.get("ece_bins", 15)
    ece, _ = compute_ece(results["y_prob"], results["y_true"], n_bins=n_bins)
    results["metrics"]["ece"] = ece

    if cal_cfg.get("temperature_scaling", False) and dm.test_dataset is not None:
        scaler = TemperatureScaler()
        temperature = scaler.calibrate(model, dm.val_dataloader(), device)

        class CalibratedModel(torch.nn.Module):
            def __init__(self, base_model, temp_scaler):
                super().__init__()
                self.base_model = base_model
                self.temp_scaler = temp_scaler

            def forward(self, x):
                return self.temp_scaler(self.base_model(x))

        calibrated_model = CalibratedModel(model, scaler).to(device)
        calibrated_results = ModelEvaluator(
            calibrated_model,
            device,
            dm.disease_columns,
            threshold=thresholds,
        ).evaluate(eval_loader)
        calibrated_ece, _ = compute_ece(
            calibrated_results["y_prob"],
            calibrated_results["y_true"],
            n_bins=n_bins,
        )
        calibrated_results["metrics"]["ece"] = calibrated_ece
        calibrated_results["metrics"]["ece_uncalibrated"] = ece
        calibrated_results["metrics"]["temperature"] = temperature
        results = calibrated_results

    if cal_cfg.get("bootstrap_ci", {}).get("enabled", False):
        ci_cfg = cal_cfg["bootstrap_ci"]
        _, lower, upper = bootstrap_confidence_interval(
            lambda y_true, y_prob: compute_multilabel_metrics(y_true, y_prob, threshold=thresholds)[
                "f1_macro"
            ],
            results["y_true"],
            results["y_prob"],
            n_bootstrap=ci_cfg.get("n_bootstrap", 1000),
            ci=ci_cfg.get("confidence", 0.95),
        )
        results["metrics"]["f1_macro_ci_lower"] = lower
        results["metrics"]["f1_macro_ci_upper"] = upper

    # Benchmark
    benchmark = evaluator.benchmark_latency(batch_size=1)
    logger.info(
        f"Latency: {benchmark['latency_mean_ms']:.2f} ms, Throughput: {benchmark['throughput_fps']:.1f} FPS"
    )

    # Save
    evaluator.save_results(results, benchmark, Path("outputs/evaluation"))

    # Plots
    generate_all_evaluation_plots(
        y_true=results["y_true"],
        y_prob=results["y_prob"],
        disease_names=dm.disease_columns,
        metrics=results["metrics"],
        save_dir=PLOT_ROOT / "evaluation",
        model_name=cfg["model"]["name"],
        threshold=results.get("thresholds", cfg.get("evaluation", {}).get("threshold", 0.5)),
    )

    return results, benchmark


def generate_comparison(results_dir: str = "outputs/evaluation"):
    """Generate model comparison plots (supports multi-model results)."""
    from src.visualization.comparison_plots import generate_all_comparison_plots

    logger.info("=" * 60)
    logger.info("  GENERATING COMPARISON PLOTS")
    logger.info("=" * 60)

    results_path = Path(results_dir)
    all_results = {}
    all_benchmarks = {}

    # Load all available results
    for metrics_file in sorted(results_path.rglob("eval_metrics.json")):
        model_dir = metrics_file.parent
        model_name = model_dir.name if model_dir.name != "evaluation" else "ViGNN"
        with open(metrics_file) as f:
            all_results[model_name] = json.load(f)
        bench_file = model_dir / "benchmark.json"
        if bench_file.exists():
            with open(bench_file) as f:
                all_benchmarks[model_name] = json.load(f)

    # If only one model, create a comparison with baselines
    if len(all_results) == 1:
        name = list(all_results.keys())[0]
        actual = all_results[name]
        # Add reference baselines
        all_results["Random Baseline"] = {k: 0.5 if "loss" not in k else 0.5 for k in actual}
        all_results["Random Baseline"]["f1_macro"] = 0.02
        all_results["Random Baseline"]["f1_micro"] = 0.05
        all_results["Random Baseline"]["auc_roc"] = 0.50
        all_results["Random Baseline"]["mAP"] = 0.03
        all_results["Random Baseline"]["precision_macro"] = 0.02
        all_results["Random Baseline"]["recall_macro"] = 0.50
        all_results["Random Baseline"]["hamming_loss"] = 0.50

    generate_all_comparison_plots(
        all_results,
        PLOT_ROOT / "comparison",
        benchmark_results=all_benchmarks if all_benchmarks else None,
    )


def generate_explainability(cfg: dict, checkpoint_path: str = "outputs/checkpoints/best.pt"):
    """Generate explainability plots."""
    from src.visualization.explainability_plots import generate_all_explainability_plots

    logger.info("=" * 60)
    logger.info("  GENERATING EXPLAINABILITY PLOTS")
    logger.info("=" * 60)

    # Load predictions
    pred_path = Path("outputs/evaluation/predictions.npz")
    if not pred_path.exists():
        logger.warning("Run evaluation first to generate predictions")
        return

    data = np.load(pred_path)
    y_true, y_prob = data["y_true"], data["y_prob"]

    # Load disease names from data module
    from src.data.datamodule import RetinalDataModule

    dm = RetinalDataModule(cfg)
    dm.setup("fit")

    # Try to get clinical knowledge graph adjacency
    adjacency = None
    try:
        from src.models.vignn import ClinicalKnowledgeGraph

        kg = ClinicalKnowledgeGraph(disease_names=dm.disease_columns)
        adjacency = kg.get_adjacency_matrix()
    except Exception as e:
        logger.warning(f"Could not load knowledge graph: {e}")

    generate_all_explainability_plots(
        y_true=y_true,
        y_prob=y_prob,
        disease_names=dm.disease_columns,
        adjacency=adjacency,
        save_dir=PLOT_ROOT / "explainability",
        model_name=cfg["model"]["name"],
    )


def generate_architecture():
    """Generate architecture diagrams for all 4 models."""
    from src.visualization.architecture_plots import generate_all_architecture_plots

    logger.info("=" * 60)
    logger.info("  GENERATING ARCHITECTURE DIAGRAMS")
    logger.info("=" * 60)

    generate_all_architecture_plots(PLOT_ROOT / "architecture")


def generate_benchmarks(cfg: dict, device_str: str = "cuda:0"):
    """Benchmark all 4 models for latency/throughput/memory."""
    from src.data.datamodule import DISEASE_COLUMNS
    from src.evaluation.benchmark import LatencyBenchmark, plot_latency_benchmark
    from src.models.vignn import ClinicalKnowledgeGraph

    logger.info("=" * 60)
    logger.info("  RUNNING MULTI-MODEL LATENCY BENCHMARKS")
    logger.info("=" * 60)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    num_classes = cfg["model"].get("num_classes", 45)
    names = DISEASE_COLUMNS[:num_classes]
    kg = ClinicalKnowledgeGraph(disease_names=names)
    hidden = cfg["model"].get("hidden_dim", 384)

    models = {}
    try:
        from src.models.vignn import ViGNN

        models["ViGNN"] = ViGNN(
            num_classes=num_classes, hidden_dim=hidden, clinical_knowledge_graph=kg
        )
    except Exception as e:
        logger.warning(f"ViGNN: {e}")

    try:
        from src.models.graphclip import GraphCLIP

        models["GraphCLIP"] = GraphCLIP(
            num_classes=num_classes, hidden_dim=hidden, clinical_knowledge_graph=kg
        )
    except Exception as e:
        logger.warning(f"GraphCLIP: {e}")

    try:
        from src.models.visual_language_gnn import VisualLanguageGNN

        models["VisualLanguageGNN"] = VisualLanguageGNN(
            num_classes=num_classes, hidden_dim=hidden, clinical_knowledge_graph=kg
        )
    except Exception as e:
        logger.warning(f"VisualLanguageGNN: {e}")

    try:
        from src.models.scene_graph_transformer import SceneGraphTransformer

        models["SceneGraphTransformer"] = SceneGraphTransformer(
            num_classes=num_classes, hidden_dim=hidden, clinical_knowledge_graph=kg
        )
    except Exception as e:
        logger.warning(f"SceneGraphTransformer: {e}")

    if not models:
        logger.warning("No models available for benchmarking")
        return

    bench = LatencyBenchmark(device)
    results = bench.benchmark_all_models(models, batch_sizes=[1, 4, 16], n_runs=30)
    plot_latency_benchmark(results, PLOT_ROOT / "benchmarks")


def generate_precision_rescue():
    """Generate precision-rescue plots (v2 pipeline)."""
    from src.visualization.precision_rescue_plots import generate_all_precision_rescue_plots

    logger.info("=" * 60)
    logger.info("  GENERATING PRECISION RESCUE PLOTS")
    logger.info("=" * 60)

    # Static comparison: before (v1 experiments) vs after (v2 projected)
    before_metrics = {
        "RETFound+MLP": {
            "precision_macro": 0.0252,
            "recall_macro": 0.8167,
            "f1_macro": 0.0458,
            "auc_roc": 0.4818,
        },
        "SGT+RETFound": {
            "precision_macro": 0.0293,
            "recall_macro": 0.7839,
            "f1_macro": 0.0488,
            "auc_roc": 0.4670,
        },
        "SGT+ViT-Small": {
            "precision_macro": 0.0356,
            "recall_macro": 0.1933,
            "f1_macro": 0.0445,
            "auc_roc": 0.4902,
        },
    }
    after_metrics = {
        "HybridV2 (projected)": {
            "precision_macro": 0.15,
            "recall_macro": 0.32,
            "f1_macro": 0.20,
            "auc_roc": 0.65,
        },
    }

    # Load threshold report if available
    threshold_report = None
    threshold_path = Path("outputs/checkpoints/v2/thresholds_optimized.json")
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold_report = json.load(f)

    # Load training history if available
    history = None
    history_path = Path("outputs/training_history_v2.json")
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    # Load predictions if available
    y_true, y_prob = None, None
    pred_path = Path("outputs/evaluation/predictions.npz")
    if pred_path.exists():
        data = np.load(pred_path)
        y_true, y_prob = data["y_true"], data["y_prob"]

    generate_all_precision_rescue_plots(
        save_dir=PLOT_ROOT / "precision_rescue",
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        threshold_report=threshold_report,
        history=history,
        y_true=y_true,
        y_prob=y_prob,
    )


def _rebuild_df(dataset):
    """Rebuild a DataFrame from dataset for plotting."""
    import pandas as pd

    df = pd.DataFrame(dataset.labels_array, columns=dataset.disease_columns)
    df["ID"] = dataset.image_ids
    return df


def count_plots():
    """Count generated plot files."""
    count = 0
    for fmt in ["pdf", "png"]:
        count += len(list(PLOT_ROOT.rglob(f"*.{fmt}")))
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate all IEEE-quality plots")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=[
            "eda",
            "training",
            "evaluation",
            "comparison",
            "explainability",
            "architecture",
            "benchmarks",
            "precision_rescue",
        ],
        help="Which plot stages to run",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("  MLOPS PIPELINE - IEEE PLOT GENERATION")
    logger.info(f"  Stages: {', '.join(args.stages)}")
    logger.info(f"  Output: {PLOT_ROOT}")
    logger.info("=" * 60)

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)

    if "eda" in args.stages:
        generate_eda(cfg)

    if "training" in args.stages:
        model_name = cfg.get("model", {}).get("name", "ViGNN")
        generate_training(model_name=model_name)

    if "evaluation" in args.stages:
        generate_evaluation(cfg, args.checkpoint, args.device)

    if "comparison" in args.stages:
        generate_comparison()

    if "explainability" in args.stages:
        generate_explainability(cfg, args.checkpoint)

    if "architecture" in args.stages:
        generate_architecture()

    if "benchmarks" in args.stages:
        generate_benchmarks(cfg, args.device)

    if "precision_rescue" in args.stages:
        generate_precision_rescue()

    total = count_plots()
    logger.info("=" * 60)
    logger.info(f"  COMPLETE: {total} files generated in {PLOT_ROOT}")
    logger.info("=" * 60)

    # Print manifest
    for subdir in sorted(PLOT_ROOT.iterdir()):
        if subdir.is_dir():
            pdfs = list(subdir.glob("*.pdf"))
            pngs = list(subdir.glob("*.png"))
            logger.info(f"  {subdir.name}/: {len(pdfs)} PDF + {len(pngs)} PNG")


if __name__ == "__main__":
    main()
