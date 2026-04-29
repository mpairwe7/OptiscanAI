# Evaluation

## Metrics

| Metric | Description | v1 Target | v2 Target |
|---|---|---|---|
| **Precision Macro** | Per-class precision averaged (**v2 primary**) | Any | >= 0.12 |
| **F1 Macro** | Per-class F1 averaged | Higher | >= 0.18 |
| **AUC-ROC** | Area under ROC curve (macro average) | > 0.5 | >= 0.62 |
| **Recall Macro** | Per-class recall averaged | Higher | >= 0.25 |
| **mAP** | Mean Average Precision | Higher | Higher |
| **Hamming Loss** | Fraction of wrong labels | Lower | Lower |

> **Important**: In v2 (precision rescue), the primary metric is **Precision Macro**, not F1.
> The v1 experiments showed P=0.025 with R=0.82 — the model predicted positive for everything.
> V2 intentionally trades recall for precision.

## Running Evaluation

```bash
# After training, generate evaluation plots
make plots-eval

# Or run full evaluation + comparison + explainability
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 scripts/generate_all_plots.py \
    --stages evaluation comparison explainability \
    --checkpoint outputs/checkpoints/best.pt
```

## Pipeline

```
ModelEvaluator (src/evaluation/evaluator.py)
  .evaluate(dataloader)     -> Full inference, compute all metrics
  .benchmark_latency()      -> Latency/throughput/memory profiling
  .save_results()           -> Save metrics + predictions to JSON/NPZ
```

## Generated Plots (IEEE 300 DPI)

### Evaluation (`outputs/plots/evaluation/`)
- **ROC curves** — Per-class ROC with AUC ranking (top-10)
- **PR curves** — Precision-Recall with AP ranking
- **Confusion matrix** — TP/FP/FN/TN bars + per-class heatmap
- **Threshold analysis** — Optimal threshold search across F1/Hamming
- **Precision-floor threshold analysis** — (v2) Per-class precision improvement with floor=0.10
- **Metrics summary** — Publication-ready table

### Precision Rescue (`outputs/plots/precision_rescue/`) — NEW
- **Before/After comparison** — v1 (45 classes) vs v2 (precision rescue) metrics
- **Threshold heatmap** — Per-class optimized thresholds with precision/recall achieved
- **ASL vs Focal Loss** — Loss weight curves showing FP suppression advantage
- **Class filtering** — 45-class distribution with keep/drop visualization
- **Staged unfreezing** — Loss/precision/recall dynamics with unfreeze transition
- **Precision-Recall trade-off** — Operating point surface with precision floor
- **TTA improvement** — Metrics with and without Test-Time Augmentation
- **HybridV2 architecture** — Architecture diagram with all 7 strategies annotated

### Comparison (`outputs/plots/comparison/`)
- **Bar comparison** — Multi-model metric bars (now includes HybridV1, HybridV2)
- **Radar chart** — Multi-dimensional capability profile
- **Efficiency plot** — Latency vs accuracy Pareto frontier
- **Leaderboard** — Ranked metrics table

### Explainability (`outputs/plots/explainability/`)
- **Clinical knowledge graph** — Disease relationships (adjacency + network)
- **Confidence distribution** — Positive vs negative class probabilities
- **GradCAM grid** — Attention heatmaps (when model + images available)

## Experimental Results

### V1 Experiments (45 classes, precision crisis)

| Experiment | F1 | AUC | Precision | Recall |
|---|---|---|---|---|
| RETFound + MLP | 0.0458 | 0.4818 | 0.0252 | 0.8167 |
| SGT + RETFound | 0.0488 | 0.4670 | 0.0293 | 0.7839 |
| SGT + ViT-Small | 0.0445 | 0.4902 | 0.0356 | 0.1933 |
| K-Fold best | 0.0625 | 0.5789 | ~0.04 | ~0.30 |

### V2 Actual Results (24 classes, precision rescue, 25 epochs)

| Metric | v1 Actual | v2 Target | **v2 ACTUAL** | vs v1 |
|---|---|---|---|---|
| Precision (macro) | 0.025 | 0.12-0.18 | **0.312** | **12.5x** |
| Recall (macro) | 0.82 | 0.25-0.40 | **0.456** | Trade-off |
| F1 (macro) | 0.046 | 0.18-0.25 | **0.362** | **7.9x** |
| AUC (macro) | 0.48 | 0.62-0.68 | **0.888** | **+85%** |
| Accuracy (macro) | N/A | N/A | **0.954** | New |
| Model size (INT8) | Failed | <65 MB | 296 MB | Fixed |

## Benchmark Results

| Model | Latency | FPS | Memory | Params |
|---|---|---|---|---|
| **HybridV2** (A100 FP16) | ~4ms | ~250 | 0.7 GB | 315M (11M trainable) |
| **HybridV2** (A100 INT8) | ~3ms | ~330 | 0.4 GB | 315M (quantized) |
| ViGNN (legacy) | 25.3ms | 39.6 | 204 MB | 26.1M |
| GraphCLIP (legacy) | 28.3ms | 35.4 | 291 MB | 24.8M |
| SceneGraphTransformer (legacy) | 26.4ms | 37.8 | 519 MB | 31.2M |

## Key Files

| File | Purpose |
|---|---|
| `src/evaluation/evaluator.py` | ModelEvaluator with metrics + benchmark |
| `src/evaluation/precision_threshold_optimizer.py` | **V2** per-class precision-floor thresholds |
| `src/evaluation/benchmark.py` | LatencyBenchmark + plot generation |
| `scripts/generate_all_plots.py` | Master plot orchestrator (includes precision_rescue stage) |
| `src/visualization/evaluation_plots.py` | ROC, PR, confusion, threshold, precision-floor plots |
| `src/visualization/precision_rescue_plots.py` | **V2** before/after, ASL, class filtering, staged unfreeze |
| `src/visualization/comparison_plots.py` | Multi-model comparison charts |
| `src/visualization/explainability_plots.py` | KG, confidence, GradCAM |
