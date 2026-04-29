# Production Improvements

55+ production gaps addressed across 7 implementation stages adding 24 modules, followed by 4 MLOps hardening phases adding 20+ more modules for testing, security, governance, and advanced pipeline orchestration.

## V2 Precision Rescue (April 2026)

The most critical production improvement: fixing the precision crisis (P=0.025) that made the model unusable in practice.

| Module | Purpose |
|---|---|
| `src/models/retinal_foundation_hybrid_v2.py` | Bottleneck head (drop 0.5/0.3), 1-layer graph, no MoE, staged unfreeze, TTA |
| `src/evaluation/precision_threshold_optimizer.py` | Per-class thresholds with precision floor >= 0.10 |
| `src/data/fundus_gate_learned.py` | MobileNetV3-Small learned fundus gate (<5ms) |
| `src/data/fundus_gate_v2.py` | **V2 fusion gate**: statistical + learned fusion, explainable rejection, visual evidence |
| `src/monitoring/gate_monitor.py` | Gate monitoring: pass/reject rates, disagreement tracking, alerting |
| `backend/app/routers/gate.py` | Gate status + validate debug endpoints |
| `src/visualization/precision_rescue_plots.py` | 8 new plot types: before/after, ASL vs Focal, thresholds, class filtering |
| `scripts/train_hybrid_precision_v2.py` | Training with ASL (gamma_pos=0), class-balanced sampling, precision early-stop |
| `scripts/export_production_v2.py` | Dtype-safe ONNX/TorchScript/INT8 export (fixes quantization failure) |
| `configs/hybrid_precision_2026.yaml` | All v2 hyperparameters |

See [Precision Rescue Verification](precision-rescue-verification.md) and [Architecture](architecture-retinal-foundation-hybrid.md#8-v2-precision-rescue-architecture-april-2026) for details.

## Stage 1: Data Quality + Validation

**Module**: `src/data/validation.py`

| Component | Purpose |
|---|---|
| `DataValidator` | 7 automated checks: schema, label range, null IDs, distribution, duplicates, correlation, image existence |
| `ValidationReport` | Structured report (pass rate, per-check details, severity levels) |

**Fix**: `dataset.py.__getitem__` catches PIL errors and returns a black placeholder + zeroed labels instead of crashing.

**Config**: `data_validation.validate_images: true`

**CLI**: `make validate-data` or `dvc repro validate_data`

## Stage 2: Advanced Augmentation

**Modules**: `src/data/mixup.py`, updates to `src/data/augmentation.py`

| Component | Purpose |
|---|---|
| `MixUpCutMix` | Batch-level mixing (alpha=0.2/1.0), works with multi-label float targets |
| `get_tta_transforms()` | Test-time augmentation (hflip, vflip, rotate variants) |
| `TTAPredictor` | Averages logits across TTA variants for better calibration |

**Config**: `augmentation.mixup_alpha: 0.2`, `cutmix_alpha: 1.0`

## Stage 3: Advanced Training

**Modules**: `src/training/ema.py`, `src/training/lr_finder.py`, updated `src/training/losses.py`

| Component | Purpose |
|---|---|
| `ModelEMA` | Shadow weights (decay=0.9999), typically +1-2% improvement |
| `LRFinder` | Exponential LR sweep with IEEE plot |
| **Label smoothing** | `FocalLoss` and `AsymmetricLoss` apply `label_smoothing` from config |
| `set_seed()` | Reproducibility across torch/numpy/random/cudnn |

**Config**: `training.ema.enabled: true`, `training.seed: 42`

## Stage 4: Advanced Evaluation

**Modules**: `src/evaluation/calibration.py`, `src/evaluation/statistical_tests.py`

| Component | Purpose |
|---|---|
| `TemperatureScaler` | Post-hoc calibration via learned temperature (Guo et al. 2017) |
| `compute_ece()` | Expected Calibration Error with reliability diagram data |
| `bootstrap_confidence_interval()` | 95% CI on F1, AUC, mAP (1000 samples) |
| `mcnemar_test()` | Statistical significance between two models |
| `paired_bootstrap_test()` | Bootstrap p-value for metric differences |
| `wilcoxon_test()` | Non-parametric test for k-fold score comparison |

**Config**: `calibration.temperature_scaling: true`, `calibration.bootstrap_ci.enabled: true`

## Stage 5: Production Export

**Modules**: `src/export/onnx_export.py`, `torchscript_export.py`, `quantization.py`, `scripts/export_model.py`

| Component | Purpose |
|---|---|
| `ONNXExporter` | Export with opset 17, dynamic batch axis |
| `TorchScriptExporter` | torch.jit.trace for optimized inference |
| `QuantizationBenchmark` | FP32 vs FP16 vs INT8 latency/size comparison |

**CLI**: `make export`

**Config**: `export.onnx.enabled: true`, `export.quantization.benchmark_int8: true`

## Stage 6: Enhanced Visualization

**Modules**: `src/visualization/calibration_plots.py`, `failure_analysis_plots.py`, `gradient_plots.py`

| Plot | Description |
|---|---|
| `plot_reliability_diagram` | Accuracy vs confidence bins with ECE annotation |
| `plot_calibration_before_after` | Side-by-side before/after temperature scaling |
| `plot_confidence_histogram` | Positive vs negative probability distributions |
| `plot_per_class_error_breakdown` | Stacked FP+FN bars per disease |
| `plot_error_by_label_cardinality` | Error rate vs number of positive labels |
| `plot_gradient_norms` | Gradient norm history with explosion/vanishing thresholds |
| `plot_lr_vs_loss` | Learning rate overlaid with loss curve |

## Stage 7: Production Monitoring

**Modules**: `src/monitoring/drift.py`, `src/monitoring/health.py`

| Component | Purpose |
|---|---|
| `DataDriftDetector` | KS test + PSI on pixel intensity vs training baseline |
| `ModelDriftDetector` | Confidence drop + per-class distribution shift detection |
| `HealthMonitor` | Latency P50/P95/P99, throughput, error rate, SLA compliance |

**API**: `GET /health/model` returns detailed HealthReport

**Config**: `monitoring.enabled: true`, `monitoring.sla.max_latency_p99_ms: 100`

## MLOps Hardening (Phases 1-4)

Beyond the 7 stages above, 4 additional hardening phases were implemented:

### Phase 1: Foundations

| Component | Files | Purpose |
|---|---|---|
| Unit tests | `tests/test_*.py` (9 files) | 68 tests covering all modules |
| Data validation | `src/data/validation.py` | 7 quality checks with structured reporting |
| DVC pipeline | `dvc.yaml` | 4-stage reproducible pipeline |
| Pipeline scripts | `scripts/validate_data.py`, `evaluate_model.py`, `export_model.py` | CLI entry points |

See [Testing](09-testing.md) and [Advanced MLOps](12-advanced-mlops.md).

### Phase 2: Production Hardening

| Component | Files | Purpose |
|---|---|---|
| JWT auth | `backend/app/core/auth.py`, `routers/auth.py` | Token-based access control |
| Structured logging | `backend/app/core/logging_config.py` | JSON logs with request correlation |
| Prediction logging | `backend/app/core/prediction_logger.py` | Audit trail for all inferences |
| Rate limiting | `backend/app/middleware/rate_limit.py` | Per-IP request throttling |
| Request tracing | `backend/app/middleware/request_id.py` | X-Request-ID propagation |
| Security CI | `.github/workflows/security-scan.yml` | pip-audit, Trivy, SBOM |

See [Security](10-security.md).

### Phase 3: Compliance & Governance

| Component | Files | Purpose |
|---|---|---|
| Model cards | `src/governance/model_card.py` | Standardized model documentation |
| Dataset cards | `src/governance/dataset_card.py` | Gebru et al. datasheet framework |
| Fairness evaluation | `src/governance/fairness.py` | Subgroup performance analysis |
| Audit trail | `src/governance/audit.py` | Immutable event log with checksums |
| Human review | `src/governance/human_review.py`, `routers/review.py` | Clinical review queue |

See [Governance](11-governance.md).

### Phase 4: Advanced MLOps

| Component | Files | Purpose |
|---|---|---|
| HPO | `src/training/hpo.py` | Optuna hyperparameter optimization |
| Retraining triggers | `src/training/retraining.py` | Automated retraining evaluation |
| Updated CI | `.github/workflows/ml-pipeline.yml` | Modern quality gates |

See [Advanced MLOps](12-advanced-mlops.md).

## Complete File Summary

| Category | New Files | Modified Files |
|---|---|---|
| Stages 1-7 | 15 new modules | 5 modified |
| Phase 1 (Tests + DVC) | 12 files | 1 (pyproject.toml) |
| Phase 2 (Security) | 8 files | 3 (config, main, predict) |
| Phase 3 (Governance) | 8 files | 1 (main.py) |
| Phase 4 (MLOps) | 4 files | 3 (pyproject, Makefile, CI) |
| **Total** | **47 new files** | **13 modified files** |
