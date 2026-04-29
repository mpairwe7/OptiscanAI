# Testing

198 tests covering models (v1+v2), training, data, API, monitoring, governance, safety gates, and validation.

## Test Suite Overview

| File | Tests | Coverage |
|---|---|---|
| **`test_fundus_gate_v2.py`** | **24** | **Gate v2**: GateResultV2, fusion logic, fallback, visual evidence, thread safety |
| **`test_fundus_gate_v2_adversarial.py`** | **33** | **Adversarial**: 7 categories of non-fundus edge cases (30+ synthetic images) |
| `test_hybrid_v2.py` | 19 | V2 model: forward, TTA, clinical reasoning, bottleneck, ASL loss, thresholds |
| `test_hybrid_model.py` | 16 | V1 hybrid: forward, uncertainty, MoE, clinical reasoning, LoRA, encoder |
| `test_quantization.py` | 7 | INT8/FP16 quantization, distillation loss, benchmarks |
| `test_active_learning.py` | 12 | Flagging, review queue, approval/correction, retraining trigger |
| `test_bias_auditor.py` | 8 | Bias audit, fairness detection, subgroups, report saving |
| `test_models.py` | 9 | Legacy models: forward pass (all 4 architectures), knowledge graph |
| `test_losses.py` | 12 | FocalLoss, AsymmetricLoss, `build_loss` factory, label smoothing |
| `test_metrics.py` | 8 | MetricTracker update/reset/compute, optimal thresholds |
| `test_dataset.py` | 8 | Dataset creation, length, getitem, class weights |
| `test_data_validation.py` | 14 | Schema, label range, null IDs, class distribution |
| `test_api.py` | 12 | Root, health, diseases, predict, gate status/validate, gate rejection |
| `test_monitoring.py` | 11 | HealthMonitor, SLA compliance, DataDrift, ModelDrift |
| `test_mixup.py` | 1 | MixUp/CutMix configuration |
| `test_train_entry.py` | 1 | Training entry point builds model correctly |

## Running Tests

```bash
# All tests
make test

# Fail-fast mode
make test-fast

# Specific test file
PYTHONPATH=. pytest tests/test_models.py -v

# Fundus gate v2 tests (57 tests: 24 unit + 33 adversarial)
make test-gate

# Gate latency benchmark (p50/p95/p99)
make benchmark-gate

# With coverage
PYTHONPATH=. pytest tests/ --cov=src --cov-report=term-missing
```

## Data Validation

The `DataValidator` class (`src/data/validation.py`) runs 7 automated checks on training data before pipeline execution:

| Check | Severity | Description |
|---|---|---|
| `schema_validation` | error | All required columns (ID + 45 diseases) present |
| `label_range` | error | Labels are binary (0 or 1) |
| `null_id_check` | error | No null image IDs |
| `class_distribution` | warning | Every class has at least 1 sample |
| `duplicate_id_check` | error | No duplicate image IDs |
| `label_correlation` | warning | No suspiciously high correlations (> 0.99) |
| `image_existence` | error | Images exist on disk for all IDs |

### Running Validation

```bash
# Standalone
make validate-data

# As part of DVC pipeline (runs before training)
dvc repro validate_data
```

### Validation Report Output

```json
{
  "passed": true,
  "summary": {"total": 7, "passed": 7, "failed": 0, "pass_rate": 1.0},
  "checks": [
    {"check": "schema_validation", "passed": true, "details": "All required columns present", "severity": "error"},
    ...
  ]
}
```

## CI Quality Gates

The GitHub Actions ML pipeline (`.github/workflows/ml-pipeline.yml`) runs:

1. **Lint** — ruff + black format check on `src/`, `backend/`, `scripts/`, `train.py`
2. **Test** — Full pytest suite (CPU-only, no GPU required)
3. **Model Validation** — Import and forward-pass all architectures
4. **Data Quality** — Validate data pipeline structure (on main branch)

Tests must pass before model validation and deployment can proceed.

## Writing New Tests

Shared fixtures are in `tests/conftest.py`:

| Fixture | Description |
|---|---|
| `sample_image` | Random 224x224 RGB PIL Image |
| `sample_batch` | (4, 3, 224, 224) tensor + (4, 45) binary labels |
| `disease_columns` | List of 45 disease column names |
| `train_config` | Minimal training config dict |
| `sample_labels_df` | 20-row DataFrame with ID + 45 disease columns |
| `sample_img_dir` | Temp directory with synthetic PNG images |
| `_safe_tmpdir` | Temporary directory (ownership-safe) |

## Key Files

| File | Purpose |
|---|---|
| `tests/conftest.py` | Shared fixtures |
| `tests/test_*.py` | Test modules (9 files) |
| `src/data/validation.py` | DataValidator with 7 checks |
| `scripts/validate_data.py` | CLI script for data validation |
| `.github/workflows/ml-pipeline.yml` | CI pipeline with test + validation gates |
