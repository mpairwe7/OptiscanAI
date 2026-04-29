# Advanced MLOps

DVC pipeline orchestration, hyperparameter optimization, automated retraining triggers, model export, active learning, bias auditing, and precision-rescue training.

## V2 MLOps Additions

| Component | Module | Purpose |
|---|---|---|
| **Active Learning** | `src/active_learning/manager.py` | Uncertainty-based flagging + ophthalmologist review loop + incremental LoRA fine-tuning |
| **Bias Auditing** | `src/governance/bias_auditor.py` | Automated demographic parity + equalized odds across age/sex/device |
| **Immutable Audit Log** | `src/governance/audit_logger.py` | SHA-256 hash-chain, Kafka-ready, thread-safe |
| **Ray Serve** | `src/serving/ray_serve_deployment.py` | Auto-scaling 1-8 replicas, batched inference, health checks |
| **Precision Thresholds** | `src/evaluation/precision_threshold_optimizer.py` | Per-class precision-floor optimization |
| **Precision Plots** | `src/visualization/precision_rescue_plots.py` | 8 new visualization types for v2 analysis |

## DVC Pipeline

The ML pipeline is defined in `dvc.yaml` with 4 reproducible stages:

```
validate_data → train → evaluate → export
```

### Stages

| Stage | Command | Inputs | Outputs |
|---|---|---|---|
| `validate_data` | `scripts/validate_data.py` | `configs/train.yaml`, `src/data/validation.py` | `outputs/validation/data_report.json` |
| `train` | `train.py --config configs/train.yaml` | `configs/`, `src/models/`, `src/training/`, `src/data/` | `outputs/checkpoints/`, `outputs/training_metrics.json` |
| `evaluate` | `scripts/evaluate_model.py` | `outputs/checkpoints/`, `src/evaluation/` | `outputs/evaluation_metrics.json`, `outputs/plots/` |
| `export` | `scripts/export_model.py` | `outputs/checkpoints/` | `outputs/export/` (ONNX + TorchScript) |

### Running the Pipeline

```bash
# Reproduce full pipeline (only runs stages with changed dependencies)
dvc repro

# Run specific stage
dvc repro validate_data

# Force re-run all stages
dvc repro --force

# View pipeline DAG
dvc dag
```

### Parameter Tracking

DVC tracks parameters from `configs/train.yaml`:

```yaml
params:
  - configs/train.yaml:
      - model       # Architecture, hidden_dim, heads, layers
      - training    # LR, loss, epochs, scheduler
      - data        # Batch size, splits, image size
```

Changing any tracked parameter triggers re-execution of dependent stages.

### Metrics Tracking

```bash
# View current metrics
dvc metrics show

# Compare across experiments
dvc metrics diff
```

## Hyperparameter Optimization

Optuna-based HPO with median pruning for efficient search.

### Search Space

| Parameter | Range | Scale |
|---|---|---|
| Learning rate | 1e-5 to 1e-2 | Log |
| Weight decay | 1e-6 to 1e-2 | Log |
| Label smoothing | 0.0 to 0.15 | Linear |
| Gradient clipping | 0.5 to 5.0 | Linear |
| Loss function | focal, asymmetric | Categorical |
| Focal alpha | 0.1 to 0.5 | Linear |
| Focal gamma | 1.0 to 4.0 | Linear |
| Dropout | 0.0 to 0.3 | Linear |
| Hidden dim | 256, 384, 512 | Categorical |
| Scheduler | cosine, step, onecycle | Categorical |
| Warmup epochs | 1 to 5 | Integer |

### Running HPO

```bash
# Default: 20 trials
make hpo

# Custom trials and timeout
PYTHONPATH=. python3 scripts/run_hpo.py \
  --config configs/train.yaml \
  --n-trials 50 \
  --timeout 7200 \
  --output configs/train_optimized.yaml
```

### How It Works

1. Each trial samples hyperparameters from the search space
2. Runs abbreviated training (max 10 epochs, 50 batches/epoch)
3. MedianPruner stops unpromising trials early (after 2 warm-up steps)
4. Best config is written to `configs/train_optimized.yaml`
5. Use optimized config for full training: `python3 train.py --config configs/train_optimized.yaml`

Implementation: `src/training/hpo.py`, `scripts/run_hpo.py`

## Automated Retraining

The retraining trigger system evaluates 4 conditions to determine whether the model should be retrained.

### Triggers

| Trigger | Condition | Priority |
|---|---|---|
| **Time-based** | Model older than 30 days | high |
| **Data volume** | 500+ new samples available | normal |
| **Drift-based** | Critical drift detected, or 5+ accumulated drift events | critical |
| **Performance drop** | F1 macro dropped by > 0.05 from last training | critical |

### Checking Retraining Status

```bash
# Check if retraining is needed
make check-retrain

# Exit codes:
#   0 = no retraining needed
#   2 = retraining recommended

# With current metrics
PYTHONPATH=. python3 scripts/check_retraining.py \
  --metrics outputs/evaluation_metrics.json
```

### Retraining State

Persistent state stored in `outputs/retraining_state.json`:

```json
{
  "last_training": "2026-04-25T10:00:00+00:00",
  "last_metrics": {"f1_macro": 0.72, "auc_roc": 0.85},
  "new_samples_count": 0,
  "drift_events": 0
}
```

### Integration with Monitoring

The retraining trigger integrates with drift detection:

```python
from src.training.retraining import RetrainingTrigger
from src.monitoring.drift import DataDriftDetector

trigger = RetrainingTrigger()
trigger.record_drift(drift_report)  # Accumulates drift events
decision = trigger.evaluate()       # Checks all triggers

if decision.should_retrain:
    print(f"Retrain needed ({decision.priority}): {decision.reason}")
```

Implementation: `src/training/retraining.py`, `scripts/check_retraining.py`

## Model Export

Export trained models to optimized formats for production serving.

### Supported Formats

| Format | Use Case | Config Key |
|---|---|---|
| ONNX (opset 17) | Cross-platform inference, ONNX Runtime | `export.onnx.enabled` |
| TorchScript | PyTorch-native optimized inference | `export.torchscript.enabled` |

### Running Export

```bash
make export

# Or directly
PYTHONPATH=. python3 scripts/export_model.py \
  --config configs/train.yaml \
  --checkpoint outputs/checkpoints/best.pt
```

### Export Configuration

In `configs/train.yaml`:

```yaml
export:
  onnx:
    enabled: true
    opset_version: 17
    output_path: "outputs/export/model.onnx"
  torchscript:
    enabled: true
    output_path: "outputs/export/model.pt"
```

### Output

```
outputs/export/
├── model.onnx              # ONNX model with dynamic batch axis
├── model.pt                # TorchScript traced model
└── export_manifest.json    # Metadata (model name, classes, formats)
```

## Full MLOps Pipeline

```bash
# End-to-end: validate data → train → export → generate governance docs
make mlops-pipeline
```

This runs:

1. `validate-data` — Data quality checks (7 validators)
2. `train` — Multi-GPU DDP training
3. `export` — ONNX + TorchScript export
4. `model-card` — Model card + dataset card generation

## Key Files

| File | Purpose |
|---|---|
| `dvc.yaml` | Pipeline stage definitions |
| `.dvcignore` | Files excluded from DVC tracking |
| `src/training/hpo.py` | Optuna hyperparameter optimization |
| `src/training/retraining.py` | Retraining trigger evaluation |
| `scripts/run_hpo.py` | HPO CLI entry point |
| `scripts/check_retraining.py` | Retraining check CLI |
| `scripts/validate_data.py` | Data validation CLI |
| `scripts/evaluate_model.py` | Model evaluation CLI |
| `scripts/export_model.py` | ONNX/TorchScript export CLI |
| `scripts/generate_model_card.py` | Model + dataset card generator |
