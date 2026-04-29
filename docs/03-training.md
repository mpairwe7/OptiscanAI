# Training

## Architecture

Multi-GPU distributed training using PyTorch native DDP (DistributedDataParallel) via `torchrun`.

### Production Model (Recommended)

| Model | Params | Trainable | Key Innovation | Config |
|---|---|---|---|---|
| **RetinalFoundationHybridV2** | 315M | ~11M (LoRA) | RETFound ViT-L + bottleneck head + ASL + precision-floor thresholds | `hybrid_v2` |

The v2 model addresses the precision crisis (P=0.025) with 7 strategies:
1. **Asymmetric Loss** (gamma_pos=0, gamma_neg=4) — suppresses false positives
2. **Class filtering** — drops ultra-rare classes (<10 samples): 45 -> ~25-28 classes
3. **Precision-floor thresholds** — per-class thresholds ensuring precision >= 0.10
4. **Bottleneck head** — 512 (drop 0.5) -> 128 (drop 0.3) -> N classes
5. **Staged unfreezing** — head-only for 10 epochs, then last 4 ViT blocks at 1e-6
6. **LoRA adapters** — rank 16, alpha 32 on backbone QKV projections
7. **Test-Time Augmentation** — average 6 augmented views at inference

### Legacy Models

| Model | Params | Key Innovation | Config name |
|---|---|---|---|
| ViGNN | 26.1M | Graph message passing + disease prototypes | `vignn` |
| GraphCLIP | 24.8M | Dynamic graph adjacency + sparse attention | `graphclip` |
| VisualLanguageGNN | 24.3M | Cross-modal visual-text fusion | `visual_language_gnn` |
| SceneGraphTransformer | 31.2M | Ensemble branches + uncertainty estimation | `scene_graph_transformer` |

All legacy models share:
- **MultiResolutionEncoder** (ViT-Small backbone, 3 resolution levels)
- **ClinicalKnowledgeGraph** (45 diseases, 144 clinical relationships)
- **SparseTopKAttention** (O(n*k) complexity)

## Quick Start

```bash
# Precision-rescue training (recommended)
python3 scripts/train_hybrid_precision_v2.py --config configs/hybrid_precision_2026.yaml

# Multi-GPU precision-rescue
torchrun --nproc_per_node=4 scripts/train_hybrid_precision_v2.py --config configs/hybrid_precision_2026.yaml

# Legacy: single GPU
make train-1gpu

# Legacy: all 8 GPUs
make train
```

## Configuration

All training parameters in `configs/train.yaml`:

```yaml
training:
  max_epochs: 30
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 3
  scheduler: "cosine"
  loss: "asymmetric"        # asymmetric | focal | bce
  gradient_clip_val: 1.0
  precision: "16-mixed"     # FP16 mixed precision
  early_stopping_patience: 7

distributed:
  strategy: "ddp"
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]
  sync_batchnorm: true
```

## Training Pipeline

```
train.py
  1. setup_distributed()     -> Init NCCL process group
  2. RetinalDataModule       -> Download data (rank 0), setup splits
  3. build_model()           -> Create model + knowledge graph
  4. build_loss()            -> FocalLoss / AsymmetricLoss with pos_weights
  5. DDPTrainer              -> Wrap model in DDP, create optimizer
  6. trainer.train()         -> Main loop with:
     - Warmup LR schedule
     - Mixed precision (AMP)
     - Gradient accumulation + clipping
     - Per-epoch validation
     - Metric tracking (F1, AUC, mAP)
     - Early stopping
     - Checkpoint saving (top-k + last + best)
     - W&B logging (optional)
```

## Multi-GPU Launch

`scripts/train_multigpu.sh` uses `torchrun`:

```bash
torchrun \
    --standalone \
    --nproc_per_node=8 \
    --master_port=29500 \
    train.py --config configs/train.yaml
```

Effective batch size = `batch_size * num_gpus * gradient_accumulation_steps`
= 32 * 8 * 1 = **256**

## Loss Functions

| Loss | Best For | Config |
|---|---|---|
| **Asymmetric Loss** (default) | Extreme class imbalance, 2026 SOTA | `loss: "asymmetric"` |
| **Focal Loss** | Moderate imbalance | `loss: "focal"` |
| **BCE** | Balanced data | `loss: "bce"` |

## Outputs

```
outputs/
  checkpoints/
    best.pt              # Best model by val/f1_macro
    last.pt              # Latest epoch
    epoch_XX_f1_0.XXXX.pt  # Top-k checkpoints
  training_history.json  # Per-epoch metrics
  training_metadata.json # Final summary
  best_model.pth         # Deployment-ready weights
```

## Hyperparameter Optimization

Optuna-based HPO searches across 11 hyperparameters with median pruning:

```bash
# Run 20-trial optimization
make hpo

# Custom trials and timeout
PYTHONPATH=. python3 scripts/run_hpo.py --config configs/train.yaml --n-trials 50 --timeout 7200
```

Optimized config saved to `configs/train_optimized.yaml`. See [Advanced MLOps](12-advanced-mlops.md) for full search space.

## Automated Retraining

The retraining trigger system checks 4 conditions:

```bash
make check-retrain  # Exit 0 = no action, Exit 2 = retrain needed
```

Triggers: model age > 30 days, 500+ new samples, critical drift, F1 drop > 0.05. See [Advanced MLOps](12-advanced-mlops.md).

## Key Files

| File | Purpose |
|---|---|
| `train.py` | Entry point, model factory, DDP setup (supports all models) |
| `scripts/train_hybrid_precision_v2.py` | **V2 precision-rescue training** with ASL, staged unfreeze, threshold optimization |
| `src/models/retinal_foundation_hybrid_v2.py` | HybridV2 model with bottleneck head, TTA, threshold management |
| `src/models/retinal_foundation_hybrid.py` | HybridV1 model (MoE, uncertainty heads) |
| `src/training/trainer.py` | DDPTrainer with AMP, gradient accum, early stopping |
| `src/training/losses.py` | FocalLoss, AsymmetricLoss |
| `src/training/metrics.py` | MetricTracker (F1, AUC, mAP, Hamming) |
| `src/evaluation/precision_threshold_optimizer.py` | Per-class precision-floor threshold optimization |
| `src/training/early_stopping.py` | AdvancedEarlyStopping with analysis |
| `src/training/hpo.py` | Optuna hyperparameter optimization |
| `src/training/retraining.py` | Automated retraining triggers |
| `configs/hybrid_precision_2026.yaml` | **V2 training config** (ASL, staged unfreeze, class filtering) |
| `configs/train.yaml` | Legacy training config |
