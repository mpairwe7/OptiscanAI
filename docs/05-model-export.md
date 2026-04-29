# Model Export to Backend

## Training to Production Flow

```
V2 Precision Rescue Pipeline               Production Backend
────────────────────────────               ──────────────────
train_hybrid_precision_v2.py               backend/app/core/model_service.py
  └─> outputs/checkpoints/v2/               └─> loads checkpoint at startup
      ├─> best.pth                              └─> serves via /api/v1/predict
      ├─> final_with_thresholds.pth         (model + per-class thresholds)
      ├─> thresholds_optimized.json         (precision-floor thresholds)
      └─> outputs/export_v2/
          ├─> model.onnx                   (ONNX opset 18, float32 safe)
          ├─> model.pt                     (TorchScript, dtype-verified)
          └─> model_int8.pth               (INT8 quantized, <65MB)
```

## V2 Export (Precision-Safe)

```bash
# Export with dtype safety (fixes float/half mismatch from v1)
python3 scripts/export_production_v2.py \
    --checkpoint outputs/checkpoints/v2/final_with_thresholds.pth \
    --formats onnx torchscript \
    --quantize

# Key: ensure_float32() is called before all exports to prevent dtype mismatches
```

The v2 export script (`scripts/export_production_v2.py`) fixes the quantization failure from v1 by:
1. Casting all params/buffers to float32 before ONNX export
2. Running numerical verification (max diff < 1e-4) after each export
3. INT8 quantization via `torch.quantization.quantize_dynamic` with proper pre-casting

## Legacy Export

```bash
make export
make mlops-pipeline
dvc repro export
```

See `scripts/export_model.py` for the legacy export CLI and [Advanced MLOps](12-advanced-mlops.md) for DVC pipeline details.

## Checkpoint Format

The training pipeline saves checkpoints as:

```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,  # Model weights
    "optimizer_state_dict": OrderedDict,
    "best_f1": float,
    "metrics": dict,                  # All eval metrics
    "config": dict,                   # Full training config
}
```

## Export Best Model

After training completes, the best model is automatically saved to `outputs/best_model.pth` with a deployment-friendly format:

```python
{
    "model_state_dict": OrderedDict,  # Weights only
    "num_classes": 45,
    "best_f1": float,
    "config": {"name": "vignn", "hidden_dim": 384, ...},
}
```

## Loading in Backend

The `ModelService` singleton (`backend/app/core/model_service.py`) handles the full lifecycle:

```python
class ModelService:
    def load(self):
        # 1. Auto-detect device (CUDA/CPU)
        # 2. Create ClinicalKnowledgeGraph (45 diseases, 144 relationships)
        # 3. Load model via create_vignn_model(checkpoint_path=...)
        # 4. Move to device, set eval mode
    
    def predict(self, image: PIL.Image) -> dict:
        # 1. Preprocess: resize 224x224, normalize (ImageNet stats)
        # 2. Forward pass with torch.no_grad()
        # 3. Sigmoid activation -> probabilities
        # 4. Apply clinical reasoning (knowledge graph refinement)
        # 5. Compute referral priority
        # 6. Return structured predictions
```

## Manual Export Steps

```bash
# 1. Train the model
make train

# 2. Verify checkpoint exists
ls outputs/checkpoints/best.pt

# 3. Copy to model serving location
cp outputs/best_model.pth src/models/model_vignn_rank1.pth

# 4. Start backend (it will auto-load from MODEL_PATH)
make backend
```

## Configuration

Set the model path via environment or `backend/app/core/config.py`:

```python
# config.py (pydantic-settings)
model_path: str = "models/model_vignn_rank1.pth"  # Default
```

Override via environment:
```bash
MODEL_PATH=outputs/best_model.pth make backend
```

## Docker Deployment

```bash
# Build GPU image
docker compose build api

# Run with custom model
docker compose up -d

# The Dockerfile copies src/models/ into the container
# Volume mount for live updates:
#   volumes:
#     - ./src/models:/app/src/models
```

## Switching Models

To deploy a different architecture:

```bash
# 1. Change model in config
sed -i 's/name: "vignn"/name: "graphclip"/' configs/train.yaml

# 2. Retrain
make train

# 3. Export
cp outputs/best_model.pth src/models/model_graphclip.pth

# 4. Update backend config
export MODEL_PATH=src/models/model_graphclip.pth
make backend
```
