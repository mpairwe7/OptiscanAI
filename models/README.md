# Models Directory

This directory stores trained models, checkpoints, and outputs.

## 📁 Structure

```
models/
├── checkpoints/     # Model checkpoints (.pth, .pt files)
├── exports/         # Exported models (ONNX, TorchScript, TFLite)
└── outputs/         # Training outputs and visualizations
```

## 💾 Checkpoints

Model checkpoints are saved during training:
- Best model per fold
- Final model after training
- Named format: `{model_name}_fold{n}_epoch{e}.pth`

## 📤 Exports

Exported models for deployment:
- **ONNX**: Cross-platform inference
- **TorchScript**: PyTorch mobile
- **TFLite**: TensorFlow Lite for mobile

## 📊 Outputs

Training visualizations and results:
- Training/validation curves
- Confusion matrices
- ROC curves
- Model architecture diagrams

## ⚠️ Note

Large model files are excluded from git (see .gitignore).
Download pre-trained models separately if needed.
