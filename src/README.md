# Source Code Directory

Production-ready Python scripts for model training and deployment.

## 📁 Files

### Training Scripts
- **02_Model_Development.py**
  - Standalone training script
  - Can be run from command line
  - Supports all 4 model architectures
  - Configurable via command-line arguments

### Deployment Scripts
- **mobile_deployment.py**
  - Model export utilities (ONNX, TorchScript)
  - Mobile optimization
  - Inference utilities
  - Quantization helpers

## 🚀 Usage

### Training Models
```bash
python 02_Model_Development.py --epochs 30 --batch-size 32
```

### Exporting Models
```bash
python mobile_deployment.py --export --model GraphCLIP --output ../models/exports/
```

## 🔧 Development

Add new model architectures or utilities here. Keep code modular and well-documented.
