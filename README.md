# Multi Retinal Disease Classification Model

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Deep learning models for multi-label retinal disease classification using advanced graph neural networks and vision transformers.

## 🏗️ Project Structure

```
Multi-Retinal-Disease-Model/
├── notebooks/              # Kaggle notebooks & experiments
│   ├── notebookc18697ca98.ipynb     # Main training notebook
│   ├── EDA_Analysis_Clean.ipynb     # Data exploration
│   ├── Model_Development.ipynb      # Model development
│   ├── 03_Mathematical_Foundations.md
│   └── 04_Pitch_Deck.md
├── src/                    # Production-ready code
│   ├── 02_Model_Development.py      # Model training script
│   └── mobile_deployment.py         # Mobile deployment utilities
├── models/                 # Trained models & outputs
│   ├── checkpoints/        # Model checkpoints
│   ├── exports/            # Exported models (ONNX, TorchScript)
│   └── outputs/            # Training outputs & visualizations
├── deployment/             # Deployment configurations
│   ├── setup.sh            # Deployment setup script
│   └── install_dependencies.sh
├── .github/workflows/      # CI/CD pipelines
│   └── ml-pipeline.yml     # Automated testing & deployment
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mpairwe7/MLOPS_V1.git
cd MLOPS_V1
```

### 2. Setup Environment
```bash
# Run automated setup
./deployment/setup.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Training
```bash
# Activate environment
source .venv/bin/activate

# Run training script
python src/02_Model_Development.py

# Or use Jupyter notebooks
jupyter notebook notebooks/
```

## 🧠 Models

This project implements 4 state-of-the-art architectures:

| Model | Parameters | Features |
|-------|-----------|----------|
| **GraphCLIP** | ~45M | CLIP + Graph Attention |
| **VisualLanguageGNN** | ~48M | Visual-Language Fusion |
| **SceneGraphTransformer** | ~52M | Spatial Scene Understanding |
| **ViGNN** | ~50M | Visual Graph Neural Network |

### Key Features:
- ✅ Multi-label classification (45 retinal diseases)
- ✅ Cross-validation training (5-fold)
- ✅ Multi-GPU support (parallel training)
- ✅ Memory-optimized for Kaggle (2x T4 GPUs)
- ✅ Mobile deployment ready

## 📊 Training Configuration

Current setup for **Kaggle 2x T4 GPUs**:

```python
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_CLASSES = 45
K_FOLDS = 5

# GPU Configuration
models_per_gpu = 1  # Memory-safe mode
max_workers = 2     # Parallel training on 2 GPUs
```

## 📈 Performance

| Model | F1 Score | AUC-ROC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| GraphCLIP | TBD | TBD | TBD | TBD |
| VisualLanguageGNN | TBD | TBD | TBD | TBD |
| SceneGraphTransformer | TBD | TBD | TBD | TBD |
| ViGNN | TBD | TBD | TBD | TBD |

## 🔧 Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black src/
flake8 src/
```

### CI/CD Pipeline
- Automated testing on push/PR
- Model validation
- Notebook compatibility checks
- Deployment to production (on main branch)

## 📝 Notebooks

- **notebookc18697ca98.ipynb**: Main training pipeline with all 4 models
- **EDA_Analysis_Clean.ipynb**: Exploratory data analysis
- **Model_Development.ipynb**: Model architecture development
- **Mathematical_Foundations.md**: Mathematical documentation
- **Pitch_Deck.md**: Project presentation

## 🚢 Deployment

### Local Deployment
```bash
./deployment/setup.sh
```

### Kaggle Deployment
```bash
# Upload notebook to Kaggle
# Configure 2x T4 GPU runtime
# Run Cell 46 for parallel training
```

### Production Deployment
```bash
# Export model
python src/mobile_deployment.py --export --model-name GraphCLIP

# Deploy to cloud (configure in .github/workflows/ml-pipeline.yml)
```

## 📦 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 32GB RAM (recommended)
- 2x GPU with 16GB VRAM each

See `requirements.txt` for full dependencies.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **mpairwe7** - [GitHub Profile](https://github.com/mpairwe7)

## 🙏 Acknowledgments

- Kaggle for GPU resources
- PyTorch and timm libraries
- Research papers on graph neural networks and vision transformers

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [MLOPS_V1/issues](https://github.com/mpairwe7/MLOPS_V1/issues)
- Repository: [MLOPS_V1](https://github.com/mpairwe7/MLOPS_V1)

---

**Note**: This is an active research project. Models are continuously being improved.
