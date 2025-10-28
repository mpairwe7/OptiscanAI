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
│   ├── mobile_deployment.py         # Mobile deployment utilities
│   ├── export_models.py             # Kaggle model export
│   └── api_server.py                # FastAPI REST API server
├── models/                 # Trained models & outputs
│   ├── checkpoints/        # Model checkpoints
│   ├── exports/            # Exported models (ONNX, TorchScript)
│   └── outputs/            # Training outputs & visualizations
├── deployment/             # Deployment configurations
│   ├── setup.sh            # Deployment setup script
│   ├── install_dependencies.sh
│   ├── local_test.sh       # Local Docker testing
│   ├── DEPLOYMENT_GUIDE.md # Complete deployment guide
│   ├── API_DOCUMENTATION.md # API reference
│   └── README.md           # Deployment overview
├── .github/workflows/      # CI/CD pipelines
│   ├── ml-pipeline.yml     # Automated testing
│   └── deploy-gcp.yml      # Production deployment (Docker + GCP)
├── Dockerfile              # Container definition
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

## 🧪 Testing

### Unit Tests
```bash
pytest tests/ -v
```

### API Testing
```bash
# Local testing
./deployment/local_test.sh

# Test production API
curl https://YOUR-API.run.app/health
```

### Code Quality
```bash
black src/
flake8 src/
```

## 📦 Requirements

- **notebookc18697ca98.ipynb**: Main training pipeline with all 4 models
- **EDA_Analysis_Clean.ipynb**: Exploratory data analysis
- **Model_Development.ipynb**: Model architecture development
- **Mathematical_Foundations.md**: Mathematical documentation
- **Pitch_Deck.md**: Project presentation

## 🚢 Deployment

### 🐳 Docker Deployment (NEW!)

**Quick Test Locally:**
```bash
# Automated local testing
./deployment/local_test.sh

# Manual Docker deployment
docker build -t retinal-disease-model .
docker run -p 8080:8080 retinal-disease-model
# API available at http://localhost:8080
```

**Production Deployment to Google Cloud:**
```bash
# See complete guide
cat deployment/DEPLOYMENT_GUIDE.md

# Quick steps:
1. Configure GitHub Secrets (DOCKERHUB_*, GCP_*)
2. Export models from Kaggle
3. Push to GitHub (auto-deploys via Actions)
4. Access API at https://YOUR-SERVICE.run.app
```

### 📚 Deployment Documentation
- **[DEPLOYMENT_GUIDE.md](deployment/DEPLOYMENT_GUIDE.md)** - Complete setup guide
- **[API_DOCUMENTATION.md](deployment/API_DOCUMENTATION.md)** - API reference & examples
- **[deployment/README.md](deployment/README.md)** - Infrastructure overview

### 🚀 CI/CD Pipeline

**GitHub Actions Workflow:**
1. **Trigger:** Push to `main` with model changes
2. **Build:** Docker image with latest model
3. **Push:** Image to Docker Hub
4. **Deploy:** Automatically to Google Cloud Run
5. **Test:** Health checks & validation

### 🔌 API Endpoints

```bash
# Health check
curl https://YOUR-API.run.app/health

# List all 45 diseases
curl https://YOUR-API.run.app/diseases

# Predict diseases from image
curl -X POST https://YOUR-API.run.app/predict \
  -F "file=@retinal_image.jpg"
```

Response:
```json
{
  "predictions": [
    {
      "disease": "Diabetic Retinopathy (DR)",
      "probability": 0.87,
      "confidence": "high"
    }
  ]
}
```

### Kaggle Model Export

**Add to your training notebook (Cell 46):**
```python
# After training completes
from export_models import export_all_trained_models
export_paths = export_all_trained_models(cv_results, selected_models)
# Download from /kaggle/working/exports/
```

### Kaggle Training Environment
```bash
# Upload notebook to Kaggle
# Configure 2x T4 GPU runtime
# Run Cell 46 for parallel training
```

## 🌐 MLOps Pipeline

### Complete Workflow: Kaggle → GitHub → Docker Hub → GCP

```
┌─────────────┐
│   Kaggle    │  1. Train models (2x T4 GPU)
│  Training   │  2. Export with export_models.py
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   GitHub    │  3. Push models to repository
│ Repository  │  4. Trigger GitHub Actions
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Docker Hub  │  5. Build & push Docker image
│   Registry  │  6. Tag with version
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Google    │  7. Deploy to Cloud Run
│  Cloud Run  │  8. Auto-scale REST API
└─────────────┘
```

### Infrastructure as Code
- **Dockerfile** - Container definition
- **deploy-gcp.yml** - GitHub Actions workflow
- **api_server.py** - FastAPI application
- **requirements.txt** - Python dependencies

All automatically deployed on `git push`!

## 🧪 Testing

## 📦 Requirements

### Core Dependencies
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 32GB RAM (recommended)
- 2x GPU with 16GB VRAM each

### Deployment Dependencies
- Docker
- Google Cloud SDK (for GCP deployment)
- FastAPI & Uvicorn (API server)

See `requirements.txt` for full dependencies.

## 📝 Notebooks

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
