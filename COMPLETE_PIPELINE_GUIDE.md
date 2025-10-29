# Complete MLOps Pipeline Setup Guide
## Kaggle Training → DockerHub → GCP Cloud Run

This guide covers the complete workflow from training 4 models on Kaggle to deploying the best model on GCP via CI/CD.

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Pipeline Overview](#pipeline-overview)
3. [Step 1: Kaggle Training](#step-1-kaggle-training)
4. [Step 2: Model Selection & Optimization](#step-2-model-selection--optimization)
5. [Step 3: GitHub Secrets Setup](#step-3-github-secrets-setup)
6. [Step 4: Push to GitHub](#step-4-push-to-github)
7. [Step 5: Automated Deployment](#step-5-automated-deployment)
8. [Testing & Monitoring](#testing--monitoring)

---

## Prerequisites

### Required Accounts
- ✅ **Kaggle Account** - For model training
- ✅ **DockerHub Account** - Username: `landwind`, Password: `alien123.com`
- ✅ **Google Cloud Platform** - Free tier available
- ✅ **GitHub Account** - For code repository and CI/CD

### Local Tools (Optional for Testing)
```bash
# Install Podman (alternative to Docker)
sudo apt-get update
sudo apt-get install podman

# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

---

## Pipeline Overview

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌─────────────┐
│   Kaggle    │ ───> │    GitHub    │ ───> │  DockerHub  │ ───> │  GCP Cloud  │
│  Training   │      │   (CI/CD)    │      │  (landwind) │      │     Run     │
└─────────────┘      └──────────────┘      └─────────────┘      └─────────────┘
     4 Models          Auto Build            CPU + GPU            Singapore
     ↓                     ↓                   Images              (Production)
  Best Model          GitHub Actions             ↓                      ↓
     ↓                     ↓                 Automated             Public API
  Optimized         Tests & Deploy             Push                Endpoint
```

---

## Step 1: Kaggle Training

### 1.1 Upload Notebook to Kaggle
1. Go to your Kaggle account
2. Upload `notebooks/notebookc18697ca98.ipynb`
3. Or create a new notebook and copy the cells

### 1.2 Run Training
```python
# This trains 4 models and selects the best one
# The notebook is already configured to:
# - Train EfficientNet-B3, B4, ResNet50, DenseNet121
# - Evaluate using F1 Score and AUC
# - Select best model
# - Optimize for mobile deployment
# - Export to PyTorch and ONNX formats
```

### 1.3 Download Results
After training completes on Kaggle:
- Download the `models/` folder containing:
  - `best_model.pth` - PyTorch checkpoint
  - `best_model.onnx` - ONNX format (optional)
  - `model_metrics.json` - Performance metrics

---

## Step 2: Model Selection & Optimization

The notebook automatically:
1. **Trains 4 models** with different architectures
2. **Evaluates** each on validation set
3. **Selects best** based on F1 Score
4. **Optimizes** for mobile:
   - Quantization (INT8)
   - Pruning (40% sparsity)
   - ONNX export for cross-platform
5. **Exports** model files ready for deployment

---

## Step 3: GitHub Secrets Setup

### 3.1 Create GitHub Repository Secrets
Go to your GitHub repository: `Settings → Secrets and variables → Actions → New repository secret`

#### Required Secrets:

**1. DOCKERHUB_PASSWORD**
```
Name: DOCKERHUB_PASSWORD
Value: alien123.com
```

**2. GCP_PROJECT_ID**
```bash
# Find your project ID
gcloud projects list

# Copy the PROJECT_ID
Name: GCP_PROJECT_ID
Value: your-project-id-12345
```

**3. GCP_SA_KEY** (Service Account Key)
```bash
# Create service account
export PROJECT_ID="your-project-id"

gcloud iam service-accounts create github-actions \
  --display-name "GitHub Actions CI/CD" \
  --project $PROJECT_ID

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Create key
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com

# Copy the entire contents of key.json
cat key.json

# Create secret in GitHub
Name: GCP_SA_KEY
Value: <paste entire JSON content>
```

---

## Step 4: Push to GitHub

### 4.1 Initialize Repository (First Time)
```bash
cd /home/darkhorse/Downloads/MLOPS_V1

# Initialize git if needed
git init
git add .
git commit -m "Initial commit: Complete MLOps pipeline"

# Add remote and push
git remote add origin https://github.com/mpairwe7/MLOPS_V1.git
git branch -M main
git push -u origin main
```

### 4.2 Update Model Files (After Kaggle Training)
```bash
# Copy trained models from Kaggle download
cp -r ~/Downloads/models/* models/

# Commit and push
git add models/
git commit -m "Update: Best model from Kaggle training"
git push
```

---

## Step 5: Automated Deployment

### 5.1 Trigger Workflow
Once you push to GitHub, the CI/CD pipeline automatically:

1. **Builds** both CPU and GPU Docker images
2. **Pushes** to DockerHub (`landwind/retinal-disease-api`)
3. **Deploys** to GCP Cloud Run (Singapore region)
4. **Tests** the deployed API

### 5.2 Monitor Progress
```bash
# Watch workflow in GitHub
Go to: Repository → Actions → Build and Deploy - Complete Pipeline

# Or use GitHub CLI
gh run list
gh run watch
```

### 5.3 Manual Trigger (Optional)
```bash
# Trigger workflow manually
gh workflow run complete-pipeline.yml
```

---

## Testing & Monitoring

### Test Deployed API
```bash
# Get service URL from GitHub Actions output
# Or find it manually:
gcloud run services describe retinal-disease-api \
  --region asia-southeast1 \
  --format 'value(status.url)'

# Test health endpoint
curl https://retinal-disease-api-xxx.run.app/health

# View API documentation
# Open in browser:
https://retinal-disease-api-xxx.run.app/docs
```

### View Logs
```bash
# Stream logs
gcloud run services logs read retinal-disease-api \
  --region asia-southeast1 \
  --limit 50

# Follow logs in real-time
gcloud run services logs tail retinal-disease-api \
  --region asia-southeast1
```

### Check DockerHub Images
```bash
# View on DockerHub
https://hub.docker.com/r/landwind/retinal-disease-api

# Pull locally for testing
podman pull landwind/retinal-disease-api:latest
podman pull landwind/retinal-disease-api:latest-gpu
```

---

## 🚀 Quick Start Commands

```bash
# 1. Train on Kaggle (use the notebook)

# 2. Download models from Kaggle

# 3. Update repository
cd /home/darkhorse/Downloads/MLOPS_V1
cp -r ~/Downloads/models/* models/
git add models/
git commit -m "Update: Kaggle trained model"
git push

# 4. Wait for CI/CD (check GitHub Actions)

# 5. Test deployed API
SERVICE_URL=$(gcloud run services describe retinal-disease-api \
  --region asia-southeast1 \
  --format 'value(status.url)')

curl $SERVICE_URL/health
echo "API Docs: $SERVICE_URL/docs"
```

---

## 📁 Repository Structure

```
MLOPS_V1/
├── .github/
│   └── workflows/
│       ├── complete-pipeline.yml    # Main CI/CD pipeline
│       └── deploy-gcp.yml           # GCP deployment
├── models/                          # Kaggle trained models
│   ├── best_model.pth              # From Kaggle
│   ├── best_model.onnx             # Mobile optimized
│   └── model_metrics.json          # Performance stats
├── src/
│   ├── api_server.py               # FastAPI server
│   └── export_models.py            # Model export utilities
├── deployment/
│   └── scripts/
│       ├── podman_build.sh         # Local build
│       ├── podman_push.sh          # Manual push
│       ├── podman_test.sh          # Local test
│       └── gcp_deploy.sh           # Manual GCP deploy
├── Dockerfile                       # CPU image
├── Dockerfile.gpu                   # GPU image
└── requirements.txt                 # Python dependencies
```

---

## Troubleshooting

### Issue: Workflow fails at DockerHub push
**Solution**: Verify `DOCKERHUB_PASSWORD` secret is set correctly in GitHub

### Issue: GCP deployment fails
**Solution**: 
```bash
# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Verify service account permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:github-actions@*"
```

### Issue: Model files too large for GitHub
**Solution**: Use Git LFS
```bash
git lfs install
git lfs track "models/*.pth"
git lfs track "models/*.onnx"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

---

## Next Steps

1. ✅ Complete Kaggle training
2. ✅ Set up GitHub secrets
3. ✅ Push trained model to repository
4. ✅ Monitor CI/CD pipeline
5. ✅ Test deployed API
6. 📊 Set up monitoring (Prometheus/Grafana)
7. 🔄 Set up model retraining pipeline
8. 📱 Create mobile app integration

---

## Support & Resources

- **GitHub Actions**: Check `.github/workflows/` for workflow definitions
- **DockerHub**: https://hub.docker.com/r/landwind/retinal-disease-api
- **GCP Console**: https://console.cloud.google.com/run
- **API Docs**: `<YOUR_SERVICE_URL>/docs`

---

**🎉 Your complete MLOps pipeline is ready!**

Train on Kaggle → Push to GitHub → Auto-deploy to production!
