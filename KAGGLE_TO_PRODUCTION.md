# 🚀 Kaggle to Production - Complete Workflow

## 🎯 The Correct Separation

```
┌──────────────────────┐
│   KAGGLE NOTEBOOK    │  ← Train models ONLY
│   (Cells 1-55)       │  ← NO deployment code
└──────────────────────┘
         ⬇️ Download models/exports/
┌──────────────────────┐
│  LOCAL REPOSITORY    │  ← All deployment files pre-exist
│  (This repo)         │  ← Just add models, then push
└──────────────────────┘
         ⬇️ git push
┌──────────────────────┐
│  GITHUB ACTIONS      │  ← Automatic CI/CD
│  (Cloud)             │  ← Build, push, deploy
└──────────────────────┘
```

## 📋 What Goes Where

### 🔬 Kaggle Notebook (Training Only)
**Purpose:** Train and optimize models

**Contains:**
- Cell 1-54: Data loading, EDA, model training
- Cell 55: Mobile optimization and export
- Cell 56: Instructions (markdown only)

**Produces:**
```
models/exports/
├── best_model.pth          # Optimized PyTorch model
├── best_model.onnx         # Cross-platform format
└── model_metadata.json     # Model info
```

**Does NOT contain:**
- ❌ Dockerfiles
- ❌ CI/CD workflows
- ❌ API server code
- ❌ Deployment scripts

### 💻 Local Repository (This Folder)
**Purpose:** Deployment infrastructure

**Already Contains:**
```
MLOPS_V1/
├── src/
│   └── api_server.py                     # FastAPI server
├── models/
│   └── exports/                          # <- Paste Kaggle models HERE
├── .github/
│   └── workflows/
│       └── complete-pipeline.yml         # CI/CD automation
├── Dockerfile                            # CPU container
├── Dockerfile.gpu                        # GPU container
├── requirements.txt                      # Dependencies
└── deployment/
    └── scripts/
        ├── build.sh                      # Local build
        ├── test.sh                       # Local test
        ├── push.sh                       # Push to DockerHub
        └── deploy-gcp.sh                 # Manual GCP deploy
```

**You Add:**
- Models from Kaggle → `models/exports/`

### ☁️ GitHub Actions (Automatic)
**Purpose:** CI/CD automation

**Triggered by:** `git push` when `models/exports/` changes

**Does:**
1. Builds Docker images (CPU + GPU)
2. Pushes to DockerHub
3. Deploys to GCP Cloud Run
4. Runs health checks

## 🔄 Complete Workflow

### Phase 1: Training (Kaggle)

```bash
# On Kaggle:
1. Upload notebook: notebooks/notebookc18697ca98.ipynb
2. Run cells 1-55
3. Download output: models/exports/ folder
```

### Phase 2: Local Setup (One-Time)

```bash
# On your local machine:

# 1. Clone repository (if not already)
git clone https://github.com/mpairwe7/MLOPS_V1.git
cd MLOPS_V1

# 2. Verify deployment files exist
ls -la src/api_server.py           # ✓ Should exist
ls -la Dockerfile                   # ✓ Should exist
ls -la .github/workflows/           # ✓ Should exist

# 3. Setup GitHub Secrets (one-time)
# Go to: GitHub repo → Settings → Secrets → Actions
# Add these 3 secrets:
# - DOCKERHUB_PASSWORD
# - GCP_PROJECT_ID
# - GCP_SA_KEY
```

### Phase 3: Deployment (After Each Training)

```bash
# After downloading models from Kaggle:

# 1. Copy models to repository
cp -r ~/Downloads/models/exports/* models/exports/

# 2. Verify files
ls -la models/exports/
# Should show:
# - best_model.pth
# - best_model.onnx
# - model_metadata.json

# 3. Commit and push
git add models/exports/
git commit -m "Update: Mobile-optimized model from Kaggle training"
git push origin main

# 4. Watch automatic deployment
# Go to: GitHub repo → Actions tab
# Or use: gh run watch
```

### Phase 4: Verification

```bash
# Get your API URL (after deployment completes)
SERVICE_URL=$(gcloud run services describe retinal-disease-api \
  --region asia-southeast1 \
  --format 'value(status.url)')

# Test the API
curl $SERVICE_URL/health
curl $SERVICE_URL/diseases

# View API documentation
echo "Docs: $SERVICE_URL/docs"
```

## 🎯 Key Points

### ✅ DO:
- Run training cells (1-55) on Kaggle
- Download `models/exports/` from Kaggle
- Copy models to local `models/exports/`
- Run `git push` to trigger deployment
- Let GitHub Actions handle everything else

### ❌ DON'T:
- Don't run deployment code on Kaggle
- Don't create Dockerfiles on Kaggle
- Don't manually build Docker images (unless testing)
- Don't manually deploy to GCP (unless needed)

## 📊 What Gets Deployed

```yaml
Container: landwind/retinal-disease-api:latest
  ├── Base: python:3.10-slim
  ├── Code: src/api_server.py (from repo)
  ├── Models: models/exports/* (from Kaggle)
  └── Port: 8080

Deployed to: GCP Cloud Run
  ├── Region: asia-southeast1 (Singapore)
  ├── Memory: 4Gi
  ├── CPU: 2
  └── Scaling: 0-10 instances
```

## 🔧 Optional: Local Testing

If you want to test before pushing:

```bash
# Build Docker image locally
./deployment/scripts/build.sh

# Test locally
./deployment/scripts/test.sh

# If good, push to trigger deployment
git push
```

## 🆘 Troubleshooting

### Problem: GitHub Actions fails
```bash
# Check GitHub Secrets are set:
# - DOCKERHUB_PASSWORD
# - GCP_PROJECT_ID  
# - GCP_SA_KEY

# View logs in GitHub Actions tab
```

### Problem: Models not found
```bash
# Verify models exist locally
ls -la models/exports/best_model.pth

# If missing, download again from Kaggle
```

### Problem: API not responding
```bash
# Check Cloud Run logs
gcloud run services logs read retinal-disease-api \
  --region asia-southeast1

# Check service status
gcloud run services describe retinal-disease-api \
  --region asia-southeast1
```

## 📁 File Checklist

### Before First Deployment:

- [ ] Kaggle notebook runs successfully (Cell 55 completes)
- [ ] Downloaded `models/exports/` from Kaggle
- [ ] Copied models to local `models/exports/`
- [ ] GitHub secrets configured (3 secrets)
- [ ] Repository cloned locally

### For Each New Training:

- [ ] Ran training on Kaggle
- [ ] Downloaded new `models/exports/`
- [ ] Copied to local `models/exports/`
- [ ] Ran `git add models/exports/`
- [ ] Ran `git commit -m "Update: model"`
- [ ] Ran `git push origin main`
- [ ] Verified GitHub Actions succeeded
- [ ] Tested deployed API

## 🎊 Success Criteria

Your deployment is successful when:

1. ✅ GitHub Actions workflow completes (green checkmark)
2. ✅ DockerHub shows new image: `landwind/retinal-disease-api:latest`
3. ✅ GCP Cloud Run service is "Active"
4. ✅ `curl SERVICE_URL/health` returns `{"status": "healthy"}`
5. ✅ API docs accessible at `SERVICE_URL/docs`

## 🚀 Summary

```bash
# The only commands you need after training:
cp -r ~/Downloads/models/exports/* models/exports/
git add models/exports/
git commit -m "Update: model"
git push

# Everything else is automatic! 🎉
```

---

**Remember:** Kaggle = Training | Local Repo = Infrastructure | GitHub Actions = Deployment
