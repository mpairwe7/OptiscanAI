# 🚀 Quick Start - After Kaggle Training

## One-Time Setup (First Time Only)

```bash
# 1. Setup GitHub Secrets
# Go to: GitHub repo → Settings → Secrets → Actions
# Add 3 secrets:
# - DOCKERHUB_PASSWORD: alien123.com
# - GCP_PROJECT_ID: your-gcp-project-id
# - GCP_SA_KEY: <service account JSON>
```

## Every Training Run

```bash
# === Step 1: Download from Kaggle ===
# After running cells 1-55 on Kaggle
# Download: models/exports/ folder

# === Step 2: Copy & Push ===
cd /home/darkhorse/Downloads/MLOPS_V1
cp -r ~/Downloads/models/exports/* models/exports/
git add models/exports/
git commit -m "Update: model from Kaggle $(date +%Y%m%d)"
git push origin main

# === Step 3: Wait (Automatic) ===
# GitHub Actions builds & deploys (3-5 minutes)
# Watch: https://github.com/mpairwe7/MLOPS_V1/actions

# === Step 4: Test ===
SERVICE_URL=$(gcloud run services describe retinal-disease-api \
  --region asia-southeast1 \
  --format 'value(status.url)')

curl $SERVICE_URL/health
echo "🎉 API Live: $SERVICE_URL/docs"
```

## That's It! 3 Commands After Training:

```bash
cp -r ~/Downloads/models/exports/* models/exports/
git add models/exports/ && git commit -m "Update: model" && git push
# Wait for GitHub Actions → API is live! 🚀
```

---

**No Docker builds. No manual deployments. Just push!**
