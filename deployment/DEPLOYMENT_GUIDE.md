# Deployment Setup Guide

This guide walks you through setting up automated deployment from Kaggle → GitHub → Docker Hub → Google Cloud Platform.

## 📋 Prerequisites

- GitHub account with repository access
- Docker Hub account
- Google Cloud Platform account with billing enabled
- Kaggle account with trained models

---

## 🔐 Step 1: Configure GitHub Secrets

Navigate to your repository on GitHub: `Settings > Secrets and variables > Actions`

### Required Secrets:

#### 1. **DOCKERHUB_USERNAME**
- Your Docker Hub username
- Example: `mpairwe7`

#### 2. **DOCKERHUB_TOKEN**
- Create at: https://hub.docker.com/settings/security
- Click "New Access Token"
- Name: `github-actions`
- Access permissions: `Read, Write, Delete`
- Copy the token immediately (shown only once!)

#### 3. **GCP_PROJECT_ID**
- Your Google Cloud project ID
- Find at: https://console.cloud.google.com → Select your project
- Example: `retinal-disease-prod`

#### 4. **GCP_SA_KEY**
- Service account key JSON
- Follow these steps:

```bash
# 1. Create service account
gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions Deployment"

# 2. Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# 3. Create and download key
gcloud iam service-accounts keys create gcp-key.json \
    --iam-account=github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com

# 4. Copy the entire contents of gcp-key.json
cat gcp-key.json

# 5. Paste into GitHub Secret: GCP_SA_KEY
# 6. Delete local key file for security
rm gcp-key.json
```

---

## 🏗️ Step 2: Google Cloud Platform Setup

### Enable Required APIs:

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com

# Enable Cloud Build API (optional, for faster builds)
gcloud services enable cloudbuild.googleapis.com
```

### Configure Cloud Run Region:

Choose a region close to your users:
- `us-central1` (Iowa, USA)
- `us-east1` (South Carolina, USA)
- `europe-west1` (Belgium)
- `asia-northeast1` (Tokyo, Japan)

Update in `.github/workflows/deploy-gcp.yml`:
```yaml
env:
  REGION: us-central1  # Change to your preferred region
```

---

## 🐳 Step 3: Docker Hub Setup

1. **Create Repository:**
   - Go to: https://hub.docker.com/repositories
   - Click "Create Repository"
   - Name: `retinal-disease-model`
   - Visibility: Public (or Private if you have a paid plan)

2. **Verify Repository Name:**
   - Full name should be: `YOUR_USERNAME/retinal-disease-model`
   - Example: `mpairwe7/retinal-disease-model`

---

## 📦 Step 4: Export Models from Kaggle

### Option A: In Kaggle Notebook (Recommended)

Add this code to the end of your training notebook (Cell 46):

```python
# Import export functions
import sys
sys.path.append('/kaggle/input/your-code-dataset/src')
from export_models import export_all_trained_models

# Export all trained models
if cv_results:
    export_paths = export_all_trained_models(cv_results, selected_models)
    print("\n🚀 Models ready for deployment!")
    print("Download from: /kaggle/working/exports/")
```

### Option B: Manual Download

1. Run your training notebook on Kaggle
2. After training completes, navigate to Output section
3. Download the `exports/` folder
4. Extract to your local repository: `models/exports/`

### Required Files:
```
models/exports/
├── best_model.pth          # Main model checkpoint
├── deployment_manifest.json # Deployment configuration
└── *_metadata.json         # Model metadata
```

---

## 🚀 Step 5: Deploy to Production

### Automatic Deployment (GitHub Actions):

1. **Commit and Push:**
   ```bash
   cd "Multi Rentinal Disease Model"
   
   # Add exported models
   git add models/exports/
   git commit -m "Add trained models for deployment"
   git push origin main
   ```

2. **Monitor Deployment:**
   - Go to: https://github.com/YOUR_USERNAME/MLOPS_V1/actions
   - Click on the latest workflow run
   - Watch the deployment progress

3. **Get Service URL:**
   - After successful deployment, check the workflow summary
   - Service URL will be displayed: `https://retinal-disease-api-xxxxx-uc.a.run.app`

### Manual Deployment (Alternative):

```bash
# Build Docker image locally
docker build -t mpairwe7/retinal-disease-model:latest .

# Test locally
docker run -p 8080:8080 mpairwe7/retinal-disease-model:latest

# Push to Docker Hub
docker push mpairwe7/retinal-disease-model:latest

# Deploy to Cloud Run
gcloud run deploy retinal-disease-api \
  --image mpairwe7/retinal-disease-model:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --port 8080
```

---

## 🧪 Step 6: Test Your Deployment

### Health Check:
```bash
curl https://YOUR-SERVICE-URL.run.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### List Diseases:
```bash
curl https://YOUR-SERVICE-URL.run.app/diseases
```

### Make a Prediction:
```bash
curl -X POST https://YOUR-SERVICE-URL.run.app/predict \
  -F "file=@path/to/retinal_image.jpg"
```

Expected response:
```json
{
  "predictions": [
    {
      "disease": "Diabetic Retinopathy",
      "probability": 0.87,
      "confidence": "high"
    }
  ]
}
```

---

## 📊 Step 7: Monitoring and Logs

### View Cloud Run Logs:
```bash
gcloud run services logs read retinal-disease-api \
  --region us-central1 \
  --limit 50
```

### View Metrics:
1. Go to: https://console.cloud.google.com/run
2. Click on `retinal-disease-api`
3. Navigate to "Metrics" tab
4. Monitor: Requests, Latency, Memory, CPU

### Set up Alerts (Optional):
```bash
# Alert on high error rate
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="High Error Rate" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s
```

---

## 🔄 Continuous Deployment Workflow

### Typical Workflow:

1. **Train on Kaggle:**
   - Run training notebook
   - Export models to `/kaggle/working/exports/`

2. **Download Models:**
   - Download exports from Kaggle Output
   - Place in `models/exports/` locally

3. **Commit Changes:**
   ```bash
   git add models/exports/
   git commit -m "Update model: improved F1 to 0.95"
   git push origin main
   ```

4. **Automatic Deployment:**
   - GitHub Actions triggers on push
   - Builds Docker image
   - Pushes to Docker Hub
   - Deploys to Cloud Run
   - Runs health checks

5. **Verify:**
   - Check workflow status on GitHub
   - Test API endpoints
   - Monitor logs and metrics

---

## 🐛 Troubleshooting

### Issue: "Permission Denied" on GCP

**Solution:**
```bash
# Verify service account permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:github-actions@*"

# Re-add missing roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"
```

### Issue: "Docker Image Too Large"

**Solution:**
```bash
# Use multi-stage build (already implemented in Dockerfile)
# Reduce model size by quantization (optional)
```

### Issue: "Model Not Found"

**Solution:**
```bash
# Ensure model exists in correct path
ls -lh models/exports/best_model.pth

# Update Dockerfile if path changed
```

### Issue: "API Timeout"

**Solution:**
```yaml
# Increase timeout in deploy-gcp.yml
--timeout 600  # 10 minutes for large model loading
```

---

## 📈 Cost Optimization

### Cloud Run Pricing (Approximate):

- **Free Tier:** 
  - 2 million requests/month
  - 360,000 GB-seconds/month
  - 180,000 vCPU-seconds/month

- **Estimated Cost (after free tier):**
  - Memory (4GB): $0.0000025/GB-second
  - CPU (2 vCPU): $0.000024/vCPU-second
  - Requests: $0.40/million

### Tips to Reduce Costs:

1. **Set min-instances to 0:**
   ```yaml
   --min-instances 0  # Scale to zero when not in use
   ```

2. **Use smaller instance:**
   ```yaml
   --memory 2Gi  # If model fits
   --cpu 1       # For lower traffic
   ```

3. **Enable request timeout:**
   ```yaml
   --timeout 300  # Kill long-running requests
   ```

---

## 🎯 Next Steps

- [ ] Set up GitHub Secrets
- [ ] Enable GCP APIs
- [ ] Create Docker Hub repository
- [ ] Export models from Kaggle
- [ ] Test local Docker build
- [ ] Push to GitHub and trigger deployment
- [ ] Verify API endpoints
- [ ] Set up monitoring and alerts
- [ ] Document API for users
- [ ] (Optional) Add authentication
- [ ] (Optional) Set up custom domain

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 🆘 Support

For issues or questions:
1. Check GitHub Actions logs: `Actions` tab in repository
2. Check Cloud Run logs: `gcloud run services logs read`
3. Review this guide's troubleshooting section
4. Open an issue on GitHub repository

---

**🎉 Congratulations! Your MLOps pipeline is ready for production!**
