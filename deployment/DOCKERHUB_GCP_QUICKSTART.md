# 🚀 DockerHub + GCP Quick Start Guide

## Prerequisites

- ✅ Docker installed locally
- ✅ DockerHub account (free tier works)
- ✅ Google Cloud Platform account (free tier available)
- ✅ gcloud CLI installed

## 🎯 One-Command Deployment

```bash
cd deployment
./deploy.sh
```

This script will:
1. Build Docker image
2. Test locally
3. Push to DockerHub
4. Deploy to GCP Cloud Run
5. Return your API URL

## 📋 Manual Step-by-Step

### Step 1: DockerHub Setup

```bash
# Create account at https://hub.docker.com (free)
# Login
docker login

# Build image
docker build -f docker/Dockerfile.cpu -t YOUR_USERNAME/retinal-disease-api:latest .

# Test locally
docker run -p 8000:8000 YOUR_USERNAME/retinal-disease-api:latest
curl http://localhost:8000/health

# Push to DockerHub
docker push YOUR_USERNAME/retinal-disease-api:latest
```

### Step 2: GCP Setup

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install

# Login and create project
gcloud auth login
gcloud projects create my-retinal-api --name="Retinal Disease API"
gcloud config set project my-retinal-api

# Enable billing (required for Cloud Run)
# Visit: https://console.cloud.google.com/billing

# Enable Cloud Run API
gcloud services enable run.googleapis.com
```

### Step 3: Deploy to Cloud Run

```bash
# Deploy from DockerHub
gcloud run deploy retinal-disease-api \
  --image docker.io/YOUR_USERNAME/retinal-disease-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10

# Get your API URL
gcloud run services describe retinal-disease-api \
  --region us-central1 \
  --format 'value(status.url)'
```

### Step 4: Test Deployment

```bash
# Health check
curl https://YOUR-SERVICE-URL.run.app/health

# View docs
open https://YOUR-SERVICE-URL.run.app/docs

# Test prediction (need actual image)
curl -X POST "https://YOUR-SERVICE-URL.run.app/predict" \
  -F "file=@retinal_image.jpg" \
  -F "threshold=0.25"
```

## 🔄 GitHub Actions CI/CD

### Setup (One-Time)

1. **Fork/Clone repository to GitHub**

2. **Add GitHub Secrets** (Settings → Secrets → Actions):
   - `DOCKERHUB_USERNAME`: Your DockerHub username
   - `DOCKERHUB_TOKEN`: Create at https://hub.docker.com/settings/security
   - `GCP_PROJECT_ID`: Your GCP project ID
   - `GCP_SA_KEY`: Service account JSON (see GITHUB_SECRETS_SETUP.md)

3. **Push to trigger deployment**:
   ```bash
   git checkout -b develop
   git push origin develop  # Deploys to staging
   
   git checkout main
   git push origin main     # Deploys to production
   ```

### CI/CD Workflow

```
Pull Request → Run Tests
     ↓
Push to develop → Build → DockerHub → GCP Staging
     ↓
Push to main → Build → DockerHub → GCP Production
```

## 💰 Cost Estimates

### DockerHub (Free Tier)
- ✅ Unlimited public repositories
- ✅ 1 private repository
- ✅ Unlimited pulls
- 🆓 **$0/month**

### GCP Cloud Run (Free Tier)
- ✅ 2 million requests/month
- ✅ 360,000 GB-seconds memory
- ✅ 180,000 vCPU-seconds
- 🆓 **First 2M requests free**

**Estimated cost after free tier**: $5-20/month for moderate traffic

### Cost Optimization Tips

```bash
# Reduce memory for lower cost
gcloud run services update retinal-disease-api \
  --memory 2Gi \
  --region us-central1

# Scale to zero when idle
gcloud run services update retinal-disease-api \
  --min-instances 0 \
  --region us-central1

# Monitor costs
gcloud billing accounts describe YOUR_BILLING_ACCOUNT
```

## 📊 Monitoring

### View Logs

```bash
# Real-time logs
gcloud run services logs tail retinal-disease-api --region us-central1

# Last 50 logs
gcloud run services logs read retinal-disease-api --region us-central1 --limit 50
```

### View Metrics (GCP Console)

1. Go to https://console.cloud.google.com/run
2. Click on `retinal-disease-api`
3. View: Metrics, Logs, YAML, Revisions

### Performance Testing

```bash
# Run GPU/TPU tests
python tests/test_gpu_inference.py \
  --url https://YOUR-SERVICE-URL.run.app \
  --requests 50 \
  --concurrent 10
```

## 🔒 Security

### Enable Authentication (Optional)

```bash
# Require authentication
gcloud run services update retinal-disease-api \
  --no-allow-unauthenticated \
  --region us-central1

# Add authorized users
gcloud run services add-iam-policy-binding retinal-disease-api \
  --member="user:email@example.com" \
  --role="roles/run.invoker" \
  --region us-central1
```

### Use Secret Manager for API Keys

```bash
# Store secrets
echo -n "your-api-key" | gcloud secrets create api-key --data-file=-

# Mount in Cloud Run
gcloud run services update retinal-disease-api \
  --update-secrets=API_KEY=api-key:latest \
  --region us-central1
```

## 🐛 Troubleshooting

### "Image not found" on GCP

```bash
# Verify image exists on DockerHub
docker pull YOUR_USERNAME/retinal-disease-api:latest

# Use full image path
--image docker.io/YOUR_USERNAME/retinal-disease-api:latest
```

### "Permission denied" errors

```bash
# Re-authenticate
gcloud auth login

# Check current project
gcloud config get-value project

# Enable required APIs
gcloud services enable run.googleapis.com
```

### "Out of memory" errors

```bash
# Increase memory
gcloud run services update retinal-disease-api \
  --memory 8Gi \
  --region us-central1
```

### Local Docker issues

```bash
# Check Docker is running
docker ps

# Clean up
docker system prune -a

# Rebuild without cache
docker build --no-cache -f docker/Dockerfile.cpu -t YOUR_IMAGE .
```

## 📚 Resources

- **DockerHub**: https://hub.docker.com
- **GCP Cloud Run**: https://cloud.google.com/run/docs
- **GCP Free Tier**: https://cloud.google.com/free
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **GitHub Actions**: https://docs.github.com/actions

## 🆘 Support

### Common Questions

**Q: Do I need a credit card for GCP?**
A: Yes, but you get $300 free credits for 90 days.

**Q: Will I be charged?**
A: Not until you exceed free tier limits. Set up billing alerts!

**Q: Can I use a private DockerHub repo?**
A: Yes, add `--set-secrets=DOCKER_PASSWORD=...` to Cloud Run deploy.

**Q: How do I update my deployed API?**
A: Push new image to DockerHub, then re-run `gcloud run deploy`.

### Get Help

- GitHub Issues: [repository]/issues
- GCP Support: https://cloud.google.com/support
- Stack Overflow: Tag `google-cloud-run` + `docker`

---

**Happy Deploying! 🎉**
