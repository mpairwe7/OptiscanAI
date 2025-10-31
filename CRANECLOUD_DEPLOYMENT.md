# Crane Cloud Deployment Guide

Complete guide for deploying the GPU-accelerated Retinal Disease Screening API on Crane Cloud.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [GitHub Setup](#github-setup)
3. [Building the Docker Image](#building-the-docker-image)
4. [Deploying to Crane Cloud](#deploying-to-crane-cloud)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts
- ✅ **GitHub Account** - For CI/CD pipelines
- ✅ **Docker Hub Account** - For image storage
- ✅ **Crane Cloud Account** - For deployment (https://cranecloud.io)

### Required Secrets
You'll need to configure these secrets in your GitHub repository:

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add:

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `DOCKERHUB_PASSWORD` | Docker Hub access token | Docker Hub → Account Settings → Security → New Access Token |

---

## GitHub Setup

### 1. Configure Repository Secrets

```bash
# In your GitHub repository settings, add the Docker Hub password secret
```

### 2. Enable GitHub Actions

The workflow file is already created at `.github/workflows/deploy-cranecloud.yml`. It will:
- ✅ Build Docker image with NVIDIA CUDA support
- ✅ Push to Docker Hub
- ✅ Run automated tests
- ✅ Generate deployment instructions

### 3. Trigger Build

**Option A: Push to main branch**
```bash
git add .
git commit -m "Deploy GPU version to Crane Cloud"
git push origin main
```

**Option B: Manual trigger**
1. Go to **Actions** tab in GitHub
2. Select **Build and Deploy to Crane Cloud** workflow
3. Click **Run workflow**

---

## Building the Docker Image

### Automated Build (Recommended)
GitHub Actions will automatically build when you push to `main` or manually trigger the workflow.

### Manual Build (Alternative)
```bash
# Build locally
docker build -t landwind/retinal-screening-gpu:latest .

# Test locally (requires NVIDIA Docker runtime)
docker run --gpus all -p 8080:8080 landwind/retinal-screening-gpu:latest

# Push to Docker Hub
docker push landwind/retinal-screening-gpu:latest
```

---

## Deploying to Crane Cloud

### Step 1: Access Crane Cloud Dashboard

1. Visit https://cranecloud.io
2. Log in to your account
3. Navigate to your project or create a new one

### Step 2: Create New App

1. Click **Create New App**
2. Fill in the application details:
   - **Name**: `retinal-disease-api` (or your preferred name)
   - **Description**: GPU-accelerated Retinal Disease Screening API

### Step 3: Configure Docker Deployment

Select **Docker Image** as deployment method and configure:

#### Image Configuration
```yaml
Image: landwind/retinal-screening-gpu:latest
# Or use a specific version from GitHub Actions
Image: landwind/retinal-screening-gpu:v2.0-cuda-abc1234
```

#### Port Configuration
```yaml
Container Port: 8080
Protocol: HTTP
```

#### Resource Configuration (Important for GPU)
```yaml
CPU: 2 cores (minimum)
Memory: 4 GB (minimum)
GPU: 1x NVIDIA GPU (if available)
```

⚠️ **Note**: Check with Crane Cloud support if GPU resources are available in your plan.

### Step 4: Environment Variables (Optional)

Add these if needed:
```bash
MODEL_PATH=/app/models/exports/best_model.pth
LOG_LEVEL=INFO
MAX_WORKERS=1
```

### Step 5: Deploy

1. Review your configuration
2. Click **Deploy**
3. Wait for deployment to complete (2-5 minutes)
4. Copy your app URL (e.g., `https://retinal-disease-api.cranecloud.io`)

---

## Configuration

### Docker Image Specifications

| Specification | Value |
|--------------|-------|
| Base Image | nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 |
| Python Version | 3.10 |
| CUDA Version | 11.8.0 |
| cuDNN Version | 8 |
| PyTorch Version | 2.0.1+cu118 |
| Port | 8080 |

### Required Resources

| Resource | Minimum | Recommended | GPU-Enabled |
|----------|---------|-------------|-------------|
| CPU | 2 cores | 4 cores | 4 cores |
| RAM | 4 GB | 8 GB | 8 GB |
| GPU | - | - | 1x NVIDIA T4/V100 |
| Storage | 10 GB | 20 GB | 20 GB |

---

## Testing

### 1. Health Check
```bash
# Replace with your Crane Cloud URL
curl https://your-app.cranecloud.io/health

# Expected response:
{
  "status": "healthy",
  "gpu_available": true,
  "cuda_version": "11.8"
}
```

### 2. List Available Diseases
```bash
curl https://your-app.cranecloud.io/diseases

# Expected response:
{
  "diseases": [
    "Disease Abbrev",
    "Diabetic Retinopathy",
    "Age Related Macular Degeneration",
    ...
  ],
  "total": 45
}
```

### 3. Test Prediction Endpoint
```bash
# Upload a retinal image
curl -X POST https://your-app.cranecloud.io/predict \
  -F "file=@retinal_image.jpg" \
  -F "return_visualizations=true"

# Expected response:
{
  "predictions": {
    "Disease Abbrev": 0.95,
    "Diabetic Retinopathy": 0.12,
    ...
  },
  "top_predictions": [...],
  "processing_time": 0.234
}
```

### 4. Interactive API Documentation
Visit: `https://your-app.cranecloud.io/docs`

---

## Monitoring

### View Application Logs

**In Crane Cloud Dashboard:**
1. Navigate to your app
2. Click **Logs** tab
3. View real-time logs

**Via API:**
```bash
# Check application metrics
curl https://your-app.cranecloud.io/metrics
```

### Check GPU Usage

If GPU is available:
```bash
# The health endpoint will show GPU status
curl https://your-app.cranecloud.io/health | jq '.gpu_available'
```

---

## Troubleshooting

### Issue 1: Container Fails to Start

**Symptoms:**
- Deployment shows "Failed" status
- Container crashes immediately

**Solutions:**
```bash
# Check logs in Crane Cloud dashboard
# Common issues:
# 1. Port mismatch (ensure using 8080)
# 2. Missing model files
# 3. Insufficient memory

# Verify image locally:
docker run -p 8080:8080 landwind/retinal-screening-gpu:latest
```

### Issue 2: GPU Not Available

**Symptoms:**
- API responds but shows `"gpu_available": false`
- Slower inference times

**Solutions:**
1. Check if your Crane Cloud plan includes GPU access
2. Contact Crane Cloud support to enable GPU
3. Verify GPU resources in app configuration
4. The API will work on CPU, just slower

### Issue 3: Out of Memory

**Symptoms:**
- Container crashes with OOM error
- 502/503 errors

**Solutions:**
```bash
# Increase memory allocation in Crane Cloud:
# Settings → Resources → Memory: 8 GB

# Or reduce batch size in model loading
# Edit src/api_server.py if needed
```

### Issue 4: Health Check Fails

**Symptoms:**
- `/health` endpoint returns 404 or 500
- Deployment shows unhealthy status

**Solutions:**
```bash
# 1. Wait 2-3 minutes for full startup
# 2. Check logs for errors
# 3. Verify model files are present in image
# 4. Test locally first:

docker run -p 8080:8080 landwind/retinal-screening-gpu:latest
curl http://localhost:8080/health
```

### Issue 5: Image Too Large

**Symptoms:**
- Build takes too long
- Deployment fails with storage error

**Solutions:**
```bash
# The CUDA image is large (~6-8 GB)
# Ensure Crane Cloud has sufficient storage quota

# Optimize if needed:
# 1. Use multi-stage build (already implemented)
# 2. Remove unnecessary model files
# 3. Clean up apt cache (already done)
```

---

## Updating the Deployment

### Method 1: Automated (Recommended)

1. Make changes to your code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update model or API"
   git push origin main
   ```
3. GitHub Actions will build new image
4. Update image version in Crane Cloud dashboard

### Method 2: Manual

```bash
# Build new version
docker build -t landwind/retinal-screening-gpu:v2.1 .
docker push landwind/retinal-screening-gpu:v2.1

# Update in Crane Cloud:
# 1. Go to app settings
# 2. Update image tag to v2.1
# 3. Redeploy
```

---

## Cost Optimization

### Tips for Crane Cloud

1. **Use CPU-only for development/testing**
   - Switch to CPU-based deployment for testing
   - Enable GPU only for production

2. **Auto-scaling**
   - Configure auto-scaling based on traffic
   - Minimum instances: 1
   - Maximum instances: 5

3. **Resource limits**
   - Set appropriate CPU/memory limits
   - Monitor actual usage
   - Adjust based on metrics

4. **Image optimization**
   - Use specific version tags (not `latest`)
   - Clean up old images
   - Minimize layer sizes

---

## Advanced Configuration

### Custom Domain

1. In Crane Cloud dashboard:
   - Settings → Domains
   - Add your custom domain
   - Update DNS records

2. Configure HTTPS:
   - Crane Cloud provides automatic SSL
   - No additional configuration needed

### Load Balancing

For high-traffic deployments:
```yaml
Min Instances: 2
Max Instances: 10
Auto-scaling: CPU > 70%
Health Check Path: /health
```

### Database Integration

If you add a database later:
```bash
# Add environment variables in Crane Cloud:
DATABASE_URL=postgresql://user:pass@host:5432/dbname
REDIS_URL=redis://host:6379/0
```

---

## Support

### Crane Cloud Support
- Documentation: https://docs.cranecloud.io
- Support: support@cranecloud.io
- Community: Crane Cloud Slack/Discord

### Project Issues
- GitHub Issues: https://github.com/your-repo/issues
- Contact: your-email@example.com

---

## Quick Reference

### Essential Commands
```bash
# Check health
curl https://your-app.cranecloud.io/health

# List diseases
curl https://your-app.cranecloud.io/diseases

# Make prediction
curl -X POST https://your-app.cranecloud.io/predict \
  -F "file=@image.jpg"

# View API docs
open https://your-app.cranecloud.io/docs
```

### Environment Variables
```bash
MODEL_PATH=/app/models/exports/best_model.pth
API_HOST=0.0.0.0
API_PORT=8080
LOG_LEVEL=INFO
MAX_WORKERS=1
CUDA_VISIBLE_DEVICES=0
```

---

## Next Steps

1. ✅ Complete GitHub Actions setup
2. ✅ Verify Docker image builds successfully
3. ✅ Deploy to Crane Cloud
4. ✅ Test all endpoints
5. ✅ Monitor performance
6. ✅ Set up custom domain (optional)
7. ✅ Configure auto-scaling (optional)

**Congratulations! Your GPU-accelerated API is now deployed on Crane Cloud! 🚀**
