# CraneCloud Deployment Guide - Retinal Screening API

## ✅ Pre-Deployment Checklist

- [x] Docker image built: `landwind/retinal-screening-api:latest`
- [x] Image pushed to Docker Hub (5.28 GB)
- [x] Image tags: `latest`, `v1.0-gpu`
- [x] Last pushed: October 31, 2025

## 🚀 CraneCloud Deployment Steps

### Option 1: Web Dashboard Deployment (Recommended for First Time)

#### Step 1: Login to CraneCloud
1. Navigate to: https://cranecloud.io
2. Click **"Sign In"** or **"Sign Up"** if you don't have an account
3. Login with your credentials

#### Step 2: Create New Application
1. Click **"Create Application"** or **"+ New App"** button
2. Fill in the application details:
   ```
   Application Name: retinal-screening-api
   Description: Retinal disease screening API with multi-model support
   ```

#### Step 3: Configure Container Settings
Configure your application with these settings:

**Container Image:**
```
landwind/retinal-screening-api:latest
```

**Port Configuration:**
```
Container Port: 8080
Protocol: HTTP
```

**Environment Variables:** (Optional - if needed)
```
PORT=8080
PYTHONUNBUFFERED=1
```

**Resource Allocation:**
```
CPU: 1-2 cores (minimum 1 core)
Memory: 2048 MB (2GB minimum, 4GB recommended)
Storage: 10 GB
```

**GPU Configuration:** (If GPU support available on CraneCloud)
```
GPU: 1x NVIDIA T4 or similar (optional but recommended)
CUDA Version: 11.8 (image already has this)
```

#### Step 4: Health Check Configuration
```
Health Check Path: /health
Health Check Port: 8080
Initial Delay: 30 seconds
Period: 10 seconds
Timeout: 5 seconds
```

#### Step 5: Deploy
1. Review all settings
2. Click **"Deploy"** or **"Create Application"**
3. Wait for deployment to complete (2-5 minutes)
4. Check deployment status in the dashboard

#### Step 6: Verify Deployment
Once deployed, CraneCloud will provide you with a URL. Test the endpoints:

**Health Check:**
```bash
curl https://your-app-url.cranecloud.io/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": false,
  "device": "cuda:0"
}
```

**Get Supported Diseases:**
```bash
curl https://your-app-url.cranecloud.io/diseases
```

Expected: JSON array with 45 disease names

**API Documentation:**
```
https://your-app-url.cranecloud.io/docs
```

---

### Option 2: CraneCloud CLI Deployment (If Available)

If CraneCloud provides a CLI tool, you can use these commands:

#### Step 1: Install CraneCloud CLI (if not installed)
```bash
# Check if CraneCloud CLI is available
# Installation instructions depend on CraneCloud's CLI tool
```

#### Step 2: Login
```bash
cranecloud login
# or
cranecloud login --token YOUR_API_TOKEN
```

#### Step 3: Create Application
```bash
cranecloud apps create \
  --name retinal-screening-api \
  --image landwind/retinal-screening-api:latest \
  --port 8080 \
  --memory 2048 \
  --cpu 1 \
  --env PORT=8080
```

#### Step 4: Check Status
```bash
cranecloud apps status retinal-screening-api
```

---

## 🔧 Troubleshooting

### Issue 1: Container Fails to Start
**Symptoms:** Application status shows "Failed" or "CrashLoopBackoff"

**Solution:**
1. Check CraneCloud logs: Look for error messages in the application logs
2. Verify the image is accessible: `docker pull landwind/retinal-screening-api:latest`
3. Check port configuration: Ensure port 8080 is correctly configured
4. Verify resource limits: Increase memory to 4GB if OOM errors appear

### Issue 2: Health Check Failing
**Symptoms:** Application shows "Unhealthy" status

**Solution:**
1. Verify health endpoint path is `/health`
2. Increase initial delay to 60 seconds (model loading takes time)
3. Check application logs for startup errors

### Issue 3: Image Pull Errors
**Symptoms:** "ImagePullBackoff" or "Failed to pull image"

**Solution:**
1. Verify Docker Hub repository is public (not private)
2. Check image name spelling: `landwind/retinal-screening-api:latest`
3. Try using the full image SHA: `landwind/retinal-screening-api@sha256:f2c573f83eab10f7db7d8c6e8c5f3469354e57fc4b40a1873dce5450182a7bc1`

### Issue 4: GPU Not Available
**Symptoms:** Model runs but shows `device: "cpu"` instead of `"cuda:0"`

**Solution:**
1. Check if CraneCloud supports GPU instances in your region
2. Request GPU-enabled node explicitly in resource settings
3. Verify CUDA compatibility (image has CUDA 11.8)
4. Contact CraneCloud support to enable GPU access

### Issue 5: Out of Memory (OOM)
**Symptoms:** Container restarts frequently, logs show memory errors

**Solution:**
1. Increase memory allocation to 4GB or 8GB
2. The model files are large (~5GB image + model loading)
3. Consider using a larger instance type

---

## 📊 Resource Requirements

### Minimum Configuration (CPU-only)
```
CPU: 1 core
Memory: 2048 MB (2GB)
Storage: 10 GB
GPU: None
Cost Estimate: ~$10-15/month (varies by provider)
```

### Recommended Configuration (CPU-only)
```
CPU: 2 cores
Memory: 4096 MB (4GB)
Storage: 10 GB
GPU: None
Cost Estimate: ~$20-30/month
```

### Optimal Configuration (GPU-enabled)
```
CPU: 2-4 cores
Memory: 8192 MB (8GB)
Storage: 20 GB
GPU: 1x NVIDIA T4 or similar
Cost Estimate: ~$50-100/month
```

---

## 🔐 Security Best Practices

1. **Enable HTTPS:** CraneCloud should provide SSL/TLS by default
2. **Add Authentication:** Consider adding API key authentication for production
3. **Rate Limiting:** Implement rate limiting to prevent abuse
4. **Monitoring:** Set up alerts for high resource usage or errors
5. **Private Networks:** If CraneCloud supports VPC, use private networking

---

## 📈 Monitoring & Scaling

### Health Monitoring
- Endpoint: `GET /health`
- Monitor every 30 seconds
- Alert if 3+ consecutive failures

### Performance Metrics to Track
- Request latency (aim for <2 seconds per prediction)
- Memory usage (should stay under 80% of allocated)
- CPU usage
- Error rate (should be <1%)

### Scaling Triggers
- Scale up if:
  - CPU usage > 80% for 5+ minutes
  - Memory usage > 85%
  - Request queue length > 10
  - Response time > 5 seconds

---

## 🌐 API Endpoints Summary

Once deployed, your API will have these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/diseases` | GET | List all supported diseases (45) |
| `/predict` | POST | Single image prediction |
| `/batch_predict` | POST | Batch image predictions |
| `/models` | GET | List available models |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/redoc` | GET | Alternative API documentation (ReDoc) |

---

## 📞 Support & Resources

**CraneCloud Documentation:** https://docs.cranecloud.io
**CraneCloud Support:** support@cranecloud.io

**Your Docker Image:**
- Docker Hub: https://hub.docker.com/r/landwind/retinal-screening-api
- Image Size: 5.28 GB
- Base: nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
- Python: 3.10+
- Framework: FastAPI + PyTorch

**This Project:**
- GitHub: https://github.com/mpairwe7/MLOPS_V1
- Local Path: /home/darkhorse/Downloads/MLOPS_V1

---

## ✅ Post-Deployment Verification Checklist

After deployment, verify these items:

- [ ] Application status shows "Running" or "Healthy"
- [ ] Health endpoint returns 200 OK: `/health`
- [ ] Diseases endpoint returns 45 diseases: `/diseases`
- [ ] API documentation is accessible: `/docs`
- [ ] Response time is acceptable (<5 seconds)
- [ ] Logs show no critical errors
- [ ] SSL/HTTPS is working (should be https://)
- [ ] Application restarts successfully after crash test
- [ ] Resource usage is within limits (CPU <80%, Memory <80%)

---

## 🎯 Next Steps After Deployment

1. **Test the API thoroughly**
   - Upload test retinal images
   - Verify predictions are reasonable
   - Test all 4 models (GraphCLIP, VisualLanguageGNN, etc.)

2. **Set up monitoring**
   - Configure uptime monitoring (e.g., UptimeRobot)
   - Set up error tracking (e.g., Sentry)
   - Enable resource alerts

3. **Document the deployment**
   - Save the CraneCloud app URL
   - Document any configuration changes
   - Share API endpoint with team

4. **Plan for production**
   - Add authentication/authorization
   - Implement rate limiting
   - Set up CI/CD for auto-deployment
   - Configure backup and disaster recovery

---

## 🚨 Emergency Rollback

If the deployment fails or has critical issues:

1. **Quick Rollback:**
   ```bash
   # If using previous image version
   # Update CraneCloud to use: landwind/retinal-screening-api:v1.0
   ```

2. **Local Testing:**
   ```bash
   # Test locally before re-deploying
   cd /home/darkhorse/Downloads/MLOPS_V1
   podman run -p 8080:8080 landwind/retinal-screening-api:latest
   curl http://localhost:8080/health
   ```

3. **Contact Support:**
   - CraneCloud support: support@cranecloud.io
   - Check CraneCloud status page for platform issues

---

**Document Version:** 1.0  
**Last Updated:** October 31, 2025  
**Maintained By:** MLOPS_V1 Team
