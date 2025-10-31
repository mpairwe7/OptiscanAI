## From Terminal → Docker Hub → CraneCloud

---

## ✅ COMPLETED STEPS

### Step 1: Prerequisites ✓
- ✅ Podman installed (v5.4.0)
- ✅ Model file exists (`models/best_model_mobile.pt` - 26MB)
- ✅ Dockerfile configured for GPU support (CUDA 11.8)
- ✅ Git repository initialized

### Step 2: Docker Hub Authentication ✓
```bash
echo "alien123.com" | podman login docker.io --username landwind --password-stdin
```
**Status:** Login Succeeded! ✓

### Step 3: Building GPU Docker Image ⏳ IN PROGRESS
```bash
podman build -t landwind/retinal-screening-api:latest \
  -t landwind/retinal-screening-api:v1.0-gpu \
  -f Dockerfile .
```
**Status:** Building... (Installing Python dependencies and CUDA libraries)

---

## 🔄 NEXT STEPS (After build completes)

### Step 4: Test Image Locally
```bash
# Run container
podman run -d --name test-api -p 8080:8080 \
  landwind/retinal-screening-api:latest

# Wait for startup
sleep 20

# Test health endpoint
curl http://localhost:8080/health

# Expected response:
# {"status":"healthy","model":"best_model_mobile.pt","diseases":45}

# Test diseases endpoint
curl http://localhost:8080/diseases

# Stop test container
podman stop test-api
podman rm test-api
```

### Step 5: Push Image to Docker Hub
```bash
# Push with 'latest' tag
podman push landwind/retinal-screening-api:latest

# Push with version tag
podman push landwind/retinal-screening-api:v1.0-gpu
```

### Step 6: Configure GitHub Secrets
```bash
# Install GitHub CLI if not installed
# For Fedora/RHEL: sudo dnf install gh
# For Ubuntu: sudo apt install gh

# Login to GitHub
gh auth login

# Set Docker Hub password as secret
echo "alien123.com" | gh secret set DOCKERHUB_PASSWORD

# Verify secret is set
gh secret list
```

### Step 7: Commit and Push to GitHub
```bash
# Add all changes
git add .

# Commit with meaningful message
git commit -m "Add automated GPU deployment with Podman to Docker Hub and CraneCloud"

# Push to trigger GitHub Actions
git push origin main

# Monitor workflow
gh run watch

# Or view in browser
gh browse --repo mpairwe7/MLOPS_V1 -- actions
```

### Step 8: GitHub Actions Workflow (Automatic)

The workflow `.github/workflows/dockerhub-deploy.yml` will automatically:

1. **Checkout code** from repository
2. **Build Docker image** with GPU support
3. **Run tests** on the image
4. **Push to Docker Hub** with tags:
   - `landwind/retinal-screening-api:latest`
   - `landwind/retinal-screening-api:v1.0-{sha}`
5. **Generate deployment instructions** for CraneCloud

**Monitor Progress:**
```bash
# Watch workflow run
gh run watch

# View workflow logs
gh run view --log

# List all runs
gh run list
```

### Step 9: Deploy to Crane Cloud

#### Option A: Web Dashboard (Recommended)

1. **Login to Crane Cloud**
   ```
   URL: https://cranecloud.io
   ```

2. **Create New Application**
   - Click "**Create App**" or "**New Application**"
   - Choose deployment method: **Docker Image**

3. **Configure Application**
   ```
   App Name: retinal-screening-api
   Docker Image: landwind/retinal-screening-api:latest
   Port: 8080
   ```

4. **Resource Allocation** (GPU Support)
   ```
   CPU: 2 cores (minimum)
   Memory: 4GB (minimum for GPU)
   Storage: 10GB
   GPU: ✓ Enable GPU (if available)
     - GPU Type: NVIDIA
     - GPU Memory: 4GB+ recommended
   ```

5. **Environment Variables** (Optional)
   ```
   MODEL_PATH=/app/models/best_model_mobile.pt
   PORT=8080
   CUDA_VISIBLE_DEVICES=0
   WORKERS=1
   ```

6. **Deploy**
   - Click "**Deploy**" button
   - Wait 3-5 minutes for deployment
   - Note your app URL: `https://retinal-screening-api.cranecloud.io`

#### Option B: Crane Cloud CLI (Alternative)

```bash
# Install Crane Cloud CLI (if available)
# Visit: https://docs.cranecloud.io for installation

# Login
cranecloud login

# Create project (if needed)
cranecloud projects create medical-ml

# Deploy app with GPU support
cranecloud apps deploy retinal-screening-api \
  --image landwind/retinal-screening-api:latest \
  --port 8080 \
  --cpu 2 \
  --memory 4096 \
  --gpu nvidia \
  --gpu-memory 4096 \
  --replicas 1

# Get app URL
cranecloud apps info retinal-screening-api
```

### Step 10: Verify Deployment

#### Test Endpoints

```bash
# Set your app URL
export APP_URL="https://retinal-screening-api.cranecloud.io"

# 1. Health Check
curl $APP_URL/health
# Expected: {"status":"healthy","model":"best_model_mobile.pt","diseases":45}

# 2. Get Diseases List
curl $APP_URL/diseases
# Expected: {"diseases":["DR","ARMD","MH"...],"total":45}

# 3. Get Model Info
curl $APP_URL/
# Expected: API documentation/welcome page

# 4. Test Prediction (with sample image)
curl -X POST $APP_URL/predict \
  -F "file=@path/to/retinal_image.jpg" \
  -H "Content-Type: multipart/form-data"
# Expected: {"predictions":[...],"diseases":[...],"confidence":...}
```

#### Check GPU Usage (if available)

```bash
# SSH into Crane Cloud container (if SSH access available)
# Or check logs

# View GPU info
nvidia-smi

# Check CUDA availability
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

---

## 📊 Monitoring & Logs

### View Logs in Crane Cloud

1. Go to Crane Cloud Dashboard
2. Select app: `retinal-screening-api`
3. Click "**Logs**" tab
4. View real-time logs

### Monitor with CLI

```bash
# View logs
cranecloud logs retinal-screening-api

# Follow logs (real-time)
cranecloud logs retinal-screening-api --follow

# View specific timeframe
cranecloud logs retinal-screening-api --since 1h
```

---

## 🔧 Troubleshooting

### Issue: Build Takes Too Long

**Solution:**
```bash
# Build is downloading CUDA libraries (~2-3GB)
# First build: 10-15 minutes
# Subsequent builds: 2-3 minutes (cached)

# Check progress
podman ps -a
```

### Issue: GPU Not Available on CraneCloud

**Solution:**
- Check if Crane Cloud offers GPU resources
- Contact Crane Cloud support: support@cranecloud.io
- Use CPU-only deployment (model will still work, slower inference)

### Issue: Out of Memory

**Solution:**
```bash
# Increase memory in Crane Cloud dashboard
# Minimum: 4GB for GPU workloads
# Recommended: 8GB

# Or reduce model size in Dockerfile
# Use quantized model (already using .pt format)
```

### Issue: Port Already in Use

**Solution:**
```bash
# Check running containers
podman ps

# Stop conflicting container
podman stop <container_id>

# Or use different port
podman run -p 9090:8080 landwind/retinal-screening-api:latest
```

### Issue: Push to Docker Hub Fails

**Solution:**
```bash
# Re-login
podman login docker.io -u landwind

# Check image exists
podman images | grep retinal-screening-api

# Retry push
podman push landwind/retinal-screening-api:latest
```

---

## 🎯 Quick Commands Reference

### Docker/Podman Commands
```bash
# Build
podman build -t landwind/retinal-screening-api:latest .

# Run locally
podman run -p 8080:8080 landwind/retinal-screening-api:latest

# Push to Docker Hub
podman push landwind/retinal-screening-api:latest

# View images
podman images

# View running containers
podman ps

# View logs
podman logs <container_id>

# Stop container
podman stop <container_id>

# Remove container
podman rm <container_id>

# Remove image
podman rmi landwind/retinal-screening-api:latest
```

### GitHub CLI Commands
```bash
# Set secret
gh secret set DOCKERHUB_PASSWORD

# List secrets
gh secret list

# Trigger workflow manually
gh workflow run dockerhub-deploy.yml

# Watch workflow
gh run watch

# View logs
gh run view --log

# List workflows
gh workflow list

# View repository
gh browse
```

### Git Commands
```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "message"

# Push
git push origin main

# View remote
git remote -v

# View commit history
gh log --oneline
```

---

## 📦 Docker Image Information

### Image Details
- **Repository:** `landwind/retinal-screening-api`
- **Tags:** `latest`, `v1.0-gpu`
- **Base Image:** `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- **Size:** ~3-4 GB (with CUDA libraries)
- **Port:** 8080
- **Model:** `/app/models/best_model_mobile.pt` (26MB)

### GPU Support
- **CUDA Version:** 11.8.0
- **cuDNN Version:** 8
- **PyTorch:** 2.0.1+cu118
- **TorchVision:** 0.15.2+cu118

### API Endpoints
- `GET /` - API documentation
- `GET /health` - Health check
- `GET /diseases` - List all diseases
- `POST /predict` - Predict from image
- `GET /model/info` - Model information

---

## ✅ Deployment Checklist

### Pre-Deployment
- [x] Model file in `models/` folder
- [x] Podman installed and working
- [x] Docker Hub credentials configured
- [x] Git repository initialized
- [ ] Dockerfile reviewed and tested
- [ ] GitHub secrets configured
- [ ] Build completed successfully

### Deployment
- [ ] Image pushed to Docker Hub
- [ ] GitHub Actions workflow triggered
- [ ] Workflow completed successfully
- [ ] Crane Cloud app created
- [ ] GPU resources allocated
- [ ] Environment variables set
- [ ] App deployed successfully

### Post-Deployment
- [ ] Health endpoint returns 200 OK
- [ ] Diseases endpoint working
- [ ] Prediction endpoint tested
- [ ] GPU being utilized (if available)
- [ ] Logs reviewed for errors
- [ ] Performance metrics monitored
- [ ] Backup strategy configured

---

## 🚀 Continuous Deployment

### Automatic Deployment Workflow

Every push to `main` branch will:
1. Trigger GitHub Actions
2. Build new Docker image
3. Run tests
4. Push to Docker Hub
5. Update CraneCloud deployment (manual step)

### Manual Deployment Trigger

```bash
# Trigger workflow without pushing
gh workflow run dockerhub-deploy.yml

# Or commit and push
git commit -m "Update model/code" --allow-empty
git push origin main
```

---

## 📞 Support & Resources

### Crane Cloud
- **Website:** https://cranecloud.io
- **Docs:** https://docs.cranecloud.io
- **Support:** support@cranecloud.io
- **Status:** https://status.cranecloud.io

### Docker Hub
- **Website:** https://hub.docker.com
- **Username:** landwind
- **Repository:** landwind/retinal-screening-api

### GitHub
- **Repository:** https://github.com/mpairwe7/MLOPS_V1
- **Actions:** https://github.com/mpairwe7/MLOPS_V1/actions
- **Issues:** https://github.com/mpairwe7/MLOPS_V1/issues

---

**Current Status:** 🔨 Building Docker image...
**Next Step:** Wait for build to complete, then push to Docker Hub
