# Complete Deployment Guide: Mobile → Podman → Docker Hub → GCP

**Project**: Multi-Disease Retinal Screening AI  
**Date**: October 30, 2025  
**Target**: Production deployment with GPU acceleration

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Mobile Optimization](#step-1-mobile-optimization)
3. [Step 2: Containerization with Podman](#step-2-containerization-with-podman)
4. [Step 3: Push to Docker Hub](#step-3-push-to-docker-hub)
5. [Step 4: Deploy on GCP](#step-4-deploy-on-gcp)
6. [Step 5: Monitoring & Maintenance](#step-5-monitoring--maintenance)

---

## Prerequisites

### Local Machine Requirements
- ✅ Python 3.8+
- ✅ PyTorch 1.12+
- ✅ Podman installed
- ✅ NVIDIA GPU (optional for local testing)
- ✅ Docker Hub account
- ✅ GCP account with billing enabled

### Install Required Tools

```bash
# Install Podman (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y podman

# Install NVIDIA Container Toolkit (for GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

---

## Step 1: Mobile Optimization

### 1.1 Export Model to ONNX

Create optimization script:

```bash
cd /home/darkhorse/Downloads/MLOPS_V1
python src/optimize_for_deployment.py
```

This will:
- ✅ Convert PyTorch model → ONNX format
- ✅ Apply dynamic quantization (330MB → ~80MB)
- ✅ Validate ONNX model
- ✅ Benchmark inference speed
- ✅ Create deployment package

**Expected Output:**
```
models/
├── GraphCLIP_fold1_best.pth          # Original (330 MB)
├── GraphCLIP_optimized.onnx           # Optimized (80 MB)
└── exports/
    ├── config.json
    ├── disease_info.json
    └── optimization_report.json
```

### 1.2 Test Optimized Model

```bash
python deployment/test_optimized_model.py --model models/GraphCLIP_optimized.onnx
```

**Verify:**
- ✅ Inference time < 100ms
- ✅ Accuracy within 2% of original
- ✅ Model size < 100MB

---

## Step 2: Containerization with Podman

### 2.1 Review Dockerfile

The `Dockerfile.gpu` uses multi-stage build for optimization:

```dockerfile
# Stage 1: Builder (installs dependencies)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

# Stage 2: Runtime (minimal image)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# Copies only necessary files
```

**Key Features:**
- ✅ NVIDIA CUDA 11.8 base image
- ✅ GPU support enabled
- ✅ Multi-stage build (smaller final image)
- ✅ Health checks included
- ✅ Non-root user for security

### 2.2 Build Container with Podman

```bash
cd /home/darkhorse/Downloads/MLOPS_V1

# Build the image
podman build \
  -f Dockerfile.gpu \
  -t retinal-screening-gpu:latest \
  --format docker \
  .

# Verify build
podman images | grep retinal-screening
```

**Expected Output:**
```
REPOSITORY                  TAG         IMAGE ID      CREATED        SIZE
retinal-screening-gpu       latest      abc123def456  1 minute ago   2.5 GB
```

### 2.3 Test Container Locally

```bash
# Run container with GPU
podman run -d \
  --name retinal-test \
  --device nvidia.com/gpu=all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/GraphCLIP_optimized.onnx \
  retinal-screening-gpu:latest

# Check logs
podman logs -f retinal-test

# Test API
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/info
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model": "GraphCLIP",
  "gpu_available": true,
  "version": "1.0.0"
}
```

### 2.4 Test Inference

```bash
# Test with sample image
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@test_image.jpg" \
  -H "Content-Type: multipart/form-data"
```

**Expected Response:**
```json
{
  "detected_diseases": [
    {
      "code": "DR",
      "name": "Diabetic Retinopathy",
      "probability": 0.87,
      "severity": "HIGH",
      "urgency": "URGENT"
    }
  ],
  "referral_urgency": "URGENT",
  "inference_time_ms": 45.2
}
```

### 2.5 Benchmark Performance

```bash
python deployment/benchmark_api.py --endpoint http://localhost:8000
```

**Target Metrics:**
- ✅ Inference time: < 100ms
- ✅ Throughput: > 10 requests/sec
- ✅ GPU utilization: 60-80%
- ✅ Memory usage: < 4GB

---

## Step 3: Push to Docker Hub

### 3.1 Login to Docker Hub

```bash
# Login with Podman
podman login docker.io

# Enter your Docker Hub credentials
Username: your_username
Password: your_password
```

### 3.2 Tag Image

```bash
# Replace 'your_username' with your Docker Hub username
DOCKER_USERNAME="your_username"

podman tag \
  retinal-screening-gpu:latest \
  docker.io/$DOCKER_USERNAME/retinal-screening-gpu:latest

podman tag \
  retinal-screening-gpu:latest \
  docker.io/$DOCKER_USERNAME/retinal-screening-gpu:v1.0.0
```

### 3.3 Push to Registry

```bash
# Push latest tag
podman push docker.io/$DOCKER_USERNAME/retinal-screening-gpu:latest

# Push version tag
podman push docker.io/$DOCKER_USERNAME/retinal-screening-gpu:v1.0.0
```

**Expected Output:**
```
Getting image source signatures
Copying blob sha256:abc123...
Copying config sha256:def456...
Writing manifest to image destination
Storing signatures
```

### 3.4 Verify on Docker Hub

Visit: `https://hub.docker.com/r/$DOCKER_USERNAME/retinal-screening-gpu`

**Check:**
- ✅ Image visible in repository
- ✅ Both tags (latest, v1.0.0) present
- ✅ Image size ~2-3 GB

---

## Step 4: Deploy on GCP

### 4.1 Setup GCP Project

```bash
# Set project ID
export PROJECT_ID="retinal-screening-prod"
export REGION="us-central1"
export ZONE="us-central1-a"

# Create project (if new)
gcloud projects create $PROJECT_ID --name="Retinal Screening"

# Set active project
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
```

### 4.2 Create GPU Instance

```bash
# Create instance with NVIDIA T4 GPU
gcloud compute instances create retinal-screening-gpu \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd \
  --metadata=cos-metrics-enabled=true \
  --tags=http-server,https-server \
  --scopes=cloud-platform

# Create firewall rule
gcloud compute firewall-rules create allow-retinal-api \
  --project=$PROJECT_ID \
  --allow=tcp:8000 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server \
  --description="Allow access to retinal screening API"
```

**GPU Instance Options:**

| GPU Type | vCPUs | RAM | GPU Memory | Cost/hour (approx) |
|----------|-------|-----|------------|-------------------|
| T4       | 4     | 15GB | 16GB      | $0.35            |
| V100     | 8     | 30GB | 16GB      | $2.48            |
| A100     | 12    | 85GB | 40GB      | $3.67            |

**Recommendation**: Start with T4 for cost-effectiveness

### 4.3 SSH into Instance

```bash
gcloud compute ssh retinal-screening-gpu --zone=$ZONE
```

### 4.4 Setup Instance

```bash
# Install Docker (on Container-Optimized OS)
# Docker is pre-installed on COS

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU
nvidia-smi
```

### 4.5 Pull and Run Container

```bash
# Pull image from Docker Hub
docker pull $DOCKER_USERNAME/retinal-screening-gpu:latest

# Run container with GPU
docker run -d \
  --name retinal-api \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -e MODEL_PATH=/app/models/GraphCLIP_optimized.onnx \
  -e LOG_LEVEL=INFO \
  $DOCKER_USERNAME/retinal-screening-gpu:latest

# Check logs
docker logs -f retinal-api
```

### 4.6 Test Deployment

```bash
# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe retinal-screening-gpu \
  --zone=$ZONE \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "API URL: http://$EXTERNAL_IP:8000"

# Test health endpoint
curl http://$EXTERNAL_IP:8000/health

# Test API info
curl http://$EXTERNAL_IP:8000/api/v1/info
```

### 4.7 Setup Load Balancer (Optional - for production)

```bash
# Create instance group
gcloud compute instance-groups unmanaged create retinal-screening-group \
  --zone=$ZONE

gcloud compute instance-groups unmanaged add-instances retinal-screening-group \
  --instances=retinal-screening-gpu \
  --zone=$ZONE

# Create health check
gcloud compute health-checks create http retinal-health-check \
  --port=8000 \
  --request-path=/health \
  --check-interval=10s \
  --timeout=5s \
  --unhealthy-threshold=3 \
  --healthy-threshold=2

# Create backend service
gcloud compute backend-services create retinal-backend \
  --protocol=HTTP \
  --health-checks=retinal-health-check \
  --global

# Add instance group to backend
gcloud compute backend-services add-backend retinal-backend \
  --instance-group=retinal-screening-group \
  --instance-group-zone=$ZONE \
  --global

# Create URL map
gcloud compute url-maps create retinal-lb \
  --default-service=retinal-backend

# Create HTTP proxy
gcloud compute target-http-proxies create retinal-http-proxy \
  --url-map=retinal-lb

# Create forwarding rule
gcloud compute forwarding-rules create retinal-forwarding-rule \
  --global \
  --target-http-proxy=retinal-http-proxy \
  --ports=80

# Get load balancer IP
gcloud compute forwarding-rules describe retinal-forwarding-rule \
  --global \
  --format="get(IPAddress)"
```

---

## Step 5: Monitoring & Maintenance

### 5.1 Setup Cloud Monitoring

```bash
# Create uptime check
gcloud monitoring uptime-checks create retinal-uptime \
  --display-name="Retinal API Uptime" \
  --http-check \
  --host=$EXTERNAL_IP \
  --port=8000 \
  --path=/health

# Create alert policy
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Retinal API Down" \
  --condition-display-name="API Unhealthy" \
  --condition-threshold-value=1 \
  --condition-threshold-duration=300s
```

### 5.2 View Logs

```bash
# View application logs
gcloud logging read "resource.type=gce_instance AND \
  resource.labels.instance_id=INSTANCE_ID" \
  --limit=50 \
  --format=json

# Stream logs
gcloud logging tail "resource.type=gce_instance" \
  --format="value(textPayload)"
```

### 5.3 Setup Automated Backups

```bash
# Create snapshot schedule
gcloud compute resource-policies create snapshot-schedule daily-backup \
  --region=$REGION \
  --max-retention-days=7 \
  --on-source-disk-delete=keep-auto-snapshots \
  --daily-schedule \
  --start-time=02:00

# Attach to disk
gcloud compute disks add-resource-policies retinal-screening-gpu \
  --resource-policies=daily-backup \
  --zone=$ZONE
```

### 5.4 Cost Optimization

```bash
# Create instance schedule (stop at night)
gcloud compute resource-policies create instance-schedule retinal-schedule \
  --region=$REGION \
  --vm-start-schedule="0 8 * * *" \
  --vm-stop-schedule="0 20 * * *" \
  --timezone="Africa/Kampala"

# Apply schedule to instance
gcloud compute instances add-resource-policies retinal-screening-gpu \
  --resource-policies=retinal-schedule \
  --zone=$ZONE
```

**Cost Estimates (monthly):**
- T4 GPU (8 hrs/day, 22 days/month): ~$46
- T4 GPU (24/7): ~$252
- Network egress (100GB): ~$12
- Storage (50GB SSD): ~$8.50
- **Total (8hrs/day)**: ~$66/month
- **Total (24/7)**: ~$272/month

---

## Quick Commands Reference

### Local Development
```bash
# Build container
podman build -f Dockerfile.gpu -t retinal-screening-gpu:latest .

# Run locally
podman run -d --device nvidia.com/gpu=all -p 8000:8000 retinal-screening-gpu:latest

# Test API
curl http://localhost:8000/health
```

### Docker Hub
```bash
# Login
podman login docker.io

# Tag & Push
podman tag retinal-screening-gpu:latest docker.io/$USERNAME/retinal-screening-gpu:latest
podman push docker.io/$USERNAME/retinal-screening-gpu:latest
```

### GCP Deployment
```bash
# Create instance
gcloud compute instances create retinal-screening-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1

# SSH and deploy
gcloud compute ssh retinal-screening-gpu --zone=us-central1-a
docker pull $USERNAME/retinal-screening-gpu:latest
docker run -d --gpus all -p 8000:8000 retinal-screening-gpu:latest
```

### Monitoring
```bash
# Check logs
docker logs -f retinal-api

# Monitor GPU
nvidia-smi -l 5

# Check API status
curl http://$EXTERNAL_IP:8000/health
```

---

## Troubleshooting

### Issue: GPU not detected in container

**Solution:**
```bash
# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Issue: Container fails to start

**Solution:**
```bash
# Check logs
docker logs retinal-api

# Check disk space
df -h

# Verify image
docker images | grep retinal-screening
```

### Issue: Slow inference times

**Solution:**
```bash
# Check GPU utilization
nvidia-smi

# Monitor container resources
docker stats retinal-api

# Optimize batch size in config
```

### Issue: High costs

**Solution:**
- Use instance scheduling (8 hrs/day vs 24/7)
- Consider preemptible instances (up to 80% cheaper)
- Use Cloud Run for serverless (pay per request)
- Optimize model size further

---

## Security Best Practices

1. **API Authentication**
   - Implement API keys or JWT tokens
   - Use HTTPS/TLS for all connections
   - Rate limiting to prevent abuse

2. **Network Security**
   - Use VPC and private IPs
   - Restrict firewall rules to specific IP ranges
   - Enable Cloud Armor for DDoS protection

3. **Container Security**
   - Run as non-root user (already implemented)
   - Scan images for vulnerabilities
   - Keep base images updated
   - Use secrets management for sensitive data

4. **Data Privacy**
   - Encrypt data at rest and in transit
   - Implement HIPAA-compliant logging
   - Regular security audits
   - Patient data anonymization

---

## Next Steps

- [ ] Complete Step 1: Mobile Optimization
- [ ] Complete Step 2: Build with Podman
- [ ] Complete Step 3: Push to Docker Hub
- [ ] Complete Step 4: Deploy on GCP
- [ ] Setup monitoring and alerts
- [ ] Configure automated backups
- [ ] Implement CI/CD pipeline
- [ ] Load testing and optimization
- [ ] Security hardening
- [ ] Documentation for clinical users

---

## Support & Resources

- **Documentation**: `/deployment/API_DOCUMENTATION.md`
- **API Reference**: `http://your-api/docs`
- **Issues**: GitHub Issues
- **GCP Console**: https://console.cloud.google.com
- **Docker Hub**: https://hub.docker.com

---

**Last Updated**: October 30, 2025  
**Version**: 1.0.0
