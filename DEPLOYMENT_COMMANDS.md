# Deployment Commands Summary

## Current Status ✅

### 1. Container Build - COMPLETE ✅
- **Image**: `localhost/retinal-screening-gpu:latest`
- **Size**: 11.8 GB
- **Status**: Built successfully with GPU support (CUDA 11.8, cuDNN 8)

### 2. Local Testing - COMPLETE ✅
- **Container tested**: API running on port 8000
- **Health endpoint**: `http://127.0.0.1:8000/health`
- **Response**:
  ```json
  {
    "message": "Retinal Disease Classification API",
    "version": "1.0.0",
    "status": "running",
    "model_loaded": false
  }
  ```
- **API Documentation**: `http://127.0.0.1:8000/docs`

### 3. Docker Hub Push - IN PROGRESS 🔄
- **Docker Hub Username**: `landwind`
- **Repository**: `docker.io/landwind/retinal-screening-gpu:latest`
- **Status**: Uploading (11.8 GB, ETA: 10-15 minutes)
- **Log file**: `dockerhub_push.log`

---

## Quick Deployment Commands

### Step 1: Check Docker Hub Push Status
```bash
# Check if push is complete
tail -f dockerhub_push.log

# Or check terminal output
podman images | grep retinal
```

### Step 2: Deploy to GCP with GPU

#### Option A: Using Automated Script (Recommended)
```bash
./deploy_to_gcp.sh landwind
```

#### Option B: Manual Deployment
```bash
# Set variables
export DOCKER_USERNAME=landwind
export PROJECT_ID=retinal-screening-prod
export ZONE=us-central1-a
export INSTANCE_NAME=retinal-screening-gpu-instance

# Set active project
gcloud config set project ${PROJECT_ID}

# Enable APIs
gcloud services enable compute.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com

# Create firewall rule
gcloud compute firewall-rules create allow-retinal-api \
    --allow tcp:8000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow access to Retinal Screening API"

# Create GPU instance with startup script
gcloud compute instances create ${INSTANCE_NAME} \
    --zone=${ZONE} \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --maintenance-policy=TERMINATE \
    --metadata=startup-script='#!/bin/bash
# Install Docker
apt-get update
apt-get install -y docker.io

# Install NVIDIA drivers and container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit nvidia-driver-535
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Pull and run container
docker pull docker.io/landwind/retinal-screening-gpu:latest
docker run -d \
  --name retinal-screening \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  docker.io/landwind/retinal-screening-gpu:latest
'

# Get external IP
gcloud compute instances describe ${INSTANCE_NAME} \
    --zone=${ZONE} \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

### Step 3: Verify Deployment
```bash
# Get instance external IP
EXTERNAL_IP=$(gcloud compute instances describe retinal-screening-gpu-instance \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

# Test health endpoint
curl http://${EXTERNAL_IP}:8000/health

# View API documentation
echo "API Docs: http://${EXTERNAL_IP}:8000/docs"
```

---

## Monitoring & Management

### SSH into Instance
```bash
gcloud compute ssh retinal-screening-gpu-instance --zone=us-central1-a
```

### View Container Logs
```bash
# From local machine
gcloud compute ssh retinal-screening-gpu-instance \
    --zone=us-central1-a \
    --command='docker logs -f retinal-screening'

# Or after SSH
docker logs -f retinal-screening
```

### Check GPU Status
```bash
gcloud compute ssh retinal-screening-gpu-instance \
    --zone=us-central1-a \
    --command='nvidia-smi'
```

### Stop Instance (Save Costs)
```bash
gcloud compute instances stop retinal-screening-gpu-instance --zone=us-central1-a
```

### Start Instance
```bash
gcloud compute instances start retinal-screening-gpu-instance --zone=us-central1-a
```

### Delete Instance
```bash
gcloud compute instances delete retinal-screening-gpu-instance --zone=us-central1-a
```

---

## Cost Optimization

### Current Configuration
- **Machine**: n1-standard-4 (4 vCPUs, 15 GB RAM)
- **GPU**: 1x NVIDIA Tesla T4
- **Estimated Cost**: ~$0.35/hour (~$272/month if 24/7)

### Cost Saving Tips

#### 1. Stop When Not in Use (Saves ~70%)
```bash
# Automatic shutdown at 6 PM
gcloud compute instances add-metadata retinal-screening-gpu-instance \
    --zone=us-central1-a \
    --metadata=shutdown-script='#!/bin/bash
systemctl poweroff'
```

#### 2. Use Scheduled Instances
```bash
# Create schedule (weekdays 9 AM - 6 PM)
gcloud compute resource-policies create instance-schedule retinal-schedule \
    --vm-start-schedule='0 9 * * 1-5' \
    --vm-stop-schedule='0 18 * * 1-5' \
    --timezone='America/New_York'

# Apply to instance
gcloud compute instances add-resource-policies retinal-screening-gpu-instance \
    --resource-policies=retinal-schedule \
    --zone=us-central1-a
```

#### 3. Cost with 8 hours/day (Business Hours)
- **Daily**: $0.35/hour × 8 hours = $2.80/day
- **Monthly**: $2.80 × 22 workdays = **~$66/month** (75% savings!)

---

## Troubleshooting

### Container Not Starting
```bash
# Check startup script logs
gcloud compute instances get-serial-port-output retinal-screening-gpu-instance \
    --zone=us-central1-a

# SSH and check Docker
gcloud compute ssh retinal-screening-gpu-instance --zone=us-central1-a
docker ps -a
docker logs retinal-screening
```

### GPU Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall if needed
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### API Not Accessible
```bash
# Check firewall rules
gcloud compute firewall-rules list | grep retinal

# Test from instance
curl http://localhost:8000/health

# Check if container is running
docker ps | grep retinal
```

---

## Next Steps After Deployment

1. **Upload Model File** (if using PyTorch instead of ONNX):
   ```bash
   gcloud compute scp models/GraphCLIP_fold1_best.pth \
       retinal-screening-gpu-instance:/tmp/ \
       --zone=us-central1-a
   
   # Then SSH and move to container volume
   gcloud compute ssh retinal-screening-gpu-instance --zone=us-central1-a
   sudo docker cp /tmp/GraphCLIP_fold1_best.pth retinal-screening:/app/models/
   sudo docker restart retinal-screening
   ```

2. **Setup Monitoring**:
   ```bash
   # Enable Cloud Monitoring
   gcloud services enable monitoring.googleapis.com
   
   # Create uptime check
   gcloud monitoring uptime-checks create http retinal-api-check \
       --resource-type=uptime-url \
       --display-name="Retinal API Health Check" \
       --http-check-path=/health \
       --port=8000
   ```

3. **Setup Cost Alerts**:
   - Go to GCP Console > Billing > Budgets & Alerts
   - Create budget: $100/month
   - Set alerts at 50%, 90%, 100%

4. **Configure Domain Name** (Optional):
   ```bash
   # Reserve static IP
   gcloud compute addresses create retinal-api-ip --region=us-central1
   
   # Get the IP
   gcloud compute addresses describe retinal-api-ip --region=us-central1
   
   # Update instance to use static IP
   gcloud compute instances delete-access-config retinal-screening-gpu-instance \
       --zone=us-central1-a --access-config-name="external-nat"
   
   gcloud compute instances add-access-config retinal-screening-gpu-instance \
       --zone=us-central1-a \
       --address=STATIC_IP_ADDRESS
   ```

---

## Support & Documentation

- **Docker Hub**: https://hub.docker.com/r/landwind/retinal-screening-gpu
- **GCP Console**: https://console.cloud.google.com
- **API Documentation**: http://YOUR_EXTERNAL_IP:8000/docs

---

**Last Updated**: October 30, 2025
**Deployment Status**: Docker Hub Push In Progress → GCP Deployment Pending
