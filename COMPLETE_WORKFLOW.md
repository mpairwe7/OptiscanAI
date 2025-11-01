# 🎯 Complete Workflow - Retinal API with Ngrok & CraneCloud

## Overview

This document provides the complete end-to-end workflow for:
1. Building Docker image with ngrok support
2. Testing locally with public URL
3. Deploying to CraneCloud with ngrok

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] Podman or Docker installed
- [ ] Python 3.10+ in virtual environment
- [ ] GitHub account (for mpairwe7/MLOPS_V1)
- [ ] Docker Hub account (landwind)
- [ ] Ngrok account (free tier, https://ngrok.com)
- [ ] CraneCloud account (https://cranecloud.io)

---

## 🚀 Complete Workflow (Step-by-Step)

### Phase 1: Local Development with Ngrok

#### Step 1.1: Get Ngrok Token
```bash
# Visit https://ngrok.com and sign up (free, no credit card)
# Copy your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken
# Save it as environment variable:
export NGROK_AUTHTOKEN="your_actual_token_here"
```

#### Step 1.2: Activate Virtual Environment
```bash
cd /home/darkhorse/Downloads/MLOPS_V1
source venv/bin/activate
```

#### Step 1.3: Build Docker Image with Ngrok
```bash
# Build and test the image
./build-with-ngrok.sh
```

Expected output:
- ✓ Image built successfully
- ✓ API-only mode working
- ✓ API-with-ngrok mode working (if NGROK_AUTHTOKEN set)
- Images: `landwind/retinal-screening-api:v2.1-ngrok` and `:latest`

#### Step 1.4: Test Locally with Ngrok
```bash
# Start container with ngrok
podman run -d --name retinal-api-dev \
  -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="$NGROK_AUTHTOKEN" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok

# Wait 15 seconds for services to start
sleep 15

# Check logs for public URL
podman logs retinal-api-dev | grep "Public URL"

# Or query ngrok API
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])")

echo "Your public URL: $NGROK_URL"
```

#### Step 1.5: Test the Public URL
```bash
# Test health endpoint
curl $NGROK_URL/health | python3 -m json.tool

# Test diseases endpoint
curl $NGROK_URL/diseases | python3 -m json.tool | head -20

# Open API docs in browser
xdg-open "$NGROK_URL/docs"
# Or on macOS: open "$NGROK_URL/docs"
```

#### Step 1.6: Share with Team/Clients
```bash
# Your API is now publicly accessible!
echo "Share this URL: $NGROK_URL"
echo "API Documentation: $NGROK_URL/docs"
echo "Health Check: $NGROK_URL/health"
echo "Supported Diseases: $NGROK_URL/diseases"
```

#### Step 1.7: Monitor with Ngrok Dashboard
```bash
# Open ngrok dashboard in browser
xdg-open "http://localhost:4040"

# Dashboard shows:
# - Real-time request history
# - Request/response inspection
# - Replay functionality
# - Connection statistics
```

#### Step 1.8: Stop Local Testing
```bash
# When done testing
podman stop retinal-api-dev
podman rm retinal-api-dev
```

---

### Phase 2: Push to Docker Hub

#### Step 2.1: Login to Docker Hub
```bash
podman login docker.io
# Username: landwind
# Password: alien123.com
```

#### Step 2.2: Push Images
```bash
# Push both tags
podman push landwind/retinal-screening-api:v2.1-ngrok
podman push landwind/retinal-screening-api:latest

# Verify on Docker Hub
xdg-open "https://hub.docker.com/r/landwind/retinal-screening-api/tags"
```

---

### Phase 3: Deploy to CraneCloud

#### Step 3.1: Login to CraneCloud
```bash
# Open CraneCloud in browser
xdg-open "https://cranecloud.io"

# Login with your credentials
```

#### Step 3.2: Create Application

**Option A: Without Ngrok (Standard Deployment)**

Fill in CraneCloud form:
```yaml
Application Name: retinal-screening-api
Description: Multi-model retinal disease screening API

Container Image: landwind/retinal-screening-api:latest
Container Port: 8080
Protocol: HTTP

CPU: 2 cores
Memory: 4096 MB (4 GB)
Storage: 20 GB
GPU: 1 (if available)

Health Check Path: /health
Health Check Port: 8080
Initial Delay: 30 seconds
Period: 10 seconds

Environment Variables:
  PORT: 8080
  PYTHONUNBUFFERED: 1
```

**Option B: With Ngrok (Public URL via Ngrok)**

Same as Option A, but add:
```yaml
Command Override: api-with-ngrok

Additional Environment Variables:
  NGROK_AUTHTOKEN: your_ngrok_token_here
  NGROK_ENABLED: true
  NGROK_REGION: us
```

#### Step 3.3: Deploy
```bash
# Click "Deploy" or "Create Application" in CraneCloud
# Wait 2-5 minutes for deployment to complete
```

#### Step 3.4: Get Your URLs

**If deployed without ngrok:**
- CraneCloud will provide URL: `https://retinal-screening-api-xxx.cranecloud.io`

**If deployed with ngrok:**
- Check CraneCloud logs for: `"Public URL: https://xyz.ngrok.io"`
- You'll have TWO URLs:
  1. CraneCloud URL (if available)
  2. Ngrok URL (for guaranteed public access)

#### Step 3.5: Test CraneCloud Deployment
```bash
# Save your CraneCloud URL
CRANECLOUD_URL="https://your-app.cranecloud.io"

# Test health
curl $CRANECLOUD_URL/health

# Test diseases
curl $CRANECLOUD_URL/diseases | python3 -m json.tool

# Open docs
xdg-open "$CRANECLOUD_URL/docs"
```

---

### Phase 4: Production Monitoring

#### Step 4.1: Monitor Health
```bash
# Check health every 30 seconds
watch -n 30 "curl -s $CRANECLOUD_URL/health | python3 -m json.tool"
```

#### Step 4.2: View Logs
```bash
# In CraneCloud dashboard:
# - Go to your application
# - Click "Logs" tab
# - Monitor for errors or ngrok URLs
```

#### Step 4.3: Scale if Needed
```bash
# In CraneCloud dashboard:
# - Go to Settings
# - Adjust replicas: 1 → 2 → 3
# - Adjust resources if needed
```

---

## 🔄 Quick Reference Commands

### Local Testing
```bash
# Start with ngrok
podman run -d -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="token" -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok

# Get ngrok URL
curl -s http://localhost:4040/api/tunnels | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

# Test API
curl http://localhost:8080/health
```

### Docker Hub Operations
```bash
# Login
podman login docker.io

# Push
podman push landwind/retinal-screening-api:latest

# Pull (on another machine)
podman pull landwind/retinal-screening-api:latest
```

### CraneCloud Deployment
```bash
# URLs to remember:
# - Dashboard: https://cranecloud.io
# - Docker Hub: https://hub.docker.com/r/landwind/retinal-screening-api
# - Ngrok Dashboard: https://dashboard.ngrok.com
```

---

## 🎓 Use Case Examples

### Use Case 1: Quick Demo to Stakeholder
```bash
# 1. Start local container with ngrok
podman run -d -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="$NGROK_AUTHTOKEN" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok

# 2. Get public URL (takes ~15 seconds)
sleep 15
DEMO_URL=$(curl -s http://localhost:4040/api/tunnels | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])")

# 3. Share URL with stakeholder
echo "Demo URL: $DEMO_URL/docs"

# 4. They can access immediately from anywhere!
```

### Use Case 2: Mobile App Testing
```bash
# 1. Start API with ngrok
podman run -d -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="$NGROK_AUTHTOKEN" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok

# 2. Get URL
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])")

# 3. Configure mobile app to use: $NGROK_URL
# 4. Test on iOS/Android over cellular or WiFi
# 5. No local network configuration needed!
```

### Use Case 3: Production Deployment
```bash
# Option 1: CraneCloud only (no ngrok)
# - Use CraneCloud dashboard
# - Follow Phase 3, Option A
# - Best for: Stable production deployments

# Option 2: CraneCloud + Ngrok (dual URLs)
# - Use CraneCloud dashboard
# - Follow Phase 3, Option B
# - Best for: Flexibility, guaranteed public access
```

---

## 🛠️ Troubleshooting

### Issue: Ngrok URL not appearing
```bash
# Check container logs
podman logs container_name

# Check ngrok API
curl http://localhost:4040/api/tunnels

# Verify authtoken
echo $NGROK_AUTHTOKEN
```

### Issue: CraneCloud deployment failing
```bash
# Check CraneCloud logs
# Common issues:
# - Image not found: Verify image name is exact
# - Out of memory: Increase to 4096 MB
# - Health check failing: Increase initial delay to 60s
```

### Issue: API slow to respond
```bash
# Check resource usage in CraneCloud
# Increase CPU/Memory if needed
# Consider GPU instance for better performance
```

---

## 📊 Performance Comparison

| Deployment | Setup Time | Latency | Public URL | Cost | Best For |
|------------|-----------|---------|------------|------|----------|
| Local only | 2 min | ~10ms | No | Free | Development |
| Local + Ngrok | 2 min | ~50-100ms | Yes | Free | Testing/Demo |
| CraneCloud | 5 min | ~20-40ms | Yes | ~$20/mo | Production |
| CraneCloud + Ngrok | 5 min | ~50ms | Yes (2 URLs) | ~$20/mo | Flexible Prod |

---

## ✅ Success Checklist

- [ ] Built Docker image with ngrok support
- [ ] Tested locally (API only mode)
- [ ] Tested locally with ngrok (got public URL)
- [ ] Shared ngrok URL with team member (worked!)
- [ ] Pushed image to Docker Hub
- [ ] Deployed to CraneCloud
- [ ] Verified CraneCloud health endpoint
- [ ] Tested API endpoints on CraneCloud
- [ ] Monitored logs (no errors)
- [ ] Documented production URL

---

## 📚 Additional Resources

- **Ngrok Guide**: NGROK_INTEGRATION_GUIDE.md
- **Quick Reference**: NGROK_QUICK_REFERENCE.txt
- **CraneCloud Guide**: CRANECLOUD_DEPLOYMENT_STEPS.md
- **Build Script**: build-with-ngrok.sh
- **Test Script**: test-cranecloud-deployment.sh

---

**Last Updated**: October 31, 2025  
**Version**: 2.1.0 (with ngrok support)  
**Project**: MLOPS_V1 - Retinal Disease Screening API
