# 🚀 CraneCloud Quick Start Guide

## ✅ Your Image is Ready!

Your Docker image `landwind/retinal-screening-api:latest` (4.91 GB) is successfully pushed to Docker Hub and ready for deployment.

---

## 📝 Step-by-Step Deployment (5 Minutes)

### Step 1: Login to CraneCloud
1. Open your browser and go to: **https://cranecloud.io**
2. Click **"Sign In"** (or **"Sign Up"** if you don't have an account)
3. Login with your credentials

### Step 2: Create New Application
1. Look for a button like **"Create Application"**, **"+ New App"**, or **"Deploy"**
2. Click it to start the deployment wizard

### Step 3: Fill in Application Details

Copy and paste these exact values into the CraneCloud form:

#### Basic Information
```
Application Name: retinal-screening-api
Description: Multi-model retinal disease screening API with GPU support
```

#### Container Configuration
```
Docker Image: landwind/retinal-screening-api:latest
Container Port: 8080
Protocol: HTTP
```

#### Resource Allocation
```
CPU: 2 cores
Memory: 4096 MB (or 4 GB)
Storage: 20 GB
GPU: 1 (if available, otherwise leave as 0 or None)
```

#### Health Check Settings
```
Health Check Path: /health
Health Check Port: 8080
Initial Delay: 30 seconds
Check Interval: 10 seconds
Timeout: 5 seconds
```

#### Environment Variables (Optional - only if form asks)
```
PORT = 8080
PYTHONUNBUFFERED = 1
```

### Step 4: Deploy!
1. Review all the settings
2. Click **"Deploy"** or **"Create Application"**
3. Wait 2-5 minutes for deployment to complete
4. Watch the deployment logs (usually shown in the dashboard)

### Step 5: Get Your App URL
After deployment completes, CraneCloud will give you a URL like:
```
https://retinal-screening-api-xxxxx.cranecloud.io
```
or
```
https://your-app.cranecloud.io
```

**Save this URL!** You'll need it for testing.

---

## 🧪 Test Your Deployment

Once you have your CraneCloud URL, test these endpoints:

### Test 1: Health Check
```bash
curl https://YOUR-APP-URL.cranecloud.io/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": false,
  "device": "cuda:0"
}
```

### Test 2: Get Diseases List
```bash
curl https://YOUR-APP-URL.cranecloud.io/diseases
```

**Expected Response:** JSON array with 45 disease names

### Test 3: API Documentation
Open in your browser:
```
https://YOUR-APP-URL.cranecloud.io/docs
```

You should see an interactive Swagger UI with all available endpoints.

---

## 🎯 Your API Endpoints

Once deployed, your API will have:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/diseases` | GET | List all 45 supported diseases |
| `/predict` | POST | Single image prediction |
| `/batch_predict` | POST | Batch predictions |
| `/models` | GET | List available models |
| `/docs` | GET | Interactive API documentation |

---

## 🔧 Troubleshooting

### Problem: Container won't start
**Solution:** 
- Increase memory to 4096 MB (4 GB) minimum
- Check CraneCloud logs for error messages
- Verify the image name is exactly: `landwind/retinal-screening-api:latest`

### Problem: Health check failing
**Solution:**
- Increase Initial Delay to 60 seconds (model loading takes time)
- Verify Health Path is `/health` (with the slash)
- Check that Health Port is `8080`

### Problem: GPU not available
**Solution:**
- GPU is optional - the app will work on CPU
- If you need GPU, contact CraneCloud support to enable GPU instances
- Check if GPU instances are available in your region

### Problem: Image pull failed
**Solution:**
- Verify the Docker Hub repository is public (not private)
- Try the full image path: `docker.io/landwind/retinal-screening-api:latest`
- Wait a few minutes and try again (Docker Hub rate limits)

---

## 📊 Monitoring Your Deployment

After deployment, monitor these metrics:

- **Response Time:** Should be < 5 seconds for `/health`
- **Memory Usage:** Should stay < 80% of allocated
- **CPU Usage:** Should be < 80% under normal load
- **Error Rate:** Should be < 1%

Most cloud platforms show these metrics in the dashboard.

---

## 🎉 Success Checklist

Once deployed, verify:

- [ ] Application status shows "Running" or "Healthy"
- [ ] `/health` endpoint returns 200 OK
- [ ] `/diseases` endpoint returns 45 diseases
- [ ] `/docs` shows interactive API documentation
- [ ] No errors in application logs
- [ ] HTTPS is working (URL starts with https://)

---

## 📞 Need Help?

**CraneCloud Support:**
- Email: support@cranecloud.io
- Docs: https://docs.cranecloud.io

**Your Project Files:**
- Detailed Guide: `CRANECLOUD_DEPLOYMENT_STEPS.md`
- Configuration: `cranecloud-config.yaml`
- Test Script: `test-cranecloud-deployment.sh`

**Docker Image:**
- Docker Hub: https://hub.docker.com/r/landwind/retinal-screening-api
- Image Size: 4.91 GB
- Last Updated: October 31, 2025

---

## 🚀 Ready to Deploy!

You have everything you need:
- ✅ Docker image built and pushed
- ✅ Configuration file generated
- ✅ Deployment guide ready
- ✅ Test scripts prepared

**Just go to https://cranecloud.io and follow the steps above!**

Good luck! 🎉
