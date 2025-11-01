# 🌐 Ngrok Integration Guide - Retinal Screening API

## Overview

The Docker image now includes **ngrok** support, allowing you to:
- Create secure public tunnels to your API without complex networking
- Test your API from anywhere with a public HTTPS URL
- Share your local development with team members
- Deploy to environments without public IP addresses
- Get instant HTTPS with valid SSL certificates

---

## 🚀 Quick Start

### Option 1: API Only (Default - Production Mode)
Run without ngrok for direct access on port 8080:

```bash
podman run -p 8080:8080 landwind/retinal-screening-api:latest
# OR
podman run -p 8080:8080 landwind/retinal-screening-api:latest api
```

Access: `http://localhost:8080`

### Option 2: API with Ngrok (Development/Demo Mode)
Run with ngrok to get a public URL:

```bash
podman run -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="your_ngrok_token" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok
```

Access: 
- API: Check container logs for ngrok URL (e.g., `https://abc123.ngrok.io`)
- Ngrok Dashboard: `http://localhost:4040`

### Option 3: Ngrok Only
Run only ngrok (if API is running separately):

```bash
podman run -p 4040:4040 \
  -e NGROK_AUTHTOKEN="your_ngrok_token" \
  landwind/retinal-screening-api:latest ngrok-only
```

---

## 🔑 Getting Your Ngrok Token

### Free Tier (No Credit Card Required)

1. **Sign up** at https://ngrok.com/
2. **Login** to your dashboard
3. **Copy your authtoken** from: https://dashboard.ngrok.com/get-started/your-authtoken
4. **Use the token** in the `-e NGROK_AUTHTOKEN="..."` flag

**Free tier includes:**
- 1 online ngrok process
- 4 tunnels/ngrok process
- 40 connections/minute
- HTTPS tunnels
- Random subdomain (e.g., abc123.ngrok.io)

### Paid Plans (Optional)

For production use, consider paid plans:
- **Personal ($8/mo)**: Custom subdomain, more connections
- **Pro ($20/mo)**: Reserved domains, IP whitelisting
- **Business ($45/mo)**: Multiple users, SSO

---

## 📋 Environment Variables

Configure the container behavior with these variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | API server port |
| `NGROK_AUTHTOKEN` | `""` | Your ngrok authtoken (required for paid features) |
| `NGROK_ENABLED` | `false` | Enable/disable ngrok |
| `NGROK_REGION` | `us` | Ngrok region: `us`, `eu`, `ap`, `au`, `sa`, `jp`, `in` |

---

## 🎯 Use Cases

### Use Case 1: Local Development with Remote Access

**Scenario:** You're developing locally but need to test with a mobile device or share with a colleague.

```bash
# Start with ngrok
podman run -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="your_token" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok

# Get the public URL from logs or visit http://localhost:4040
# Share the URL: https://abc123.ngrok.io
```

### Use Case 2: Demo for Stakeholders

**Scenario:** You need to demo the API to stakeholders without deploying to production.

```bash
# Start with ngrok and custom region
podman run -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="your_token" \
  -e NGROK_ENABLED=true \
  -e NGROK_REGION="eu" \
  landwind/retinal-screening-api:latest api-with-ngrok

# Share the HTTPS URL with stakeholders
# They can access: https://your-tunnel.ngrok.io/docs
```

### Use Case 3: Webhook Testing

**Scenario:** You need to test webhooks from external services (e.g., payment gateways, third-party APIs).

```bash
# Start with ngrok
podman run -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="your_token" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok

# Configure the webhook URL in the external service
# URL: https://abc123.ngrok.io/webhook-endpoint
```

### Use Case 4: Testing on Mobile Devices

**Scenario:** You need to test the API on iOS/Android devices.

```bash
# Start with ngrok
podman run -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="your_token" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok

# Open the ngrok URL on your mobile device
# URL: https://abc123.ngrok.io
# Works over cellular or WiFi from anywhere
```

---

## 🔧 CraneCloud Deployment with Ngrok

### Scenario: Deploy to CraneCloud with Public Tunnel

If CraneCloud doesn't provide a public URL or you need a custom domain:

#### Step 1: Get Ngrok Token
1. Sign up at https://ngrok.com/
2. Copy your authtoken

#### Step 2: Deploy to CraneCloud
Use the same configuration as before, but add environment variables:

```yaml
application:
  name: retinal-screening-api
  
container:
  image: landwind/retinal-screening-api:latest
  port: 8080
  command: ["api-with-ngrok"]  # Enable ngrok mode
  
environment:
  PORT: "8080"
  NGROK_AUTHTOKEN: "your_ngrok_token_here"
  NGROK_ENABLED: "true"
  NGROK_REGION: "us"
  
resources:
  cpu: 2
  memory: 4096
```

#### Step 3: Access Your API
1. Check CraneCloud logs for the ngrok URL
2. Look for a line like: `Public URL: https://abc123.ngrok.io`
3. Access your API at that URL

**Benefits:**
- HTTPS by default
- No DNS configuration needed
- Instant public access
- Valid SSL certificate

---

## 📊 Ngrok Dashboard

When running with ngrok, access the dashboard at `http://localhost:4040`:

**Features:**
- **Request History:** See all HTTP requests
- **Request Inspector:** Inspect headers, body, response
- **Replay Requests:** Replay previous requests for testing
- **Status:** Tunnel status and connection info

---

## 🧪 Testing the Ngrok Integration

### Test Script
Save this as `test-ngrok.sh`:

```bash
#!/bin/bash

echo "Starting API with Ngrok..."
podman run -d -p 8080:8080 -p 4040:4040 \
  --name retinal-api-ngrok \
  -e NGROK_AUTHTOKEN="your_token" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok

echo "Waiting for services to start..."
sleep 10

echo -e "\n=== Local Access ==="
echo "API: http://localhost:8080/health"
curl -s http://localhost:8080/health | python3 -m json.tool

echo -e "\n=== Ngrok Tunnel Info ==="
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; data = json.load(sys.stdin); print(data['tunnels'][0]['public_url'] if data.get('tunnels') else 'Not available yet')")

if [ "$NGROK_URL" != "Not available yet" ]; then
    echo "Public URL: $NGROK_URL"
    echo "Testing public access..."
    curl -s "$NGROK_URL/health" | python3 -m json.tool
    echo -e "\n✓ Public URL is working!"
    echo "  API Docs: $NGROK_URL/docs"
    echo "  Diseases: $NGROK_URL/diseases"
else
    echo "Ngrok not ready yet. Check logs:"
    podman logs retinal-api-ngrok
fi

echo -e "\n=== Ngrok Dashboard ==="
echo "Visit: http://localhost:4040"

echo -e "\n=== Container Logs ==="
podman logs --tail 20 retinal-api-ngrok
```

Run it:
```bash
chmod +x test-ngrok.sh
./test-ngrok.sh
```

---

## 🛠️ Troubleshooting

### Issue 1: Ngrok URL Not Showing

**Symptoms:** Container starts but no ngrok URL in logs

**Solution:**
```bash
# Check container logs
podman logs retinal-api-ngrok

# Access ngrok dashboard
curl http://localhost:4040/api/tunnels

# Ensure NGROK_ENABLED is set
podman inspect retinal-api-ngrok | grep NGROK_ENABLED
```

### Issue 2: "Invalid Authtoken"

**Symptoms:** Ngrok fails with authentication error

**Solution:**
1. Verify your token at https://dashboard.ngrok.com/get-started/your-authtoken
2. Ensure no extra spaces in the token
3. Re-run with correct token:
   ```bash
   podman run -e NGROK_AUTHTOKEN="correct_token" ...
   ```

### Issue 3: Tunnel Connection Limit

**Symptoms:** "Tunnel connection limit reached"

**Solution:**
- **Free tier:** Limit of 1 online process
- Stop other ngrok instances
- Or upgrade to paid plan

### Issue 4: Port Already in Use

**Symptoms:** "Port 4040 already in use"

**Solution:**
```bash
# Stop other ngrok instances
pkill ngrok

# Or use different port
podman run -p 4041:4040 ...
```

### Issue 5: Slow Response Times

**Symptoms:** API responds slowly through ngrok

**Solution:**
- Change region closer to you: `-e NGROK_REGION="eu"` or `"ap"`
- Check ngrok dashboard for connection stats
- Consider upgrading to paid plan for better performance

---

## 🔐 Security Considerations

### For Development/Demo:
- ✅ Use ngrok free tier
- ✅ Temporary public URLs are fine
- ✅ URLs expire when container stops

### For Production:
- ❌ Don't rely on ngrok for production traffic
- ✅ Use proper cloud deployment (CraneCloud, AWS, etc.)
- ✅ Use ngrok only for temporary access
- ✅ If using ngrok in production:
  - Enable IP whitelisting (paid plan)
  - Use basic auth or API keys
  - Monitor traffic via ngrok dashboard
  - Use reserved domains (paid plan)

---

## 📈 Performance Comparison

| Deployment Method | Latency | Setup Time | Cost | Use Case |
|-------------------|---------|------------|------|----------|
| **Direct (No Ngrok)** | ~10ms | 2 min | Free | Production |
| **Ngrok Free** | ~50-100ms | 2 min | Free | Development/Demo |
| **Ngrok Paid** | ~30-50ms | 2 min | $8-45/mo | Temporary Production |
| **CraneCloud Direct** | ~20-40ms | 5 min | ~$20/mo | Production |

---

## 🎓 Advanced Usage

### Custom Subdomain (Paid Plan)
```bash
podman run -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="your_token" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok

# Then configure custom subdomain in ngrok dashboard
```

### Basic Auth Protection
```bash
# Modify ngrok command in supervisord.conf or entrypoint
ngrok http 8080 --auth "username:password"
```

### IP Whitelisting (Paid Plan)
Configure in ngrok dashboard to only allow specific IPs.

### Multiple Tunnels
Run multiple containers with different ports:
```bash
# Container 1: API
podman run -p 8080:8080 -p 4040:4040 ...

# Container 2: Another service
podman run -p 8081:8081 -p 4041:4040 ...
```

---

## 📚 Additional Resources

- **Ngrok Documentation:** https://ngrok.com/docs
- **Ngrok Dashboard:** https://dashboard.ngrok.com
- **Pricing:** https://ngrok.com/pricing
- **Status Page:** https://status.ngrok.com

---

## ✅ Quick Reference

### Start with Ngrok
```bash
podman run -p 8080:8080 -p 4040:4040 \
  -e NGROK_AUTHTOKEN="your_token" \
  -e NGROK_ENABLED=true \
  landwind/retinal-screening-api:latest api-with-ngrok
```

### Get Ngrok URL
```bash
curl http://localhost:4040/api/tunnels | python3 -m json.tool
```

### View Logs
```bash
podman logs -f retinal-api-ngrok
```

### Stop Container
```bash
podman stop retinal-api-ngrok
podman rm retinal-api-ngrok
```

---

**Updated:** October 31, 2025  
**Image Version:** 2.1.0 (with ngrok support)
