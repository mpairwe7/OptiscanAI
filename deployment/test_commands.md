# Container Test Commands

## After build completes, run these commands:

### 1. Verify image was created
```bash
source venv/bin/activate
podman images | grep retinal
```

### 2. Run container locally (CPU only - no GPU required for testing)
```bash
podman run -d \
  --name retinal-test \
  -p 8000:8000 \
  -e MODEL_PATH=/app/models/exports/config.json \
  localhost/retinal-screening-gpu:latest
```

### 3. Check logs
```bash
podman logs -f retinal-test
```

### 4. Test API endpoints
```bash
# Health check
curl http://localhost:8000/health

# API info
curl http://localhost:8000/api/v1/info

# Open docs in browser
xdg-open http://localhost:8000/docs
```

### 5. Stop container
```bash
podman stop retinal-test
podman rm retinal-test
```

## Next: Push to Docker Hub

Once local testing passes, we'll:
1. Set DOCKER_USERNAME environment variable
2. Login to Docker Hub
3. Tag the image
4. Push to registry
5. Deploy to GCP

---

**Note**: Build is currently in progress. Wait for completion before running these commands.
