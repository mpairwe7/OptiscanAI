#!/bin/bash
# Deploy Retinal Screening API to Crane Cloud

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_USERNAME="${1:-landwind}"
IMAGE_NAME="retinal-screening-gpu"
DOCKER_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Crane Cloud Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Step 1: Check prerequisites
echo -e "${BLUE}[1/6] Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found. Please install Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

if ! command -v git &> /dev/null; then
    echo -e "${RED}✗ Git not found. Please install Git.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Git found${NC}"

# Step 2: Build Docker image
echo
echo -e "${BLUE}[2/6] Building Docker image...${NC}"
echo -e "${YELLOW}This may take 5-10 minutes...${NC}"

docker build -t ${DOCKER_IMAGE}:latest . || {
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Docker image built successfully${NC}"

# Step 3: Test image locally
echo
echo -e "${BLUE}[3/6] Testing image locally...${NC}"

# Check if nvidia-docker is available
if command -v nvidia-docker &> /dev/null || docker info | grep -q "Runtimes.*nvidia"; then
    echo -e "${GREEN}✓ NVIDIA Docker runtime detected${NC}"
    GPU_FLAG="--gpus all"
else
    echo -e "${YELLOW}⚠ NVIDIA Docker runtime not found. Testing without GPU.${NC}"
    GPU_FLAG=""
fi

# Start test container
docker run -d --name retinal-test ${GPU_FLAG} -p 8080:8080 ${DOCKER_IMAGE}:latest

echo "Waiting for container to start..."
sleep 15

# Test health endpoint
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8080/health &> /dev/null; then
        echo -e "${GREEN}✓ Health check passed!${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Retry $RETRY_COUNT/$MAX_RETRIES..."
    sleep 3
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}✗ Health check failed${NC}"
    docker logs retinal-test
    docker stop retinal-test
    docker rm retinal-test
    exit 1
fi

# Cleanup test container
docker stop retinal-test
docker rm retinal-test
echo -e "${GREEN}✓ Local test passed${NC}"

# Step 4: Tag image
echo
echo -e "${BLUE}[4/6] Tagging image...${NC}"

# Get version from git
if git rev-parse --git-dir > /dev/null 2>&1; then
    SHORT_SHA=$(git rev-parse --short HEAD)
    VERSION="v2.0-cuda-${SHORT_SHA}"
    docker tag ${DOCKER_IMAGE}:latest ${DOCKER_IMAGE}:${VERSION}
    docker tag ${DOCKER_IMAGE}:latest ${DOCKER_IMAGE}:cuda-11.8
    echo -e "${GREEN}✓ Tagged: latest, ${VERSION}, cuda-11.8${NC}"
else
    VERSION="latest"
    echo -e "${YELLOW}⚠ Not a git repository. Using 'latest' tag only.${NC}"
fi

# Step 5: Push to Docker Hub
echo
echo -e "${BLUE}[5/6] Pushing to Docker Hub...${NC}"
echo -e "${YELLOW}Please ensure you're logged in to Docker Hub${NC}"
echo -e "${YELLOW}Run: docker login${NC}"
echo

read -p "Continue with push? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Push cancelled. You can push manually later:${NC}"
    echo "docker push ${DOCKER_IMAGE}:latest"
    echo "docker push ${DOCKER_IMAGE}:${VERSION}"
    exit 0
fi

docker push ${DOCKER_IMAGE}:latest || {
    echo -e "${RED}✗ Push failed. Please check your Docker Hub credentials.${NC}"
    echo "Run: docker login"
    exit 1
}

if [ "$VERSION" != "latest" ]; then
    docker push ${DOCKER_IMAGE}:${VERSION}
    docker push ${DOCKER_IMAGE}:cuda-11.8
fi

echo -e "${GREEN}✓ Images pushed successfully${NC}"

# Step 6: Generate deployment instructions
echo
echo -e "${BLUE}[6/6] Deployment Instructions${NC}"
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Crane Cloud Deployment Steps${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${YELLOW}1. Log in to Crane Cloud:${NC}"
echo "   https://cranecloud.io"
echo
echo -e "${YELLOW}2. Create New App:${NC}"
echo "   • Click 'Create New App'"
echo "   • Name: retinal-disease-api"
echo "   • Select Docker deployment"
echo
echo -e "${YELLOW}3. Configure Docker Image:${NC}"
echo "   • Image: ${DOCKER_IMAGE}:${VERSION}"
echo "   • Port: 8080"
echo "   • Protocol: HTTP"
echo
echo -e "${YELLOW}4. Set Resources:${NC}"
echo "   • CPU: 2-4 cores"
echo "   • Memory: 4-8 GB"
echo "   • GPU: 1x NVIDIA (if available)"
echo
echo -e "${YELLOW}5. Environment Variables (Optional):${NC}"
echo "   MODEL_PATH=/app/models/exports/best_model.pth"
echo "   LOG_LEVEL=INFO"
echo
echo -e "${YELLOW}6. Deploy and Test:${NC}"
echo "   curl https://your-app.cranecloud.io/health"
echo "   curl https://your-app.cranecloud.io/diseases"
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Image Information${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo "Docker Image: ${DOCKER_IMAGE}:${VERSION}"
echo "Also tagged as: ${DOCKER_IMAGE}:latest"
echo "CUDA Version: 11.8.0"
echo "cuDNN Version: 8"
echo "Base: nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Quick Commands${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo "Pull image:"
echo "  docker pull ${DOCKER_IMAGE}:${VERSION}"
echo
echo "Run locally with GPU:"
echo "  docker run --gpus all -p 8080:8080 ${DOCKER_IMAGE}:${VERSION}"
echo
echo "Run locally without GPU:"
echo "  docker run -p 8080:8080 ${DOCKER_IMAGE}:${VERSION}"
echo
echo "View logs:"
echo "  docker logs -f <container-id>"
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete! 🚀${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo "For detailed documentation, see:"
echo "  - CRANECLOUD_DEPLOYMENT.md"
echo "  - README.md"
echo
echo "Need help? Contact Crane Cloud support:"
echo "  https://docs.cranecloud.io"
echo
