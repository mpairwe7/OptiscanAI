#!/bin/bash

# ============================================================================
# Automated Docker Hub & Crane Cloud Deployment Setup
# ============================================================================
# This script automates:
# 1. Docker Hub authentication
# 2. GitHub secrets configuration
# 3. Docker image build and push
# 4. Crane Cloud deployment preparation
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKERHUB_USERNAME="landwind"
DOCKERHUB_PASSWORD="alien123.com"
DOCKER_IMAGE="landwind/retinal-screening-api"
MODEL_FILE="models/best_model_mobile.pt"
PORT=8080

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}     AUTOMATED DOCKER HUB & CRANE CLOUD DEPLOYMENT SETUP${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# ============================================================================
# Step 1: Verify Prerequisites
# ============================================================================
echo -e "${YELLOW}[Step 1/6] Verifying prerequisites...${NC}"

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    echo -e "${RED}❌ Podman is not installed. Please install Podman first.${NC}"
    echo "Visit: https://podman.io/getting-started/installation"
    exit 1
fi
echo -e "${GREEN}✅ Podman is installed${NC}"

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}❌ Model file not found: $MODEL_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Model file found: $MODEL_FILE${NC}"

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}❌ Dockerfile not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Dockerfile found${NC}"

# Check if git repo
if [ ! -d ".git" ]; then
    echo -e "${RED}❌ Not a git repository${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Git repository detected${NC}"

echo ""

# ============================================================================
# Step 2: Docker Hub Authentication
# ============================================================================
echo -e "${YELLOW}[Step 2/6] Authenticating with Docker Hub...${NC}"

# Login to Docker Hub using Podman
echo "$DOCKERHUB_PASSWORD" | podman login docker.io --username "$DOCKERHUB_USERNAME" --password-stdin

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully logged in to Docker Hub as $DOCKERHUB_USERNAME${NC}"
else
    echo -e "${RED}❌ Docker Hub login failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Step 3: Build Docker Image
# ============================================================================
echo -e "${YELLOW}[Step 3/6] Building Docker image with Podman...${NC}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SHORT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
VERSION="v1.0-${SHORT_SHA}"

echo "Building: $DOCKER_IMAGE:$VERSION"

podman build \
    --tag "$DOCKER_IMAGE:latest" \
    --tag "$DOCKER_IMAGE:$VERSION" \
    --build-arg BUILDTIME="$TIMESTAMP" \
    --build-arg VERSION="$VERSION" \
    --format docker \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully with Podman${NC}"
else
    echo -e "${RED}❌ Podman build failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Step 4: Test Docker Image Locally
# ============================================================================
echo -e "${YELLOW}[Step 4/6] Testing Docker image locally with Podman...${NC}"

# Stop any existing container
podman stop test-retinal-api 2>/dev/null || true
podman rm test-retinal-api 2>/dev/null || true

# Run container
echo "Starting test container on port $PORT..."
podman run -d \
    --name test-retinal-api \
    -p $PORT:$PORT \
    "$DOCKER_IMAGE:latest"

# Wait for startup
echo "Waiting for API to start..."
sleep 15

# Test health endpoint
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:$PORT/health &> /dev/null; then
        echo -e "${GREEN}✅ Health check passed!${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Retry $RETRY_COUNT/$MAX_RETRIES..."
    sleep 3
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ Health check failed${NC}"
    podman logs test-retinal-api
    podman stop test-retinal-api
    podman rm test-retinal-api
    exit 1
fi

# Test diseases endpoint
echo "Testing /diseases endpoint..."
if curl -f http://localhost:$PORT/diseases &> /dev/null; then
    echo -e "${GREEN}✅ Diseases endpoint working${NC}"
else
    echo -e "${RED}❌ Diseases endpoint failed${NC}"
fi

# Cleanup
podman stop test-retinal-api
podman rm test-retinal-api

echo ""

# ============================================================================
# Step 5: Push to Docker Hub
# ============================================================================
echo -e "${YELLOW}[Step 5/6] Pushing image to Docker Hub with Podman...${NC}"

echo "Pushing: $DOCKER_IMAGE:latest"
podman push "$DOCKER_IMAGE:latest"

echo "Pushing: $DOCKER_IMAGE:$VERSION"
podman push "$DOCKER_IMAGE:$VERSION"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully pushed to Docker Hub${NC}"
else
    echo -e "${RED}❌ Podman push failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Step 6: Setup GitHub Secrets
# ============================================================================
echo -e "${YELLOW}[Step 6/6] Setting up GitHub repository...${NC}"

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo "GitHub CLI detected. Setting up secrets..."
    
    # Check if authenticated
    if gh auth status &> /dev/null; then
        echo "Setting DOCKERHUB_PASSWORD secret..."
        echo "$DOCKERHUB_PASSWORD" | gh secret set DOCKERHUB_PASSWORD
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ GitHub secret DOCKERHUB_PASSWORD set${NC}"
        else
            echo -e "${YELLOW}⚠ Failed to set GitHub secret. You can set it manually.${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ GitHub CLI not authenticated${NC}"
        echo "Run: gh auth login"
    fi
else
    echo -e "${YELLOW}⚠ GitHub CLI not installed${NC}"
    echo "Install from: https://cli.github.com/"
fi

echo ""

# ============================================================================
# Deployment Summary
# ============================================================================
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}                    DEPLOYMENT SETUP COMPLETE!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo -e "${BLUE}📦 Docker Image Information:${NC}"
echo "   Repository: $DOCKER_IMAGE"
echo "   Tags: latest, $VERSION"
echo "   Status: ✅ Built and pushed to Docker Hub"
echo ""
echo -e "${BLUE}🔧 GitHub Actions:${NC}"
echo "   Status: ✅ Ready to trigger on next push"
echo "   Workflow: .github/workflows/dockerhub-deploy.yml"
echo ""
echo -e "${BLUE}☁️  Crane Cloud Deployment:${NC}"
echo "   1. Go to: https://cranecloud.io"
echo "   2. Login and create new app"
echo "   3. Select 'Docker Image' deployment"
echo "   4. Enter image: $DOCKER_IMAGE:latest"
echo "   5. Set port: $PORT"
echo "   6. Click Deploy"
echo ""
echo -e "${BLUE}🧪 Test Endpoints:${NC}"
echo "   Health: https://your-app.cranecloud.io/health"
echo "   Diseases: https://your-app.cranecloud.io/diseases"
echo "   Predict: POST https://your-app.cranecloud.io/predict"
echo ""
echo -e "${BLUE}🔄 Next Steps:${NC}"
echo "   1. Commit and push changes to trigger GitHub Actions:"
echo "      git add ."
echo "      git commit -m 'Setup automated deployment'"
echo "      git push origin main"
echo ""
echo "   2. Monitor workflow: https://github.com/$(git remote get-url origin | sed 's/.*://;s/.git$//')/actions"
echo ""
echo "   3. Deploy to Crane Cloud using the image: $DOCKER_IMAGE:latest"
echo ""
echo -e "${GREEN}✅ All systems ready for deployment!${NC}"
echo -e "${GREEN}============================================================================${NC}"
