#!/bin/bash
# Push retinal screening container to Docker Hub

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Push Container to Docker Hub${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Check if Docker Hub username is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 <dockerhub-username>${NC}"
    echo -e "${YELLOW}Example: $0 myusername${NC}"
    echo
    echo -e "${YELLOW}Or set environment variable:${NC}"
    echo -e "${YELLOW}export DOCKER_USERNAME=myusername${NC}"
    echo -e "${YELLOW}$0${NC}"
    exit 1
fi

DOCKER_USERNAME="$1"
IMAGE_NAME="retinal-screening-gpu"
LOCAL_IMAGE="localhost/${IMAGE_NAME}:latest"
REMOTE_IMAGE="docker.io/${DOCKER_USERNAME}/${IMAGE_NAME}:latest"

echo -e "${GREEN}Step 1: Verify local image exists${NC}"
if ! podman images | grep -q "${IMAGE_NAME}"; then
    echo -e "${RED}✗ Local image not found: ${LOCAL_IMAGE}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Local image found${NC}"
echo

echo -e "${GREEN}Step 2: Login to Docker Hub${NC}"
echo -e "${YELLOW}Please enter your Docker Hub credentials:${NC}"
if ! podman login docker.io; then
    echo -e "${RED}✗ Docker Hub login failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Successfully logged in to Docker Hub${NC}"
echo

echo -e "${GREEN}Step 3: Tag image for Docker Hub${NC}"
echo "Tagging: ${LOCAL_IMAGE} -> ${REMOTE_IMAGE}"
if ! podman tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"; then
    echo -e "${RED}✗ Failed to tag image${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Image tagged successfully${NC}"
echo

echo -e "${GREEN}Step 4: Push image to Docker Hub${NC}"
echo -e "${YELLOW}This may take 10-15 minutes (11.8 GB image)...${NC}"
if ! podman push "${REMOTE_IMAGE}"; then
    echo -e "${RED}✗ Failed to push image${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Image pushed successfully!${NC}"
echo

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Docker Hub Push Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "Image URL: ${GREEN}https://hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}${NC}"
echo -e "Pull command: ${YELLOW}docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:latest${NC}"
echo
echo -e "${GREEN}Next step: Deploy to GCP${NC}"
echo -e "Run: ${YELLOW}./deploy_to_gcp.sh ${DOCKER_USERNAME}${NC}"
