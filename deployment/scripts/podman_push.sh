#!/bin/bash
# ============================================================================
# Podman Push Script with Automated DockerHub Login
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# DockerHub credentials (pre-configured)
DOCKERHUB_USERNAME="landwind"
DOCKERHUB_PASSWORD="alien123.com"
IMAGE_NAME="retinal-disease-api"

echo "============================================================================"
echo "Pushing Container Images to DockerHub"
echo "============================================================================"

cd "$(dirname "$0")/.."

# Login to DockerHub
echo -e "\n${YELLOW}Logging in to DockerHub...${NC}"
echo "$DOCKERHUB_PASSWORD" | podman login docker.io -u "$DOCKERHUB_USERNAME" --password-stdin

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ DockerHub login successful${NC}"
else
    echo -e "${RED}✗ DockerHub login failed${NC}"
    exit 1
fi

# Push CPU image
echo -e "\n${YELLOW}Pushing CPU image...${NC}"
podman push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CPU image pushed: docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest${NC}"
else
    echo -e "${RED}✗ Failed to push CPU image${NC}"
    exit 1
fi

# Push GPU image
echo -e "\n${YELLOW}Pushing GPU image...${NC}"
podman push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GPU image pushed: docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu${NC}"
else
    echo -e "${RED}✗ Failed to push GPU image${NC}"
    exit 1
fi

echo -e "\n${GREEN}============================================================================${NC}"
echo -e "${GREEN}✓ All images pushed successfully!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo -e "\nImages available at:"
echo -e "  • https://hub.docker.com/r/${DOCKERHUB_USERNAME}/${IMAGE_NAME}"
echo -e "\nNext step:"
echo -e "  Deploy to GCP: ./scripts/gcp_deploy.sh"
