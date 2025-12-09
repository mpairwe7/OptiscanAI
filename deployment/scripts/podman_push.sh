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

# DockerHub credentials (use environment variables for security)
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-landwind}"
DOCKERHUB_PASSWORD="${DOCKERHUB_PASSWORD:-alien123.com}"

# Check if credentials are set
if [ -z "$DOCKERHUB_USERNAME" ] || [ -z "$DOCKERHUB_PASSWORD" ]; then
    echo -e "${RED}Error: DockerHub credentials not set!${NC}"
    echo -e "Please set the following environment variables:"
    echo -e "  export DOCKERHUB_USERNAME=your_username"
    echo -e "  export DOCKERHUB_PASSWORD=your_password"
    echo -e "\nOr run with:"
    echo -e "  DOCKERHUB_USERNAME=your_username DOCKERHUB_PASSWORD=your_password $0"
    exit 1
fi

# Warn about default credentials
if [ "$DOCKERHUB_USERNAME" = "landwind" ] && [ "$DOCKERHUB_PASSWORD" = "alien123.com" ]; then
    echo -e "${YELLOW}⚠ Warning: Using default credentials. Consider setting custom environment variables.${NC}"
fi
IMAGE_NAME="${IMAGE_NAME:-retinal-screening}"
VERSION="${VERSION:-gpu-v2.1.0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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

# Push version tag
echo -e "\n${YELLOW}Pushing version tag...${NC}"
podman push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION} docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Version tag pushed: docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}${NC}"
else
    echo -e "${RED}✗ Failed to push version tag${NC}"
    exit 1
fi

# Push latest-gpu tag
echo -e "\n${YELLOW}Pushing latest-gpu tag...${NC}"
podman push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Latest-gpu tag pushed: docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu${NC}"
else
    echo -e "${RED}✗ Failed to push latest-gpu tag${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ All images pushed successfully!${NC}"
echo -e "Available tags:"
echo -e "  - docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo -e "  - docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu"

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
