#!/bin/bash
# ============================================================================
# Podman Build Script with DockerHub Credentials
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# DockerHub credentials
DOCKERHUB_USERNAME="landwind"
IMAGE_NAME="retinal-disease-api"

echo "============================================================================"
echo "Building Container Image with Podman"
echo "============================================================================"

cd "$(dirname "$0")/.."

echo -e "\n${YELLOW}Building CPU image...${NC}"
podman build -f docker/Dockerfile.cpu -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest .

echo -e "\n${GREEN}✓ Image built: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest${NC}"

echo -e "\n${YELLOW}Building GPU image...${NC}"
podman build -f docker/Dockerfile.gpu -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu .

echo -e "\n${GREEN}✓ Image built: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu${NC}"

# Show images
echo -e "\n${YELLOW}Available images:${NC}"
podman images | grep ${IMAGE_NAME}

echo -e "\n${GREEN}✓ Build complete!${NC}"
echo "Next steps:"
echo "  1. Test locally: ./scripts/podman_test.sh"
echo "  2. Push to DockerHub: ./scripts/podman_push.sh"
