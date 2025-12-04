#!/bin/bash
# ============================================================================
# Podman Build Script with DockerHub Credentials
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# DockerHub configuration
DOCKERHUB_USERNAME="landwind"
IMAGE_NAME="${IMAGE_NAME:-retinal-screening}"
VERSION="${VERSION:-gpu-v2.1.0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================================"
echo "Building Container Image with Podman"
echo "============================================================================"

cd "$(dirname "$0")/../.."

echo -e "\n${YELLOW}Building GPU image...${NC}"
podman build -f Dockerfile -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION} -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TIMESTAMP} -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu .

echo -e "\n${GREEN}✓ Image built with tags:${NC}"
echo -e "  - ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo -e "  - ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TIMESTAMP}"
echo -e "  - ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-gpu"

# Show images
echo -e "\n${YELLOW}Available images:${NC}"
podman images | grep ${IMAGE_NAME}

echo -e "\n${GREEN}✓ Build complete!${NC}"
echo "Next steps:"
echo "  1. Test locally: ./scripts/podman_test.sh"
echo "  2. Push to DockerHub: ./scripts/podman_push.sh"
