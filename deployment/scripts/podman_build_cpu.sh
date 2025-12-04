#!/bin/bash
# ============================================================================
# Podman CPU-Only Build Script
# Builds CPU version of retinal disease API for Crane Cloud deployment
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
IMAGE_NAME="${IMAGE_NAME:-retinal-screening}"
VERSION="${VERSION:-cpu}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================================"
echo "Building CPU-Only Container Image"
echo "============================================================================"

cd "$(dirname "$0")/.."

# Build the image with multiple tags
echo -e "\n${YELLOW}Building CPU image with tags:${NC}"
echo -e "  - ${IMAGE_NAME}:${VERSION}"
echo -e "  - ${IMAGE_NAME}:${TIMESTAMP}"
echo -e "  - ${IMAGE_NAME}:latest-cpu"

podman build \
    --file Dockerfile.cpu \
    --tag ${IMAGE_NAME}:${VERSION} \
    --tag ${IMAGE_NAME}:${TIMESTAMP} \
    --tag ${IMAGE_NAME}:latest-cpu \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ CPU image built successfully${NC}"
    echo -e "Image: ${IMAGE_NAME}"
    echo -e "Tags: ${VERSION}, ${TIMESTAMP}, latest-cpu"
    echo -e "Size: $(podman images ${IMAGE_NAME}:${VERSION} --format '{{.Size}}')"
else
    echo -e "\n${RED}✗ Failed to build CPU image${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Build complete!${NC}"
echo -e "Next step: Run ./deployment/scripts/podman_push_cpu.sh to push to DockerHub"