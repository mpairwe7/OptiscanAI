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
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-landwind}"
IMAGE_NAME="${IMAGE_NAME:-retinal-screening}"

# Automatic versioning
if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
    # Get git information
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    GIT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

    # Use git tag if available, otherwise use commit hash
    if [ -n "$GIT_TAG" ]; then
        VERSION="${VERSION:-cpu-${GIT_TAG}}"
    else
        VERSION="${VERSION:-cpu-v2.1.0-${GIT_COMMIT}}"
    fi
else
    VERSION="${VERSION:-cpu-v2.1.0}"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================================"
echo "Building CPU-Only Container Image"
echo "============================================================================"

cd "$(dirname "$0")/.."

# Build the image with multiple tags
echo -e "\n${YELLOW}Building CPU image with tags:${NC}"
echo -e "  - ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo -e "  - ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TIMESTAMP}"
echo -e "  - ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-cpu"

podman build \
    --file Dockerfile.cpu \
    --tag ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION} \
    --tag ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TIMESTAMP} \
    --tag ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-cpu \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ CPU image built successfully${NC}"
    echo -e "Image: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}"
    echo -e "Tags: ${VERSION}, ${TIMESTAMP}, latest-cpu"
    echo -e "Size: $(podman images ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION} --format '{{.Size}}')"
else
    echo -e "\n${RED}✗ Failed to build CPU image${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Build complete!${NC}"
echo -e "Next step: Run ./deployment/scripts/podman_push_cpu.sh to push to DockerHub"