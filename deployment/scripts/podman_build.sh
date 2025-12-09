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
        VERSION="${VERSION:-gpu-${GIT_TAG}}"
    else
        VERSION="${VERSION:-gpu-v2.1.0-${GIT_COMMIT}}"
    fi
else
    VERSION="${VERSION:-gpu-v2.1.0}"
fi

echo "============================================================================"
echo "Building Container Image with Podman"
echo "============================================================================"

cd "$(dirname "$0")/../.."

echo -e "\n${YELLOW}Building GPU image...${NC}"
podman build -f Dockerfile -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION} .

echo -e "\n${GREEN} Image built with tag:${NC}"
echo -e "  - ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"

# Show images
echo -e "\n${YELLOW}Available images:${NC}"
podman images | grep ${IMAGE_NAME}

echo -e "\n${GREEN} Build complete!${NC}"
echo "Next steps:"
echo "  1. Test locally: ./scripts/podman_test.sh"
echo "  2. Push to DockerHub: ./scripts/podman_push.sh"
