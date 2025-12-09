#!/bin/bash
# ============================================================================
# Podman CPU-Only Push Script
# Pushes CPU version to DockerHub for Crane Cloud deployment
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

# Automatic versioning (sync with build script)
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

echo "============================================================================"
echo "Pushing CPU-Only Container Images to DockerHub"
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

# Tag images for DockerHub
echo -e "\n${YELLOW}Tagging images for DockerHub...${NC}"
podman tag ${IMAGE_NAME}:latest-cpu ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}
podman tag ${IMAGE_NAME}:latest-cpu ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-cpu

# Push version tag
echo -e "\n${YELLOW}Pushing version tag...${NC}"
podman push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Version tag pushed: docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}${NC}"
else
    echo -e "${RED}✗ Failed to push version tag${NC}"
    exit 1
fi

# Push latest-cpu tag
echo -e "\n${YELLOW}Pushing latest-cpu tag...${NC}"
podman push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-cpu

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Latest-cpu tag pushed: docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-cpu${NC}"
else
    echo -e "${RED}✗ Failed to push latest-cpu tag${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ All CPU images pushed successfully!${NC}"
echo -e "Available tags:"
echo -e "  - docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo -e "  - docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-cpu"

echo -e "\n${GREEN}============================================================================${NC}"
echo -e "${GREEN}✓ CPU deployment ready!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo -e "\nFor Crane Cloud deployment, use:"
echo -e "  Image: docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo -e "  Port: 8080"
echo -e "  Entry Command: /app/start.sh"