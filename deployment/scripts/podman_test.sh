#!/bin/bash
# ============================================================================
# Podman Local Test Script
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
DOCKERHUB_USERNAME="landwind"
IMAGE_NAME="retinal-disease-api"
CONTAINER_NAME="test-api"
PORT=8000

echo "============================================================================"
echo "Testing Container Image Locally with Podman"
echo "============================================================================"

cd "$(dirname "$0")/.."

# Stop and remove existing container if any
if podman ps -a | grep -q $CONTAINER_NAME; then
    echo -e "\n${YELLOW}Stopping existing container...${NC}"
    podman stop $CONTAINER_NAME || true
    podman rm $CONTAINER_NAME || true
fi

# Run container
echo -e "\n${YELLOW}Starting container...${NC}"
podman run -d \
    --name $CONTAINER_NAME \
    -p ${PORT}:8000 \
    -v $(pwd)/models:/app/models:ro \
    ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest

echo -e "${GREEN}✓ Container started${NC}"

# Wait for startup
echo -e "\n${YELLOW}Waiting for API to be ready...${NC}"
sleep 15

# Test health endpoint
echo -e "\n${YELLOW}Testing health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:${PORT}/health)

if echo "$HEALTH_RESPONSE" | grep -q "healthy\|success"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo -e "\nAPI Response:"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo -e "\nContainer logs:"
    podman logs $CONTAINER_NAME
    podman stop $CONTAINER_NAME
    podman rm $CONTAINER_NAME
    exit 1
fi

# Show container info
echo -e "\n${YELLOW}Container information:${NC}"
podman ps | grep $CONTAINER_NAME

echo -e "\n${GREEN}============================================================================${NC}"
echo -e "${GREEN}✓ Container is running successfully!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo -e "\nAPI Endpoints:"
echo -e "  • Health:   http://localhost:${PORT}/health"
echo -e "  • Docs:     http://localhost:${PORT}/docs"
echo -e "  • Predict:  http://localhost:${PORT}/predict"
echo -e "  • Metrics:  http://localhost:${PORT}/metrics"
echo -e "\nView logs:"
echo -e "  podman logs -f $CONTAINER_NAME"
echo -e "\nStop container:"
echo -e "  podman stop $CONTAINER_NAME && podman rm $CONTAINER_NAME"
echo -e "\nRun performance tests:"
echo -e "  python tests/test_gpu_inference.py --url http://localhost:${PORT}"
