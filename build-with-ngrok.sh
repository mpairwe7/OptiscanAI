#!/bin/bash
# ============================================================================
# Build and Test Docker Image with Ngrok Support
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Building Retinal Screening API with Ngrok Support${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Configuration
IMAGE_NAME="landwind/retinal-screening-api"
IMAGE_TAG="v2.1-ngrok"
CONTAINER_NAME="retinal-api-test"

# Detect container runtime
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo -e "${RED}Error: Neither podman nor docker found${NC}"
    exit 1
fi

echo -e "${BLUE}Using container runtime:${NC} $CONTAINER_CMD"
echo ""

# Step 1: Build the image
echo -e "${GREEN}[Step 1/5] Building Docker image...${NC}"
$CONTAINER_CMD build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image built successfully${NC}"
else
    echo -e "${RED}✗ Image build failed${NC}"
    exit 1
fi
echo ""

# Step 2: Test API-only mode
echo -e "${GREEN}[Step 2/5] Testing API-only mode...${NC}"
$CONTAINER_CMD run -d --name ${CONTAINER_NAME}-api \
    -p 8080:8080 \
    ${IMAGE_NAME}:${IMAGE_TAG} api

echo "Waiting for API to start..."
sleep 10

echo "Testing health endpoint..."
if curl -s http://localhost:8080/health | grep -q "status"; then
    echo -e "${GREEN}✓ API-only mode working${NC}"
else
    echo -e "${RED}✗ API health check failed${NC}"
    $CONTAINER_CMD logs ${CONTAINER_NAME}-api
fi

$CONTAINER_CMD stop ${CONTAINER_NAME}-api
$CONTAINER_CMD rm ${CONTAINER_NAME}-api
echo ""

# Step 3: Test API-with-ngrok mode (requires authtoken)
echo -e "${GREEN}[Step 3/5] Testing API-with-ngrok mode...${NC}"
echo -e "${YELLOW}Note: This requires NGROK_AUTHTOKEN environment variable${NC}"

if [ -n "$NGROK_AUTHTOKEN" ]; then
    echo "Starting with ngrok..."
    $CONTAINER_CMD run -d --name ${CONTAINER_NAME}-ngrok \
        -p 8080:8080 -p 4040:4040 \
        -e NGROK_AUTHTOKEN="$NGROK_AUTHTOKEN" \
        -e NGROK_ENABLED=true \
        ${IMAGE_NAME}:${IMAGE_TAG} api-with-ngrok
    
    echo "Waiting for services to start..."
    sleep 15
    
    echo "Checking API..."
    curl -s http://localhost:8080/health | python3 -m json.tool
    
    echo -e "\nChecking ngrok tunnel..."
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | \
        python3 -c "import sys, json; data = json.load(sys.stdin); print(data['tunnels'][0]['public_url'] if data.get('tunnels') else 'Not ready')" || echo "Not ready")
    
    if [ "$NGROK_URL" != "Not ready" ]; then
        echo -e "${GREEN}✓ Ngrok tunnel established${NC}"
        echo -e "${GREEN}Public URL:${NC} $NGROK_URL"
        echo -e "${GREEN}Dashboard:${NC} http://localhost:4040"
    else
        echo -e "${YELLOW}⚠ Ngrok tunnel not ready yet (check logs)${NC}"
    fi
    
    $CONTAINER_CMD stop ${CONTAINER_NAME}-ngrok
    $CONTAINER_CMD rm ${CONTAINER_NAME}-ngrok
else
    echo -e "${YELLOW}⚠ NGROK_AUTHTOKEN not set, skipping ngrok test${NC}"
    echo "  To test ngrok, set NGROK_AUTHTOKEN environment variable:"
    echo "  export NGROK_AUTHTOKEN='your_token'"
    echo "  Then run this script again"
fi
echo ""

# Step 4: Show image info
echo -e "${GREEN}[Step 4/5] Image Information${NC}"
echo "─────────────────────────────────────────────────────────────"
$CONTAINER_CMD images ${IMAGE_NAME}
echo ""

# Step 5: Provide next steps
echo -e "${GREEN}[Step 5/5] Next Steps${NC}"
echo "─────────────────────────────────────────────────────────────"
echo ""
echo "✅ Build Complete!"
echo ""
echo "📦 Images created:"
echo "   • ${IMAGE_NAME}:${IMAGE_TAG}"
echo "   • ${IMAGE_NAME}:latest"
echo ""
echo "🚀 Run modes:"
echo ""
echo "1. API only (production):"
echo "   $CONTAINER_CMD run -p 8080:8080 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "2. API with ngrok (development/demo):"
echo "   $CONTAINER_CMD run -p 8080:8080 -p 4040:4040 \\"
echo "     -e NGROK_AUTHTOKEN='your_token' \\"
echo "     -e NGROK_ENABLED=true \\"
echo "     ${IMAGE_NAME}:${IMAGE_TAG} api-with-ngrok"
echo ""
echo "3. Push to Docker Hub:"
echo "   $CONTAINER_CMD push ${IMAGE_NAME}:${IMAGE_TAG}"
echo "   $CONTAINER_CMD push ${IMAGE_NAME}:latest"
echo ""
echo "📚 Documentation:"
echo "   • NGROK_INTEGRATION_GUIDE.md - Complete ngrok guide"
echo "   • CRANECLOUD_DEPLOYMENT_STEPS.md - CraneCloud deployment"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Build completed successfully! 🎉${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
