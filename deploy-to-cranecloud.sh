#!/bin/bash

# ============================================================================
# CraneCloud Deployment Helper Script
# ============================================================================
# This script prepares deployment information for CraneCloud

set -e

echo "════════════════════════════════════════════════════════════════════════"
echo "  CraneCloud Deployment Helper - Retinal Screening API"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="landwind/retinal-screening-api"
DOCKER_TAG="latest"
APP_PORT="8080"
HEALTH_PATH="/health"

echo "📦 DEPLOYMENT CONFIGURATION"
echo "─────────────────────────────────────────────────────────────────────────"
echo -e "${BLUE}Docker Image:${NC}     ${DOCKER_IMAGE}:${DOCKER_TAG}"
echo -e "${BLUE}Application Port:${NC} ${APP_PORT}"
echo -e "${BLUE}Health Check:${NC}     ${HEALTH_PATH}"
echo ""

# Step 1: Verify Docker image exists on Docker Hub
echo "🔍 Step 1: Verifying Docker Hub Image..."
echo "─────────────────────────────────────────────────────────────────────────"

if curl -s "https://hub.docker.com/v2/repositories/${DOCKER_IMAGE}/tags/${DOCKER_TAG}" | grep -q "name"; then
    echo -e "${GREEN}✓ Image found on Docker Hub${NC}"
    
    # Get image details
    IMAGE_INFO=$(curl -s "https://hub.docker.com/v2/repositories/${DOCKER_IMAGE}/tags/${DOCKER_TAG}")
    IMAGE_SIZE=$(echo "$IMAGE_INFO" | grep -o '"full_size":[0-9]*' | cut -d':' -f2)
    IMAGE_SIZE_GB=$(echo "scale=2; $IMAGE_SIZE / 1024 / 1024 / 1024" | bc)
    LAST_PUSHED=$(echo "$IMAGE_INFO" | grep -o '"last_pushed":"[^"]*"' | cut -d'"' -f4 | cut -d'T' -f1)
    
    echo -e "  ${BLUE}Image Size:${NC}    ${IMAGE_SIZE_GB} GB"
    echo -e "  ${BLUE}Last Pushed:${NC}  ${LAST_PUSHED}"
else
    echo -e "${RED}✗ Image not found on Docker Hub${NC}"
    echo "  Please push the image first: podman push ${DOCKER_IMAGE}:${DOCKER_TAG}"
    exit 1
fi
echo ""

# Step 2: Test local image (if available)
echo "🧪 Step 2: Local Image Verification..."
echo "─────────────────────────────────────────────────────────────────────────"

if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo -e "${YELLOW}⚠ No container runtime found (podman/docker)${NC}"
    echo "  Skipping local verification"
    CONTAINER_CMD=""
fi

if [ -n "$CONTAINER_CMD" ]; then
    if $CONTAINER_CMD images | grep -q "$DOCKER_IMAGE.*$DOCKER_TAG"; then
        echo -e "${GREEN}✓ Image found locally${NC}"
        
        # Get local image info
        IMAGE_ID=$($CONTAINER_CMD images --format "{{.ID}}" $DOCKER_IMAGE:$DOCKER_TAG | head -1)
        IMAGE_CREATED=$($CONTAINER_CMD images --format "{{.CreatedSince}}" $DOCKER_IMAGE:$DOCKER_TAG | head -1)
        IMAGE_SIZE_LOCAL=$($CONTAINER_CMD images --format "{{.Size}}" $DOCKER_IMAGE:$DOCKER_TAG | head -1)
        
        echo -e "  ${BLUE}Image ID:${NC}      ${IMAGE_ID:0:12}"
        echo -e "  ${BLUE}Created:${NC}       ${IMAGE_CREATED}"
        echo -e "  ${BLUE}Size:${NC}          ${IMAGE_SIZE_LOCAL}"
    else
        echo -e "${YELLOW}⚠ Image not found locally${NC}"
        echo "  This is fine - CraneCloud will pull from Docker Hub"
    fi
fi
echo ""

# Step 3: Generate CraneCloud configuration
echo "⚙️  Step 3: Generating CraneCloud Configuration..."
echo "─────────────────────────────────────────────────────────────────────────"

cat > cranecloud-config.yaml << EOF
# CraneCloud Application Configuration
# Copy these values to CraneCloud web dashboard

application:
  name: retinal-screening-api
  description: Multi-model retinal disease screening API with GPU support
  
container:
  image: ${DOCKER_IMAGE}:${DOCKER_TAG}
  port: ${APP_PORT}
  protocol: HTTP
  
resources:
  cpu: 2         # cores (minimum 1, recommended 2-4)
  memory: 4096   # MB (minimum 2048, recommended 4096-8192)
  storage: 20    # GB
  gpu: 1         # Optional: 1x NVIDIA T4 or similar (set to 0 if not available)
  
environment:
  PORT: "${APP_PORT}"
  PYTHONUNBUFFERED: "1"
  
health_check:
  path: ${HEALTH_PATH}
  port: ${APP_PORT}
  initial_delay: 30    # seconds
  period: 10           # seconds
  timeout: 5           # seconds
  
scaling:
  min_replicas: 1
  max_replicas: 3
  
EOF

echo -e "${GREEN}✓ Configuration saved to: cranecloud-config.yaml${NC}"
echo ""

# Step 4: Generate deployment summary
echo "📋 Step 4: CraneCloud Deployment Summary"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "Copy these values to CraneCloud Dashboard:"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}APPLICATION DETAILS${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Name:              ${GREEN}retinal-screening-api${NC}"
echo -e "Description:       Multi-model retinal disease screening API"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}CONTAINER IMAGE${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Image:             ${GREEN}${DOCKER_IMAGE}:${DOCKER_TAG}${NC}"
echo -e "Registry:          Docker Hub (public)"
echo -e "Size:              ${IMAGE_SIZE_GB} GB"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}PORT CONFIGURATION${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Container Port:    ${GREEN}${APP_PORT}${NC}"
echo -e "Protocol:          HTTP"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}RESOURCE ALLOCATION${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "CPU:               ${GREEN}2 cores${NC} (minimum 1)"
echo -e "Memory:            ${GREEN}4096 MB${NC} (4 GB, minimum 2 GB)"
echo -e "Storage:           ${GREEN}20 GB${NC}"
echo -e "GPU:               ${GREEN}1x NVIDIA T4${NC} (optional, recommended)"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}HEALTH CHECK${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Health Path:       ${GREEN}${HEALTH_PATH}${NC}"
echo -e "Health Port:       ${GREEN}${APP_PORT}${NC}"
echo -e "Initial Delay:     ${GREEN}30 seconds${NC}"
echo -e "Period:            ${GREEN}10 seconds${NC}"
echo -e "Timeout:           ${GREEN}5 seconds${NC}"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}ENVIRONMENT VARIABLES (Optional)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "PORT:              ${GREEN}8080${NC}"
echo -e "PYTHONUNBUFFERED:  ${GREEN}1${NC}"
echo ""

# Step 5: Provide deployment instructions
echo ""
echo "🚀 Step 5: Deployment Instructions"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "1. Open CraneCloud Dashboard:"
echo -e "   ${BLUE}https://cranecloud.io${NC}"
echo ""
echo "2. Login with your credentials"
echo ""
echo "3. Click 'Create Application' or '+ New App'"
echo ""
echo "4. Fill in the form using the values above"
echo ""
echo "5. Click 'Deploy' and wait 2-5 minutes"
echo ""
echo "6. Once deployed, test your endpoints:"
echo -e "   ${BLUE}curl https://your-app-url.cranecloud.io/health${NC}"
echo -e "   ${BLUE}curl https://your-app-url.cranecloud.io/diseases${NC}"
echo -e "   ${BLUE}open https://your-app-url.cranecloud.io/docs${NC}"
echo ""

# Step 6: Generate test commands
echo "🧪 Step 6: Test Commands (After Deployment)"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "Save these commands to test your deployed API:"
echo ""

cat > test-cranecloud-deployment.sh << 'TESTEOF'
#!/bin/bash
# Test CraneCloud Deployment
# Replace YOUR_APP_URL with your actual CraneCloud URL

APP_URL="YOUR_APP_URL.cranecloud.io"

echo "Testing CraneCloud Deployment..."
echo ""

echo "1. Health Check:"
curl -s "https://${APP_URL}/health" | python3 -m json.tool
echo ""

echo "2. Get Diseases:"
curl -s "https://${APP_URL}/diseases" | python3 -m json.tool | head -20
echo ""

echo "3. API Documentation:"
echo "   Open in browser: https://${APP_URL}/docs"
echo ""

echo "4. Check Response Time:"
time curl -s "https://${APP_URL}/health" > /dev/null
echo ""

echo "Done!"
TESTEOF

chmod +x test-cranecloud-deployment.sh

echo -e "${GREEN}✓ Test script saved to: test-cranecloud-deployment.sh${NC}"
echo "  Edit the script and replace YOUR_APP_URL with your actual CraneCloud URL"
echo ""

# Step 7: Additional resources
echo "📚 Step 7: Additional Resources"
echo "─────────────────────────────────────────────────────────────────────────"
echo ""
echo "Documentation:"
echo -e "  ${BLUE}• CraneCloud Docs:${NC}         https://docs.cranecloud.io"
echo -e "  ${BLUE}• Detailed Guide:${NC}          CRANECLOUD_DEPLOYMENT_STEPS.md"
echo -e "  ${BLUE}• General Deployment:${NC}      AUTOMATED_DEPLOYMENT_GUIDE.md"
echo ""
echo "Support:"
echo -e "  ${BLUE}• CraneCloud Support:${NC}      support@cranecloud.io"
echo -e "  ${BLUE}• Docker Hub:${NC}              https://hub.docker.com/r/${DOCKER_IMAGE}"
echo ""

# Final summary
echo "════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Pre-deployment verification complete!${NC}"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Next Steps:"
echo "  1. Open https://cranecloud.io and login"
echo "  2. Create new application with the configuration above"
echo "  3. Wait for deployment to complete"
echo "  4. Test endpoints using the test script"
echo ""
echo "Files created:"
echo "  • cranecloud-config.yaml         - Configuration reference"
echo "  • test-cranecloud-deployment.sh  - Test commands"
echo "  • CRANECLOUD_DEPLOYMENT_STEPS.md - Detailed guide"
echo ""
echo -e "${YELLOW}Good luck with your deployment! 🚀${NC}"
echo ""
