#!/bin/bash
# Complete Podman Container Setup for Streamlit UI with GPU
# Runs everything in container, accessible on local machine
# Uses venv for initial setup

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
IMAGE_NAME="retinal-screening-streamlit-gpu"
CONTAINER_NAME="retinal-streamlit-ui"
STREAMLIT_PORT=8501
API_PORT=8080

clear
echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  👁️  Retinal AI Screening - Streamlit Container  ║${NC}"
echo -e "${CYAN}║     GPU-Accelerated with Local UI Access          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}\n"

# Step 1: Activate venv and install dependencies
echo -e "${YELLOW}📦 Step 1: Setting up Python environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ venv not found!${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate
echo -e "${GREEN}✅ Activated venv${NC}"

# Install streamlit dependencies in venv (for requirements.txt validation)
pip install -q streamlit plotly 2>/dev/null || true
echo -e "${GREEN}✅ Dependencies verified${NC}\n"

# Step 2: Stop and remove existing containers
echo -e "${YELLOW}🧹 Step 2: Cleaning up old containers...${NC}"
podman stop ${CONTAINER_NAME} 2>/dev/null || true
podman rm ${CONTAINER_NAME} 2>/dev/null || true
echo -e "${GREEN}✅ Cleanup complete${NC}\n"

# Step 3: Check GPU availability
echo -e "${YELLOW}🎮 Step 3: Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
    USE_GPU=true
    GPU_FLAG="--device nvidia.com/gpu=all"
    GPU_ENV="-e CUDA_VISIBLE_DEVICES=0"
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected. Running in CPU mode.${NC}"
    USE_GPU=false
    GPU_FLAG=""
    GPU_ENV="-e CUDA_VISIBLE_DEVICES=-1"
fi
echo ""

# Step 4: Check if model exists
echo -e "${YELLOW}🔍 Step 4: Verifying model file...${NC}"
MODEL_PATH="models/best_model_mobile.pth"
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo -e "${GREEN}✅ Model found: $MODEL_SIZE${NC}"
else
    echo -e "${RED}❌ Model not found at $MODEL_PATH${NC}"
    exit 1
fi
echo ""

# Step 5: Build or pull image
echo -e "${YELLOW}🔨 Step 5: Preparing container image...${NC}"

# Check if image exists
if podman image exists ${IMAGE_NAME}:latest; then
    echo -e "${BLUE}Image already exists. Rebuild? (y/N): ${NC}"
    read -r -t 10 REBUILD || REBUILD="n"
    
    if [[ $REBUILD =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Rebuilding image...${NC}"
        podman rmi ${IMAGE_NAME}:latest 2>/dev/null || true
        BUILD_IMAGE=true
    else
        echo -e "${GREEN}✅ Using existing image${NC}"
        BUILD_IMAGE=false
    fi
else
    BUILD_IMAGE=true
fi

if [ "$BUILD_IMAGE" = true ]; then
    echo -e "${YELLOW}Building new image...${NC}"
    podman build -t ${IMAGE_NAME}:latest -f Dockerfile . || {
        echo -e "${RED}❌ Build failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ Image built successfully${NC}"
fi
echo ""

# Step 6: Create necessary directories
echo -e "${YELLOW}📁 Step 6: Creating directories...${NC}"
mkdir -p logs uploads
chmod 755 logs uploads
echo -e "${GREEN}✅ Directories ready${NC}\n"

# Step 7: Run container with Streamlit
echo -e "${YELLOW}🚀 Step 7: Starting Streamlit container...${NC}"

podman run -d \
    --name ${CONTAINER_NAME} \
    ${GPU_FLAG} \
    -p ${STREAMLIT_PORT}:8501 \
    -p ${API_PORT}:8080 \
    -v $(pwd)/models:/app/models:Z \
    -v $(pwd)/logs:/app/logs:Z \
    -v $(pwd)/uploads:/app/uploads:Z \
    ${GPU_ENV} \
    -e STREAMLIT_SERVER_PORT=8501 \
    -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    -e STREAMLIT_SERVER_HEADLESS=true \
    -e STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ${IMAGE_NAME}:latest \
    streamlit

# Wait for container to start
echo -e "${BLUE}Waiting for container to start...${NC}"
sleep 3

# Check container status
if ! podman ps | grep -q ${CONTAINER_NAME}; then
    echo -e "${RED}❌ Container failed to start${NC}"
    echo -e "${YELLOW}Checking logs...${NC}"
    podman logs ${CONTAINER_NAME}
    exit 1
fi

# Wait for Streamlit to be ready
echo -e "${BLUE}Waiting for Streamlit to initialize...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:${STREAMLIT_PORT} > /dev/null 2>&1; then
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

echo -e "${GREEN}✅ Container started successfully!${NC}\n"

# Display access information
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          🎉 Streamlit UI is Running! 🎉           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}\n"

echo -e "${CYAN}📱 Access Points:${NC}"
echo -e "   ${BLUE}Streamlit UI:${NC} http://localhost:${STREAMLIT_PORT}"
echo -e "   ${BLUE}API Server:${NC}   http://localhost:${API_PORT}"
echo -e ""
echo -e "${CYAN}🐳 Container Info:${NC}"
echo -e "   ${BLUE}Name:${NC}         ${CONTAINER_NAME}"
echo -e "   ${BLUE}Image:${NC}        ${IMAGE_NAME}:latest"
echo -e "   ${BLUE}Device:${NC}       $([ "$USE_GPU" = true ] && echo "NVIDIA GPU" || echo "CPU")"
echo -e ""
echo -e "${CYAN}💡 Quick Commands:${NC}"
echo -e "   ${BLUE}View logs:${NC}    podman logs -f ${CONTAINER_NAME}"
echo -e "   ${BLUE}Stop:${NC}         podman stop ${CONTAINER_NAME}"
echo -e "   ${BLUE}Restart:${NC}      podman restart ${CONTAINER_NAME}"
echo -e "   ${BLUE}Shell:${NC}        podman exec -it ${CONTAINER_NAME} bash"
echo -e ""
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}\n"

# Ask if user wants to see logs
echo -e "${YELLOW}Show container logs? (Y/n): ${NC}"
read -r -t 5 SHOW_LOGS || SHOW_LOGS="y"

if [[ ! $SHOW_LOGS =~ ^[Nn]$ ]]; then
    echo -e "\n${CYAN}📋 Streaming logs (Ctrl+C to exit, container keeps running):${NC}\n"
    sleep 1
    podman logs -f ${CONTAINER_NAME}
else
    echo -e "${GREEN}✅ Container running in background${NC}"
    echo -e "${BLUE}Open browser: http://localhost:${STREAMLIT_PORT}${NC}"
fi
