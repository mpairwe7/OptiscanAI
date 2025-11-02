#!/bin/bash
# GPU-Accelerated Streamlit Retinal Screening - Podman Development Script
# This script uses existing venv and builds/runs with Podman + NVIDIA GPU support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="retinal-screening-streamlit"
CONTAINER_NAME="retinal-screening-dev"
STREAMLIT_PORT=8501
API_PORT=8080

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Retinal AI Screening - Dev Setup${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Step 1: Clean up old images and containers
echo -e "${YELLOW}📦 Step 1: Cleaning up old images...${NC}"
podman stop ${CONTAINER_NAME} 2>/dev/null || true
podman rm ${CONTAINER_NAME} 2>/dev/null || true
podman rmi ${IMAGE_NAME}:latest 2>/dev/null || true
podman image prune -f
echo -e "${GREEN}✅ Cleanup complete${NC}\n"

# Step 2: Check venv and install streamlit dependencies
echo -e "${YELLOW}📦 Step 2: Setting up Python environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ venv not found! Please create it first.${NC}"
    exit 1
fi

source venv/bin/activate
echo -e "${GREEN}✅ Activated venv${NC}"

# Install streamlit and dependencies
pip install -q streamlit streamlit-drawable-canvas streamlit-aggrid plotly
echo -e "${GREEN}✅ Streamlit dependencies installed${NC}\n"

# Step 3: Check GPU availability
echo -e "${YELLOW}🎮 Step 3: Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    USE_GPU=true
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected. Running in CPU mode.${NC}"
    USE_GPU=false
fi
echo ""

# Step 4: Build Podman image
echo -e "${YELLOW}🔨 Step 4: Building Podman image...${NC}"
if [ "$USE_GPU" = true ]; then
    podman build -t ${IMAGE_NAME}:latest \
        --device nvidia.com/gpu=all \
        -f Dockerfile .
else
    podman build -t ${IMAGE_NAME}:latest -f Dockerfile .
fi
echo -e "${GREEN}✅ Image built successfully${NC}\n"

# Step 5: Run container
echo -e "${YELLOW}🚀 Step 5: Starting container...${NC}"

if [ "$USE_GPU" = true ]; then
    echo -e "${BLUE}Starting with GPU support...${NC}"
    podman run -d \
        --name ${CONTAINER_NAME} \
        --device nvidia.com/gpu=all \
        -p ${STREAMLIT_PORT}:8501 \
        -p ${API_PORT}:8080 \
        -v $(pwd)/models:/app/models:Z \
        -v $(pwd)/logs:/app/logs:Z \
        -v $(pwd)/src:/app/src:Z \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e STREAMLIT_SERVER_PORT=8501 \
        -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
        ${IMAGE_NAME}:latest \
        streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
else
    echo -e "${BLUE}Starting in CPU mode...${NC}"
    podman run -d \
        --name ${CONTAINER_NAME} \
        -p ${STREAMLIT_PORT}:8501 \
        -p ${API_PORT}:8080 \
        -v $(pwd)/models:/app/models:Z \
        -v $(pwd)/logs:/app/logs:Z \
        -v $(pwd)/src:/app/src:Z \
        -e CUDA_VISIBLE_DEVICES=-1 \
        -e STREAMLIT_SERVER_PORT=8501 \
        -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
        ${IMAGE_NAME}:latest \
        streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
fi

# Wait for container to start
echo -e "${BLUE}Waiting for container to start...${NC}"
sleep 5

# Check if container is running
if podman ps | grep -q ${CONTAINER_NAME}; then
    echo -e "${GREEN}✅ Container started successfully!${NC}\n"
    
    # Display access information
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🎉 Streamlit App Running!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${BLUE}📱 Streamlit UI:${NC} http://localhost:${STREAMLIT_PORT}"
    echo -e "${BLUE}🔌 API Server:${NC}   http://localhost:${API_PORT}"
    echo -e "${BLUE}🐳 Container:${NC}    ${CONTAINER_NAME}"
    echo -e "${GREEN}========================================${NC}\n"
    
    # Show container logs
    echo -e "${YELLOW}📋 Container logs (Ctrl+C to exit):${NC}"
    podman logs -f ${CONTAINER_NAME}
else
    echo -e "${RED}❌ Container failed to start${NC}"
    echo -e "${YELLOW}Checking logs...${NC}"
    podman logs ${CONTAINER_NAME}
    exit 1
fi
