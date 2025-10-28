#!/bin/bash

# Quick Start Script for Local Deployment Testing
# This script helps you test the deployment locally before pushing to production

set -e  # Exit on error

echo "=================================================="
echo "🚀 Multi-Retinal Disease Model - Local Deployment"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker installed${NC}"

# Check if model exists
if [ ! -f "models/exports/best_model.pth" ]; then
    echo -e "${YELLOW}⚠️  Model not found at models/exports/best_model.pth${NC}"
    echo "   Demo mode will be enabled (random predictions)"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ Model found${NC}"
fi

echo ""
echo "🔨 Building Docker image..."
docker build -t retinal-disease-model:local .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

echo ""
echo "🚀 Starting container..."
docker run -d \
    --name retinal-disease-api \
    -p 8080:8080 \
    retinal-disease-model:local

# Wait for container to start
sleep 3

echo ""
echo "🧪 Running health check..."
sleep 2  # Give API time to initialize

if curl -s http://localhost:8080/health > /dev/null; then
    echo -e "${GREEN}✅ API is healthy!${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    echo "Container logs:"
    docker logs retinal-disease-api
    exit 1
fi

echo ""
echo "=================================================="
echo "🎉 Deployment successful!"
echo "=================================================="
echo ""
echo "API is running at: http://localhost:8080"
echo ""
echo "📚 Available endpoints:"
echo "  - GET  http://localhost:8080/         (API info)"
echo "  - GET  http://localhost:8080/health   (Health check)"
echo "  - GET  http://localhost:8080/diseases (List diseases)"
echo "  - POST http://localhost:8080/predict  (Make prediction)"
echo ""
echo "🧪 Test commands:"
echo ""
echo "  # Health check"
echo "  curl http://localhost:8080/health"
echo ""
echo "  # List all diseases"
echo "  curl http://localhost:8080/diseases"
echo ""
echo "  # Make a prediction (replace with your image path)"
echo "  curl -X POST http://localhost:8080/predict \\"
echo "    -F \"file=@path/to/retinal_image.jpg\""
echo ""
echo "📊 View logs:"
echo "  docker logs -f retinal-disease-api"
echo ""
echo "🛑 Stop the container:"
echo "  docker stop retinal-disease-api"
echo "  docker rm retinal-disease-api"
echo ""
echo "=================================================="
