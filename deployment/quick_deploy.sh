#!/bin/bash
# ============================================================================
# Quick Deployment Script - Retinal Screening AI
# ============================================================================
# This script automates the deployment pipeline from local to GCP
#
# Usage:
#   ./deployment/quick_deploy.sh [step]
#
# Steps:
#   1. optimize    - Optimize model for deployment
#   2. build       - Build Podman container
#   3. test        - Test container locally
#   4. push        - Push to Docker Hub
#   5. deploy      - Deploy to GCP
#   all            - Run all steps

set -e  # Exit on error

# Configuration
PROJECT_NAME="retinal-screening"
DOCKER_USERNAME="${DOCKER_USERNAME:-your_username}"
GCP_PROJECT="${GCP_PROJECT:-retinal-screening-prod}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
MODEL_PATH="models/GraphCLIP_fold1_best.pth"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed. Please install it first."
        exit 1
    fi
}

# Step 1: Optimize model
optimize_model() {
    print_header "STEP 1: OPTIMIZING MODEL FOR DEPLOYMENT"
    
    if [ ! -f "$MODEL_PATH" ]; then
        print_error "Model not found at $MODEL_PATH"
        print_warning "Please download your trained model first"
        exit 1
    fi
    
    print_success "Model found at $MODEL_PATH"
    
    # Run optimization script
    python src/optimize_for_deployment.py \
        --model "$MODEL_PATH" \
        --output-dir models/exports \
        --num-classes 45
    
    print_success "Model optimization complete!"
    print_success "Optimized model saved to models/exports/"
}

# Step 2: Build container
build_container() {
    print_header "STEP 2: BUILDING CONTAINER WITH PODMAN"
    
    check_command podman
    
    # Build image
    podman build \
        -f Dockerfile.gpu \
        -t ${PROJECT_NAME}-gpu:latest \
        --format docker \
        .
    
    # Verify build
    podman images | grep ${PROJECT_NAME}
    
    print_success "Container built successfully!"
}

# Step 3: Test container locally
test_container() {
    print_header "STEP 3: TESTING CONTAINER LOCALLY"
    
    # Stop existing container if running
    podman stop ${PROJECT_NAME}-test 2>/dev/null || true
    podman rm ${PROJECT_NAME}-test 2>/dev/null || true
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        print_success "GPU detected - running with GPU support"
        GPU_FLAG="--device nvidia.com/gpu=all"
    else
        print_warning "No GPU detected - running on CPU only"
        GPU_FLAG=""
    fi
    
    # Run container
    podman run -d \
        --name ${PROJECT_NAME}-test \
        ${GPU_FLAG} \
        -p 8000:8000 \
        -v $(pwd)/models/exports:/app/models/exports:ro \
        -e MODEL_PATH=/app/models/exports/GraphCLIP_optimized.onnx \
        -e LOG_LEVEL=INFO \
        ${PROJECT_NAME}-gpu:latest
    
    print_success "Container started"
    
    # Wait for container to be ready
    echo -n "Waiting for API to be ready"
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo ""
            print_success "API is ready!"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    # Test endpoints
    echo -e "\n${YELLOW}Testing API endpoints...${NC}"
    
    echo -e "\n1. Health check:"
    curl -s http://localhost:8000/health | jq .
    
    echo -e "\n2. API info:"
    curl -s http://localhost:8000/api/v1/info | jq .
    
    # Show logs
    echo -e "\n${YELLOW}Container logs:${NC}"
    podman logs --tail=20 ${PROJECT_NAME}-test
    
    print_success "Local testing complete!"
    echo -e "\n${YELLOW}Container is running at: http://localhost:8000${NC}"
    echo -e "${YELLOW}To view logs: podman logs -f ${PROJECT_NAME}-test${NC}"
    echo -e "${YELLOW}To stop: podman stop ${PROJECT_NAME}-test${NC}"
}

# Step 4: Push to Docker Hub
push_to_dockerhub() {
    print_header "STEP 4: PUSHING TO DOCKER HUB"
    
    if [ "$DOCKER_USERNAME" == "your_username" ]; then
        print_error "Please set DOCKER_USERNAME environment variable"
        echo "  export DOCKER_USERNAME=your_dockerhub_username"
        exit 1
    fi
    
    # Login to Docker Hub
    echo "Logging in to Docker Hub..."
    podman login docker.io
    
    # Tag image
    podman tag \
        ${PROJECT_NAME}-gpu:latest \
        docker.io/${DOCKER_USERNAME}/${PROJECT_NAME}-gpu:latest
    
    podman tag \
        ${PROJECT_NAME}-gpu:latest \
        docker.io/${DOCKER_USERNAME}/${PROJECT_NAME}-gpu:v1.0.0
    
    print_success "Image tagged"
    
    # Push images
    echo "Pushing latest tag..."
    podman push docker.io/${DOCKER_USERNAME}/${PROJECT_NAME}-gpu:latest
    
    echo "Pushing version tag..."
    podman push docker.io/${DOCKER_USERNAME}/${PROJECT_NAME}-gpu:v1.0.0
    
    print_success "Images pushed to Docker Hub!"
    echo -e "\n${YELLOW}View at: https://hub.docker.com/r/${DOCKER_USERNAME}/${PROJECT_NAME}-gpu${NC}"
}

# Step 5: Deploy to GCP
deploy_to_gcp() {
    print_header "STEP 5: DEPLOYING TO GCP"
    
    check_command gcloud
    
    # Set project
    gcloud config set project $GCP_PROJECT
    
    # Check if instance exists
    if gcloud compute instances describe ${PROJECT_NAME}-gpu --zone=$GCP_ZONE &>/dev/null; then
        print_warning "Instance already exists. Updating deployment..."
        
        # SSH and update container
        gcloud compute ssh ${PROJECT_NAME}-gpu --zone=$GCP_ZONE --command="
            docker pull ${DOCKER_USERNAME}/${PROJECT_NAME}-gpu:latest
            docker stop ${PROJECT_NAME}-api || true
            docker rm ${PROJECT_NAME}-api || true
            docker run -d \
                --name ${PROJECT_NAME}-api \
                --gpus all \
                --restart unless-stopped \
                -p 8000:8000 \
                -e MODEL_PATH=/app/models/exports/GraphCLIP_optimized.onnx \
                ${DOCKER_USERNAME}/${PROJECT_NAME}-gpu:latest
        "
    else
        print_warning "Instance not found. Creating new instance..."
        
        # Create instance
        gcloud compute instances create ${PROJECT_NAME}-gpu \
            --project=$GCP_PROJECT \
            --zone=$GCP_ZONE \
            --machine-type=n1-standard-4 \
            --accelerator=type=nvidia-tesla-t4,count=1 \
            --maintenance-policy=TERMINATE \
            --image-family=cos-stable \
            --image-project=cos-cloud \
            --boot-disk-size=50GB \
            --boot-disk-type=pd-ssd \
            --tags=http-server,https-server \
            --scopes=cloud-platform
        
        # Create firewall rule
        gcloud compute firewall-rules create allow-${PROJECT_NAME}-api \
            --project=$GCP_PROJECT \
            --allow=tcp:8000 \
            --source-ranges=0.0.0.0/0 \
            --target-tags=http-server || true
        
        # Wait for instance to be ready
        sleep 30
        
        # Deploy container
        gcloud compute ssh ${PROJECT_NAME}-gpu --zone=$GCP_ZONE --command="
            docker pull ${DOCKER_USERNAME}/${PROJECT_NAME}-gpu:latest
            docker run -d \
                --name ${PROJECT_NAME}-api \
                --gpus all \
                --restart unless-stopped \
                -p 8000:8000 \
                -e MODEL_PATH=/app/models/exports/GraphCLIP_optimized.onnx \
                ${DOCKER_USERNAME}/${PROJECT_NAME}-gpu:latest
        "
    fi
    
    # Get external IP
    EXTERNAL_IP=$(gcloud compute instances describe ${PROJECT_NAME}-gpu \
        --zone=$GCP_ZONE \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    
    print_success "Deployment complete!"
    echo -e "\n${GREEN}API URL: http://${EXTERNAL_IP}:8000${NC}"
    echo -e "${GREEN}API Docs: http://${EXTERNAL_IP}:8000/docs${NC}"
    
    # Test deployment
    echo -e "\n${YELLOW}Testing deployment...${NC}"
    sleep 10
    curl -s http://${EXTERNAL_IP}:8000/health | jq . || print_warning "API not responding yet (might need more time)"
}

# Main script
main() {
    case ${1:-all} in
        optimize|1)
            optimize_model
            ;;
        build|2)
            build_container
            ;;
        test|3)
            test_container
            ;;
        push|4)
            push_to_dockerhub
            ;;
        deploy|5)
            deploy_to_gcp
            ;;
        all)
            optimize_model
            build_container
            test_container
            
            read -p "Push to Docker Hub and deploy to GCP? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                push_to_dockerhub
                deploy_to_gcp
            fi
            ;;
        *)
            echo "Usage: $0 {optimize|build|test|push|deploy|all}"
            echo ""
            echo "Steps:"
            echo "  optimize (1) - Optimize model for deployment"
            echo "  build (2)    - Build Podman container"
            echo "  test (3)     - Test container locally"
            echo "  push (4)     - Push to Docker Hub"
            echo "  deploy (5)   - Deploy to GCP"
            echo "  all          - Run all steps (interactive)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
