#!/bin/bash
# Deploy to GCP WITHOUT GPU (CPU-only) - for immediate deployment

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DOCKER_USERNAME="${1:-landwind}"
PROJECT_ID="${2:-smart-room-dashboard-2025}"
ZONE="${3:-us-central1-a}"
INSTANCE_NAME="retinal-screening-cpu-instance"
DOCKER_IMAGE="docker.io/${DOCKER_USERNAME}/retinal-screening-gpu:latest"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deploy to GCP (CPU-Only - No GPU)${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${YELLOW}Note: This deploys without GPU for immediate availability${NC}"
echo -e "${YELLOW}Performance will be slower but suitable for testing/demo${NC}"
echo

gcloud config set project ${PROJECT_ID}

echo -e "${GREEN}Creating CPU instance...${NC}"
gcloud compute instances create ${INSTANCE_NAME} \
    --zone=${ZONE} \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=15GB \
    --boot-disk-type=pd-standard \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y docker.io

# Pull and run container (CPU mode)
docker pull '"${DOCKER_IMAGE}"'
docker run -d \
  --name retinal-screening \
  --restart unless-stopped \
  -p 8000:8000 \
  '"${DOCKER_IMAGE}"'
' && {
    echo -e "${GREEN}✓ Instance created successfully!${NC}"
    
    sleep 10
    EXTERNAL_IP=$(gcloud compute instances describe ${INSTANCE_NAME} \
        --zone=${ZONE} \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    echo -e "${GREEN}API Endpoint: http://${EXTERNAL_IP}:8000${NC}"
    echo -e "${GREEN}API Docs: http://${EXTERNAL_IP}:8000/docs${NC}"
    echo -e "${YELLOW}Health Check: curl http://${EXTERNAL_IP}:8000/health${NC}"
    echo
    echo -e "${YELLOW}Cost: ~\$0.15/hour (~\$110/month if 24/7)${NC}"
    echo
} || {
    echo -e "${RED}✗ Failed to create instance${NC}"
    exit 1
}
