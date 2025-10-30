#!/bin/bash
# Deploy retinal screening container to GCP with GPU

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deploy to Google Cloud Platform (GCP)${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Check if Docker Hub username is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 <dockerhub-username> [gcp-project-id] [zone]${NC}"
    echo -e "${YELLOW}Example: $0 myusername retinal-screening-prod us-central1-a${NC}"
    echo
    exit 1
fi

DOCKER_USERNAME="$1"
PROJECT_ID="${2:-retinal-screening-prod}"
ZONE="${3:-us-central1-a}"
INSTANCE_NAME="retinal-screening-gpu-instance"
IMAGE_NAME="retinal-screening-gpu"
DOCKER_IMAGE="docker.io/${DOCKER_USERNAME}/${IMAGE_NAME}:latest"

echo -e "${BLUE}Configuration:${NC}"
echo "  Docker Hub username: ${DOCKER_USERNAME}"
echo "  GCP Project ID: ${PROJECT_ID}"
echo "  Zone: ${ZONE}"
echo "  Instance name: ${INSTANCE_NAME}"
echo "  Docker image: ${DOCKER_IMAGE}"
echo

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}✗ gcloud CLI not found${NC}"
    echo -e "${YELLOW}Please install: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi
echo -e "${GREEN}✓ gcloud CLI found${NC}"

# Step 1: Create GCP Project (if needed)
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 1: GCP Project Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${YELLOW}Creating/verifying GCP project: ${PROJECT_ID}${NC}"
echo -e "${YELLOW}Note: If project exists, this will just verify it${NC}"

# Try to create project (will fail gracefully if exists)
gcloud projects create ${PROJECT_ID} --name="Retinal Screening Production" 2>/dev/null || true

# Set active project
if ! gcloud config set project ${PROJECT_ID}; then
    echo -e "${RED}✗ Failed to set project${NC}"
    echo -e "${YELLOW}You may need to create the project manually in GCP Console${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Project set: ${PROJECT_ID}${NC}"

# Step 2: Enable Required APIs
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 2: Enable GCP APIs${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${YELLOW}Enabling required APIs (may take 2-3 minutes)...${NC}"

gcloud services enable compute.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com

echo -e "${GREEN}✓ APIs enabled${NC}"

# Step 3: Create Firewall Rule
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 3: Configure Firewall${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${YELLOW}Creating firewall rule for port 8000...${NC}"

gcloud compute firewall-rules create allow-retinal-api \
    --allow tcp:8000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow access to Retinal Screening API" \
    2>/dev/null || echo -e "${YELLOW}Firewall rule may already exist${NC}"

echo -e "${GREEN}✓ Firewall configured${NC}"

# Step 4: Create GPU Instance
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 4: Create GPU Instance${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${YELLOW}Creating VM instance with NVIDIA T4 GPU...${NC}"
echo -e "${BLUE}Machine type: n1-standard-4 (4 vCPUs, 15 GB RAM)${NC}"
echo -e "${BLUE}GPU: 1x NVIDIA Tesla T4${NC}"
echo -e "${BLUE}Estimated cost: ~\$0.35/hour (~\$272/month if 24/7)${NC}"
echo

gcloud compute instances create ${INSTANCE_NAME} \
    --zone=${ZONE} \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=15GB \
    --boot-disk-type=pd-standard \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --metadata=startup-script='#!/bin/bash
# Install Docker and NVIDIA Container Toolkit
apt-get update
apt-get install -y docker.io

# Install NVIDIA drivers and container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit nvidia-driver-535
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Pull and run the container
docker pull '"${DOCKER_IMAGE}"'
docker run -d \
  --name retinal-screening \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  '"${DOCKER_IMAGE}"'

echo "Retinal Screening API deployed successfully!"
' || {
    echo -e "${RED}✗ Failed to create instance${NC}"
    echo -e "${YELLOW}Common issues:${NC}"
    echo -e "  1. GPU quota exceeded (request quota increase)"
    echo -e "  2. Zone doesn't have T4 GPUs (try us-central1-a, us-west1-b)"
    echo -e "  3. Billing not enabled (enable in GCP Console)"
    exit 1
}

echo -e "${GREEN}✓ Instance created successfully!${NC}"

# Step 5: Get Instance Details
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 5: Instance Details${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${YELLOW}Waiting for instance to be ready (may take 3-5 minutes)...${NC}"
sleep 10

EXTERNAL_IP=$(gcloud compute instances describe ${INSTANCE_NAME} \
    --zone=${ZONE} \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Access Information${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${BLUE}API Endpoint:${NC}"
echo -e "  ${GREEN}http://${EXTERNAL_IP}:8000${NC}"
echo
echo -e "${BLUE}API Documentation:${NC}"
echo -e "  ${GREEN}http://${EXTERNAL_IP}:8000/docs${NC}"
echo
echo -e "${BLUE}Health Check:${NC}"
echo -e "  ${GREEN}curl http://${EXTERNAL_IP}:8000/health${NC}"
echo
echo -e "${BLUE}SSH Access:${NC}"
echo -e "  ${YELLOW}gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}${NC}"
echo
echo -e "${BLUE}View Logs:${NC}"
echo -e "  ${YELLOW}gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='docker logs -f retinal-screening'${NC}"
echo
echo -e "${BLUE}Stop Instance:${NC}"
echo -e "  ${YELLOW}gcloud compute instances stop ${INSTANCE_NAME} --zone=${ZONE}${NC}"
echo
echo -e "${BLUE}Start Instance:${NC}"
echo -e "  ${YELLOW}gcloud compute instances start ${INSTANCE_NAME} --zone=${ZONE}${NC}"
echo
echo -e "${BLUE}Delete Instance:${NC}"
echo -e "  ${RED}gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE}${NC}"
echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cost Optimization Tips${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "1. ${YELLOW}Stop instance when not in use:${NC}"
echo -e "   Saves ~70% of costs (only pay for disk storage)"
echo
echo -e "2. ${YELLOW}Setup automatic shutdown schedule:${NC}"
echo -e "   gcloud compute instances add-resource-policies ${INSTANCE_NAME} \\"
echo -e "     --resource-policies=retinal-schedule --zone=${ZONE}"
echo
echo -e "3. ${YELLOW}Use preemptible instances for testing:${NC}"
echo -e "   Add --preemptible flag (saves 70-90% but can be terminated)"
echo
echo -e "4. ${YELLOW}Monitor costs:${NC}"
echo -e "   Check GCP Console > Billing > Cost Table"
echo
echo -e "${GREEN}Deployment complete! 🎉${NC}"
