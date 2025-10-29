#!/bin/bash
# ============================================================================
# GCP Cloud Run Deployment Script
# Deploys from DockerHub using preconfigured credentials
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
SERVICE_NAME="retinal-disease-api"
DEFAULT_REGION="asia-southeast1"

echo "============================================================================"
echo "Deploying to Google Cloud Platform (Cloud Run)"
echo "============================================================================"

# Get GCP project ID
read -p "Enter your GCP Project ID: " GCP_PROJECT_ID

if [ -z "$GCP_PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP Project ID is required${NC}"
    exit 1
fi

# Get region
read -p "Enter GCP region (default: $DEFAULT_REGION): " GCP_REGION
GCP_REGION=${GCP_REGION:-$DEFAULT_REGION}

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  DockerHub Image: docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest"
echo "  GCP Project: $GCP_PROJECT_ID"
echo "  GCP Region: $GCP_REGION"
echo "  Service Name: $SERVICE_NAME"
echo ""

read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Set GCP project
echo -e "\n${YELLOW}Configuring GCP project...${NC}"
gcloud config set project $GCP_PROJECT_ID

# Enable Cloud Run API
echo -e "\n${YELLOW}Enabling Cloud Run API...${NC}"
gcloud services enable run.googleapis.com --quiet

# Deploy to Cloud Run
echo -e "\n${YELLOW}Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
  --image docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest \
  --platform managed \
  --region $GCP_REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 1 \
  --timeout 300 \
  --set-env-vars LOG_LEVEL=info,ENVIRONMENT=production

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --platform managed \
  --region $GCP_REGION \
  --format 'value(status.url)')

# Test deployment
echo -e "\n${YELLOW}Testing deployment...${NC}"
sleep 10

if curl -f ${SERVICE_URL}/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Deployment health check passed${NC}"
else
    echo -e "${YELLOW}⚠ Health check failed (service may still be starting)${NC}"
fi

echo -e "\n${GREEN}============================================================================${NC}"
echo -e "${GREEN}✓ Deployment Successful!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo -e "\nService URL: ${GREEN}${SERVICE_URL}${NC}"
echo -e "\nTest your deployment:"
echo -e "  curl ${SERVICE_URL}/health"
echo -e "\nAPI Documentation:"
echo -e "  ${SERVICE_URL}/docs"
echo -e "\nView logs:"
echo -e "  gcloud run services logs read ${SERVICE_NAME} --region ${GCP_REGION}"
echo -e "\nUpdate service:"
echo -e "  gcloud run services update ${SERVICE_NAME} --region ${GCP_REGION}"
echo -e "\nDelete service:"
echo -e "  gcloud run services delete ${SERVICE_NAME} --region ${GCP_REGION}"
