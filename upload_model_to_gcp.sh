#!/bin/bash
# Upload model to GCP instance and restart container with model mounted

set -e

INSTANCE_NAME="retinal-screening-cpu-instance"
ZONE="asia-southeast1-b"
PROJECT_ID="smart-room-dashboard-2025"
LOCAL_MODEL_PATH="models/GraphCLIP_fold1_best.pth"
DOCKER_IMAGE="docker.io/landwind/retinal-screening-gpu:latest"

echo "========================================"
echo "Upload Model to GCP Instance"
echo "========================================"

# Set project
gcloud config set project $PROJECT_ID

# Check if model exists locally
if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "❌ Error: Model file not found at $LOCAL_MODEL_PATH"
    exit 1
fi

echo "✓ Found model file: $LOCAL_MODEL_PATH"
MODEL_SIZE=$(du -h "$LOCAL_MODEL_PATH" | cut -f1)
echo "  Size: $MODEL_SIZE"

# Upload model to GCP instance
echo ""
echo "Uploading model to GCP instance..."
gcloud compute scp "$LOCAL_MODEL_PATH" \
    $INSTANCE_NAME:/tmp/model.pth \
    --zone=$ZONE \
    --project=$PROJECT_ID

echo "✓ Model uploaded successfully!"

# Stop and remove existing container, then restart with model mounted
echo ""
echo "Restarting container with model mounted..."
gcloud compute ssh $INSTANCE_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
        # Stop and remove existing container
        sudo docker stop retinal-screening 2>/dev/null || true
        sudo docker rm retinal-screening 2>/dev/null || true
        
        # Create models directory
        sudo mkdir -p /opt/models
        sudo mv /tmp/model.pth /opt/models/GraphCLIP_fold1_best.pth
        sudo chmod 644 /opt/models/GraphCLIP_fold1_best.pth
        
        # Run container with model mounted
        sudo docker run -d \
            --name retinal-screening \
            -p 8000:8000 \
            -v /opt/models:/app/models:ro \
            -e MODEL_PATH=/app/models/GraphCLIP_fold1_best.pth \
            $DOCKER_IMAGE
        
        echo 'Container restarted with model mounted'
        sleep 5
        sudo docker logs retinal-screening --tail 20
    "

echo ""
echo "✓ Container restarted successfully!"
echo ""
echo "Waiting for API to start..."
sleep 10

# Get instance external IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

# Test the API
echo ""
echo "Testing API..."
curl -s http://$EXTERNAL_IP:8000/health | python3 -m json.tool

echo ""
echo "========================================"
echo "Model Upload Complete!"
echo "========================================"
echo ""
echo "API Endpoint: http://$EXTERNAL_IP:8000"
echo "Check model status: curl http://$EXTERNAL_IP:8000/health"
echo ""
echo "The model should now show model_loaded: true"
