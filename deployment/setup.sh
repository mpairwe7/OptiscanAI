#!/bin/bash

# ============================================================================
# Deployment Setup Script for Multi Retinal Disease Model
# ============================================================================

echo "========================================="
echo "Multi Retinal Disease Model Deployment"
echo "========================================="

# 1. Create necessary directories
echo "Creating directory structure..."
mkdir -p models/checkpoints
mkdir -p models/exports
mkdir -p models/outputs
mkdir -p logs

# 2. Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# 3. Create virtual environment
echo "Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 4. Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Set up environment variables
echo "Setting up environment variables..."
cat > .env << EOF
# Model Configuration
NUM_EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=0.0001
NUM_CLASSES=45

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths
MODEL_DIR=models/checkpoints
OUTPUT_DIR=models/outputs
LOG_DIR=logs

# Kaggle API (if needed)
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_api_key
EOF

# 6. Download pre-trained models (if available)
echo "Checking for pre-trained models..."
# Add your model download logic here

# 7. Run validation
echo "Running validation checks..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo "========================================="
echo "Deployment setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Run training: python src/02_Model_Development.py"
echo "3. Check notebooks: jupyter notebook notebooks/"
echo ""
