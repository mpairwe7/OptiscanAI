#!/bin/bash
# Installation script for ViT model training dependencies
# Run this before training models

set -e  # Exit on error

echo "================================================================================"
echo "INSTALLING DEPENDENCIES FOR ViT MODEL TRAINING"
echo "================================================================================"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
fi

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "================================================================================"
echo "Installing PyTorch with CUDA support..."
echo "================================================================================"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "================================================================================"
echo "Installing Transformers (Hugging Face)..."
echo "================================================================================"
pip install transformers

echo ""
echo "================================================================================"
echo "Installing Albumentations (Advanced Augmentation)..."
echo "================================================================================"
pip install albumentations

echo ""
echo "================================================================================"
echo "Installing other dependencies..."
echo "================================================================================"
pip install timm opencv-python Pillow
pip install scikit-learn scipy
pip install matplotlib seaborn
pip install pandas numpy
pip install tqdm

echo ""
echo "================================================================================"
echo "Verifying installation..."
echo "================================================================================"
python verify_setup.py

echo ""
echo "================================================================================"
echo "INSTALLATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Review verification results above"
echo "2. If all checks pass, start training:"
echo "   python train_all_models.py --models all"
echo ""
echo "For detailed guide, see: TRAINING_GUIDE.md"
echo "================================================================================"
