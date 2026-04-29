#!/usr/bin/env bash
# =============================================================================
# Dataset Download Script (RFMiD + Infant ROP)
# =============================================================================
set -euo pipefail

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

RFMID_DIR="data/rfmid"
ROP_DIR="data/rop_infants"
mkdir -p "$RFMID_DIR" "$ROP_DIR"

# ---- Helper: download via kagglehub ----
download_kagglehub() {
    local dataset="$1"
    local dest="$2"
    python3 -c "
import kagglehub
path = kagglehub.dataset_download('$dataset')
print(f'Downloaded to: {path}')
import shutil, os
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join('$dest', item)
    if not os.path.exists(dst):
        if os.path.isdir(src):
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
print('Linked to $dest')
"
}

# ==========================================================================
#  1) RFMiD Dataset
# ==========================================================================
echo "============================================================"
echo "  RFMiD Dataset Download"
echo "============================================================"

if command -v kaggle &> /dev/null && [ -f ~/.kaggle/kaggle.json ]; then
    echo "Using Kaggle CLI..."
    kaggle datasets download -d ritheshsreenivasan/rfmid-dataset -p "$RFMID_DIR" --unzip
    echo "RFMiD download complete!"
elif [ -n "${KAGGLE_API_TOKEN:-}" ] || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; }; then
    echo "Using kagglehub..."
    download_kagglehub "ritheshsreenivasan/rfmid-dataset" "$RFMID_DIR"
    echo "RFMiD download complete!"
else
    echo "Skipping RFMiD — no Kaggle credentials found."
fi

# ==========================================================================
#  2) Infant Retinal / ROP Dataset
# ==========================================================================
echo ""
echo "============================================================"
echo "  Infant Retinal (ROP) Dataset Download"
echo "============================================================"

if command -v kaggle &> /dev/null && [ -f ~/.kaggle/kaggle.json ]; then
    echo "Using Kaggle CLI..."
    kaggle datasets download -d jananowakova/retinal-image-dataset-of-infants-and-rop -p "$ROP_DIR" --unzip
    echo "ROP dataset download complete!"
elif [ -n "${KAGGLE_API_TOKEN:-}" ] || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; }; then
    echo "Using kagglehub..."
    download_kagglehub "jananowakova/retinal-image-dataset-of-infants-and-rop" "$ROP_DIR"
    echo "ROP dataset download complete!"
else
    echo "Skipping ROP — no Kaggle credentials found."
fi

# ==========================================================================
#  Credentials help (if nothing was downloaded)
# ==========================================================================
if [ -z "${KAGGLE_API_TOKEN:-}" ] && { [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; } && ! { command -v kaggle &> /dev/null && [ -f ~/.kaggle/kaggle.json ]; }; then
    echo ""
    echo "No Kaggle credentials found. Set up one of:"
    echo ""
    echo "  Option A: KAGGLE_API_TOKEN (recommended)"
    echo "    Add to your .env file:"
    echo "    KAGGLE_API_TOKEN=your_token_here"
    echo ""
    echo "  Option B: Kaggle API key file"
    echo "    1. Go to https://www.kaggle.com/settings"
    echo "    2. Click 'Create New Token' to download kaggle.json"
    echo "    3. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/"
    echo "    4. chmod 600 ~/.kaggle/kaggle.json"
    echo "    5. Re-run this script"
    echo ""
    echo "  Option C: Environment variables"
    echo "    export KAGGLE_USERNAME=your_username"
    echo "    export KAGGLE_KEY=your_api_key"
    echo ""
    echo "  Option D: Generate synthetic data for pipeline testing"
    echo "    python3 scripts/generate_synthetic_data.py"
    echo ""
    exit 1
fi
