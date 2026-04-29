#!/usr/bin/env bash
# =============================================================================
# Deploy RetinalAI to Hugging Face Spaces
# Usage: ./scripts/deploy_hf.sh
# Requires: HF_TOKEN env var or ~/.huggingface/token
# =============================================================================
set -euo pipefail

HF_SPACE="Mpairwe49/retinal-screening"
HF_REPO_URL="https://huggingface.co/spaces/${HF_SPACE}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORK_DIR=$(mktemp -d)

trap "rm -rf $WORK_DIR" EXIT

echo "=== RetinalAI → Hugging Face Spaces Deployment ==="
echo "Space: ${HF_SPACE}"
echo "URL:   ${HF_REPO_URL}"
echo ""

# Check HF token
if [ -z "${HF_TOKEN:-}" ]; then
    if [ -f ~/.huggingface/token ]; then
        HF_TOKEN=$(cat ~/.huggingface/token)
    else
        echo "ERROR: Set HF_TOKEN or run 'huggingface-cli login' first"
        exit 1
    fi
fi

# Install HF CLI if needed
if ! command -v huggingface-cli &>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub[cli]
fi

echo "[1/5] Cloning Space repo..."
cd "$WORK_DIR"
GIT_LFS_SKIP_SMUDGE=1 git clone "https://Mpairwe49:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE}" space 2>/dev/null || {
    echo "Space not found — creating it..."
    python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='${HF_TOKEN}')
api.create_repo(
    repo_id='${HF_SPACE}',
    repo_type='space',
    space_sdk='docker',
    private=False,
)
print('Space created!')
"
    git clone "https://Mpairwe49:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE}" space
}
cd space

echo "[2/5] Writing Space metadata..."
cat > README.md << 'EOF'
---
title: RetinalAI Clinical Screening
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: cc-by-4.0
short_description: AI-powered retinal disease screening — 45 diseases, <200ms, clinical reasoning
---

# RetinalAI: AI-Powered Retinal Disease Screening

Multi-label classification of 45 retinal diseases from fundus images using Graph Neural Networks with Clinical Knowledge Graph reasoning.

**Features:**
- 45-disease simultaneous screening from single fundus image
- Clinical Knowledge Graph with 144 disease relationships
- 5 explainability methods (GradCAM, LIME, SHAP, IG, ELI5)
- Referral priority ranking (Emergency / Urgent / Routine)
- Sub-200ms inference latency
EOF

echo "[3/5] Syncing project files..."
rsync -av --delete \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='.next' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='outputs' \
    --exclude='data' \
    --exclude='wandb' \
    --exclude='mlruns' \
    --exclude='.env' \
    --exclude='.env.local' \
    --exclude='*.onnx' \
    --exclude='docs/*.pdf' \
    --exclude='logs' \
    --exclude='hf_space' \
    --exclude='pretrained_weights' \
    --exclude='ipynb_checkpoints' \
    "${PROJECT_DIR}/" ./

# Use Dockerfile.hf as the Space Dockerfile
cp Dockerfile.hf Dockerfile

echo "[4/5] Configuring Git LFS for model weights..."
git lfs install
git lfs track "*.pth"
git lfs track "models/*.pth"
git add .gitattributes

echo "[5/5] Committing and pushing..."
git add -A
if git diff --cached --quiet; then
    echo "No changes to deploy."
    exit 0
fi

COMMIT_MSG="Deploy $(date +%Y-%m-%d\ %H:%M) — $(cd "$PROJECT_DIR" && git log -1 --format='%h %s' 2>/dev/null || echo 'manual')"
git commit -m "$COMMIT_MSG"
git push

echo ""
echo "=== Deployment triggered! ==="
echo "Space:   https://${HF_SPACE/\//-}.hf.space"
echo "Monitor: ${HF_REPO_URL}"
echo "Logs:    ${HF_REPO_URL}/logs"
echo ""
echo "Docker build takes 5-15 minutes on HF Spaces."
echo "Check build status at the URLs above."
