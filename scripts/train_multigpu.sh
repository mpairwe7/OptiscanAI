#!/usr/bin/env bash
# =============================================================================
# Multi-GPU Training Launcher for Retinal Disease Classification
# Uses torchrun (PyTorch native) for DDP across 8x NVIDIA RTX A6000
# =============================================================================
set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-configs/train.yaml}"
NUM_GPUS="${2:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "$PROJECT_DIR"

echo "============================================================"
echo "  Retinal Disease MLOps - Multi-GPU Training"
echo "============================================================"
echo "  Project dir : $PROJECT_DIR"
echo "  Config      : $CONFIG"
echo "  GPUs        : $NUM_GPUS"
echo "  Master port : $MASTER_PORT"
echo "  Timestamp   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# --- GPU Status ---
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | head -${NUM_GPUS}
echo ""

# --- Environment ---
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Prevent fragmentation on A6000s
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Launch Training ---
echo "Launching torchrun with $NUM_GPUS GPUs..."
echo ""

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    "${@:3}"

echo ""
echo "============================================================"
echo "  Training Complete!"
echo "  Checkpoints: outputs/checkpoints/"
echo "  History:     outputs/training_history.json"
echo "============================================================"
