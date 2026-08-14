#!/usr/bin/env bash
# Train every baseline / ablation arm sequentially on one GPU.
set -u
cd "$(dirname "$0")/.."
DEV="${1:-cuda:7}"
for arm in resnet50 vit_base_patch16_224 tf_efficientnet_b3 retfound_head retfound_lora; do
    echo "=== $arm ==="
    python3 -u scripts/ncc2026_train_arm.py --arm "$arm" --device "$DEV" \
        > "outputs/ncc2026/logs/arm_${arm}.log" 2>&1
    echo "$arm exit=$?"
done
echo "ALL ARMS DONE"
