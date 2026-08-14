#!/usr/bin/env bash
# Preprocessing ablation: the two choices reviewers questioned, measured rather
# than argued. Same arm, same recipe, only the input transform changes.
set -u
cd "$(dirname "$0")/.."
DEV="${1:-cuda:5}"
run() { # name norm preproc
    python3 -u scripts/ncc2026_train_arm.py --arm retfound_lora --device "$DEV" \
        --norm "$2" --preproc "$3" --tag "$1" \
        > "outputs/ncc2026/logs/arm_retfound_lora_$1.log" 2>&1
    echo "$1 exit=$?"
}
run datasetnorm dataset none
run clahe       imagenet clahe
echo "PREPROC DONE"
