#!/usr/bin/env bash
# FP32 vs INT8 on identical CPU threading, so accuracy and latency both compare.
set -u
cd "$(dirname "$0")/.."
for v in fp32_cpu int8; do
    python3 -u scripts/ncc2026_infer.py --split test --variant "$v" --threads 16 \
        > "outputs/ncc2026/logs/infer_test_${v}.log" 2>&1
    echo "$v exit=$?"
done
echo "QUANT DONE"
