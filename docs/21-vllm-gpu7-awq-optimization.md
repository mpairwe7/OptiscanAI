# vLLM GPU 7 AWQ Optimization Runbook

> Production runtime change validated on 2026-05-07 for the OpenAI-compatible vLLM service.

## Summary

The vLLM service was moved from the BF16 `Qwen/Qwen3-8B` runtime to the AWQ-quantized `Qwen/Qwen3-8B-AWQ` runtime and the KV-cache reservation was right-sized. The final container keeps the 8K context window while freeing enough GPU 7 memory for companion services such as CosyVoice2 and Qwen2-VL.

| Metric | Before | Final | Change |
|---|---:|---:|---:|
| GPU 7 VRAM used | 41.1 GB / 49.1 GB | 15.5 GB / 49.1 GB | -25.6 GB |
| GPU 7 utilization by memory | 84% | 32% | -52 percentage points |
| Free GPU 7 VRAM | ~8.0 GB | ~33.6 GB | +25.6 GB |
| Model weights | 16.4 GB BF16 | 5.7 GB AWQ INT4 | -10.7 GB |
| vLLM KV cache | 155K tokens baseline | 31,424 tokens final | Right-sized |
| Full 8,192-token concurrency | Over-provisioned | 3.84x | Fits current workload |
| Quality gate | `faith=1.0`, `approve` | `faith=1.0`, `approve` | No regression observed |

## Final Runtime Profile

The production container is `ura-vllm`:

```bash
docker run -d \
  --name ura-vllm \
  --gpus '"device=7"' \
  -p 8011:8001 \
  -v /home/developer/.cache/huggingface:/root/.cache/huggingface \
  -v /home/developer/models/huggingface:/root/models/huggingface \
  vllm/vllm-openai:v0.8.5 \
  --model Qwen/Qwen3-8B-AWQ \
  --download-dir /root/models/huggingface \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.30 \
  --quantization awq \
  --dtype auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Important settings:

| Setting | Value | Reason |
|---|---|---|
| `--model` | `Qwen/Qwen3-8B-AWQ` | Uses 4-bit AWQ weights instead of BF16 weights |
| `--quantization` | `awq` | Required for the AWQ checkpoint |
| `--gpu-memory-utilization` | `0.30` | Prevents vLLM from filling GPU 7 with excess KV cache |
| `--max-model-len` | `8192` | Keeps the 8K context window |
| Host port | `8011` | Existing clients continue to call `http://localhost:8011` |
| Container port | `8001` | vLLM API server port |
| GPU | `7` | Leaves other GPUs untouched |

## Tuning Results

The work was validated in three steps:

| Phase | Runtime | vLLM memory budget | Observed result |
|---|---|---:|---|
| Baseline | `Qwen/Qwen3-8B` BF16 | `0.85` | 41.1 GB VRAM used, 155K-token KV cache |
| Phase 1 | `Qwen/Qwen3-8B` BF16 | `0.50` | 25.0 GB VRAM used, ~16.1 GB saved, 31K-token KV cache |
| Phase 2 | `Qwen/Qwen3-8B-AWQ` | `0.85` | 5.7 GB weights, but vLLM expanded KV cache to 221K tokens |
| Final | `Qwen/Qwen3-8B-AWQ` | `0.30` | 15.5 GB VRAM used, 31,424-token KV cache, 3.84x full 8K concurrency |

The key operational lesson is that AWQ alone reduces model-weight memory, but vLLM will reuse the freed budget for KV cache unless `--gpu-memory-utilization` is also lowered. The final profile combines AWQ with a smaller memory budget to actually free GPU memory.

If future traffic requires more simultaneous full-context requests, test `--gpu-memory-utilization 0.50` with the AWQ model and re-check GPU headroom. The current `0.30` setting is intentionally optimized for freeing GPU 7 memory, not maximum vLLM concurrency.

## Verification

Use these checks after restart:

```bash
curl -s http://localhost:8011/health

docker logs ura-vllm 2>&1 | rg 'Model loading|GPU KV cache size|Maximum concurrency'

nvidia-smi -i 7 \
  --query-gpu=index,memory.used,memory.total \
  --format=csv,noheader,nounits
```

Expected vLLM log lines for the final profile:

```text
Model loading took 5.7071 GiB
GPU KV cache size: 31,424 tokens
Maximum concurrency for 8,192 tokens per request: 3.84x
```

Run a direct vLLM sanity check:

```bash
curl -s -X POST http://localhost:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": "What is VAT in Uganda? One sentence."}],
    "temperature": 0.2,
    "max_tokens": 80
  }'
```

Then verify the full application pipeline. The validated run returned grounded sources with `faith=1.0` and judge decision `approve`.

## Rollback

If AWQ quality, latency, or compatibility regresses, stop `ura-vllm` and relaunch the BF16 model:

```bash
docker stop ura-vllm
docker rm ura-vllm

docker run -d \
  --name ura-vllm \
  --gpus '"device=7"' \
  -p 8011:8001 \
  -v /home/developer/.cache/huggingface:/root/.cache/huggingface \
  -v /home/developer/models/huggingface:/root/models/huggingface \
  vllm/vllm-openai:v0.8.5 \
  --model Qwen/Qwen3-8B \
  --download-dir /root/models/huggingface \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.50 \
  --dtype auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

The rollback profile keeps the right-sized KV cache behavior from Phase 1 and should use about 25 GB on GPU 7.

## Notes

- vLLM 0.8.5 reported that `awq_marlin` may be faster for this model, but the validated production profile uses explicit `--quantization awq`.
- The AWQ model cache lives under `/home/developer/models/huggingface`; keep that mount in place so restarts do not redownload the 5.7 GB checkpoint.
- Health can return before graph capture and warmup fully finish. Check the KV-cache and concurrency log lines before declaring the service ready for production traffic.
