#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
PORT="${PORT:-8000}"

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --enable-prefix-caching \
  --enable-chunked-prefill
