#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    /root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507 \
    /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507; do
    if [[ -d "$candidate" ]]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
fi
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3 model directory}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}"
TP_SIZE="${TP_SIZE:-1}"
# The 32K fixtures include chat-template and suffix tokens beyond the shared prefix.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
PORT="${PORT:-8000}"

case "$ENABLE_PREFIX_CACHING" in
  1|true) prefix_flag="--enable-prefix-caching" ;;
  0|false) prefix_flag="--no-enable-prefix-caching" ;;
  *) echo "ENABLE_PREFIX_CACHING must be 0/1/true/false" >&2; exit 2 ;;
esac
case "$ENABLE_CHUNKED_PREFILL" in
  1|true) chunked_flag="--enable-chunked-prefill" ;;
  0|false) chunked_flag="--no-enable-chunked-prefill" ;;
  *) echo "ENABLE_CHUNKED_PREFILL must be 0/1/true/false" >&2; exit 2 ;;
esac

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  "$prefix_flag" \
  "$chunked_flag"
