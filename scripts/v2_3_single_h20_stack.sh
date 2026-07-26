#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507}"
export SMOKE_MODEL_NAME="${SMOKE_MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}"
export V2_LITE_LOG_ROOT="${V2_3_SINGLE_H20_LOG_ROOT:-/root/log/workload-aware-kv-cache/v2-3-single-h20}"
export SMOKE_LMCACHE_CONFIG="${SMOKE_LMCACHE_CONFIG:-$ROOT/configs/v2_3_single_h20/lmcache.yaml}"
export SMOKE_ROUTER_CONFIG="${SMOKE_ROUTER_CONFIG:-$ROOT/configs/v2_3_single_h20/agent-slo-adaptive.yaml}"
export SMOKE_LMCACHE_INSTANCE_ID="${SMOKE_LMCACHE_INSTANCE_ID:-v2-3-single-h20}"
export SMOKE_MOONCAKE_CLIENT_METRICS_PORT="${SMOKE_MOONCAKE_CLIENT_METRICS_PORT:-9300}"
export SMOKE_MAX_NUM_SEQS="${SMOKE_MAX_NUM_SEQS:-12}"
export SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-40960}"
export SMOKE_GPU_MEMORY_UTILIZATION="${SMOKE_GPU_MEMORY_UTILIZATION:-0.90}"
export SMOKE_MODEL_DTYPE="${SMOKE_MODEL_DTYPE:-auto}"
export SMOKE_ENFORCE_EAGER="${SMOKE_ENFORCE_EAGER:-0}"
export SMOKE_KV_EVENTS_CONFIG="${SMOKE_KV_EVENTS_CONFIG:-{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:9400\",\"topic\":\"workload-aware-kv\"}}"
export SMOKE_SAFETENSORS_LOAD_STRATEGY="${SMOKE_SAFETENSORS_LOAD_STRATEGY:-}"

exec "$ROOT/scripts/v2_lite_2080ti_stack.sh" "$@"
