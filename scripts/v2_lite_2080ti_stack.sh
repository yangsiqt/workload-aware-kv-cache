#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
MODEL="${MODEL_PATH:-/root/autodl-fs/models/Qwen3-0.6B}"
MODEL_NAME="${SMOKE_MODEL_NAME:-Qwen3-0.6B}"
WORKER_PYTHON="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
ROUTER="${ROUTER_BIN:-/root/.venvs/vllm-router/bin/vllm-router}"
MOONCAKE_MASTER="/root/.venvs/kv-worker/lib/python3.12/site-packages/mooncake/mooncake_master"
LOG_ROOT="${V2_LITE_LOG_ROOT:-/root/log/workload-aware-kv-cache/v2-lite-2080ti}"
PID_DIR="$LOG_ROOT/pids"
MAX_NUM_SEQS="${SMOKE_MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${SMOKE_GPU_MEMORY_UTILIZATION:-0.85}"
MODEL_DTYPE="${SMOKE_MODEL_DTYPE:-half}"
ENFORCE_EAGER="${SMOKE_ENFORCE_EAGER:-1}"
LMCACHE_CONFIG="${SMOKE_LMCACHE_CONFIG:-$ROOT/configs/v2_lite/lmcache-2080ti.yaml}"
ROUTER_CONFIG="${SMOKE_ROUTER_CONFIG:-$ROOT/configs/v2_lite/agent-slo-2080ti.yaml}"
LMCACHE_INSTANCE_ID="${SMOKE_LMCACHE_INSTANCE_ID:-v2-lite-2080ti}"
MOONCAKE_CLIENT_METRICS_PORT="${SMOKE_MOONCAKE_CLIENT_METRICS_PORT:-9300}"
KV_EVENTS_CONFIG="${SMOKE_KV_EVENTS_CONFIG:-}"
SAFETENSORS_LOAD_STRATEGY="${SMOKE_SAFETENSORS_LOAD_STRATEGY:-}"

mkdir -p "$PID_DIR" "$LOG_ROOT/components" "$LOG_ROOT/serving" "$LOG_ROOT/routing"

start_process() {
  local name="$1"
  shift
  nohup "$@" >"$LOG_ROOT/components/$name.log" 2>&1 &
  echo "$!" >"$PID_DIR/$name.pid"
}

wait_http() {
  local name="$1" url="$2"
  for _ in $(seq 1 600); do
    curl -fsS "$url" >/dev/null 2>&1 && return
    if ! kill -0 "$(<"$PID_DIR/$name.pid")" 2>/dev/null; then
      tail -100 "$LOG_ROOT/components/$name.log" >&2
      return 1
    fi
    sleep 1
  done
  echo "timeout waiting for $name" >&2
  return 1
}

wait_port() {
  local name="$1" port="$2"
  for _ in $(seq 1 120); do
    "$WORKER_PYTHON" - "$port" <<'PY' >/dev/null 2>&1 && return
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.2):
    pass
PY
    if ! kill -0 "$(<"$PID_DIR/$name.pid")" 2>/dev/null; then
      tail -100 "$LOG_ROOT/components/$name.log" >&2
      return 1
    fi
    sleep 1
  done
  return 1
}

stop_one() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 0
  local pid
  pid="$(<"$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

stop_all() {
  for name in router backend lmcache-controller mooncake-metadata mooncake-master; do
    stop_one "$PID_DIR/$name.pid"
  done
  for _ in $(seq 1 60); do
    active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')"
    [[ -z "$active_gpu_pids" ]] && return
    sleep 0.5
  done
  echo "GPU process did not exit after stack stop" >&2
  return 1
}

start_all() {
  test -d "$MODEL"
  [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1 ]]
  stop_all
  for trace in \
    "$LOG_ROOT/serving/backend.connector-trace.jsonl" \
    "$LOG_ROOT/serving/backend.connector-actual-trace.jsonl" \
    "$LOG_ROOT/routing/router-trace.jsonl"
  do
    if [[ -f "$trace" ]]; then
      mv "$trace" "$trace.previous.$(date -u +%Y%m%dT%H%M%SZ)"
    fi
  done

  start_process mooncake-master "$MOONCAKE_MASTER" -v=1 --metrics_port=9004
  wait_port mooncake-master 50051
  start_process mooncake-metadata \
    /root/.venvs/kv-worker/bin/mooncake_http_metadata_server --port 8005
  wait_port mooncake-metadata 8005
  start_process lmcache-controller env PYTHONHASHSEED=123 \
    /root/.venvs/kv-worker/bin/lmcache_controller \
    --host 127.0.0.1 --port 9000 \
    --monitor-ports '{"pull":9101,"reply":9102,"heartbeat":9103}'
  wait_http lmcache-controller http://127.0.0.1:9000/

  local vllm_args=(
    "$WORKER_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL"
    --served-model-name "$MODEL_NAME" --host 127.0.0.1 --port 8000
    --dtype "$MODEL_DTYPE" --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enable-prefix-caching --enable-chunked-prefill
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
  )
  [[ "$ENFORCE_EAGER" == 1 ]] && vllm_args+=(--enforce-eager)
  [[ -n "$KV_EVENTS_CONFIG" ]] && vllm_args+=(--kv-events-config "$KV_EVENTS_CONFIG")
  if [[ -n "$SAFETENSORS_LOAD_STRATEGY" ]]; then
    vllm_args+=(--safetensors-load-strategy "$SAFETENSORS_LOAD_STRATEGY")
  fi
  start_process backend env \
    CUDA_VISIBLE_DEVICES=0 \
    LMCACHE_CONFIG_FILE="$LMCACHE_CONFIG" \
    LMCACHE_WORKLOAD_AWARE_TRACE_PATH="$LOG_ROOT/serving/backend.connector-trace.jsonl" \
    LMCACHE_WORKLOAD_AWARE_ACTUAL_TRACE_PATH="$LOG_ROOT/serving/backend.connector-actual-trace.jsonl" \
    VLLM_SERVER_DEV_MODE=1 PYTHONHASHSEED=123 \
    "${vllm_args[@]}"
  wait_http backend http://127.0.0.1:8000/v1/models

  if grep -q 'enable_client_http_server: true' "$LMCACHE_CONFIG"; then
    wait_http backend "http://127.0.0.1:$MOONCAKE_CLIENT_METRICS_PORT/health"
  fi

  start_router "$ROUTER_CONFIG"
}

start_router() {
  local config="$1"
  stop_one "$PID_DIR/router.pid"
  start_process router "$ROUTER" \
    --host 127.0.0.1 --port 9003 \
    --service-discovery static \
    --static-backends http://127.0.0.1:8000 \
    --static-models "$MODEL_NAME" \
    --static-model-labels monolithic \
    --routing-logic agent_slo_aware \
    --engine-stats-interval 0.25 \
    --agent-slo-config "$config" \
    --agent-slo-trace-path "$LOG_ROOT/routing/router-trace.jsonl" \
    --session-key X-Session-ID
  wait_http router http://127.0.0.1:9003/health
}

reset_hbm() {
  local reset_external="${1:-false}"
  curl -fsS -X POST \
    "http://127.0.0.1:8000/reset_prefix_cache?reset_external=$reset_external" \
    >/dev/null
}

clear_l1() {
  local payload
  payload="{\"instance_id\":\"$LMCACHE_INSTANCE_ID\",\"location\":\"LocalCPUBackend\"}"
  curl -fsS -X POST -H 'Content-Type: application/json' -d "$payload" \
    http://127.0.0.1:9000/clear >/dev/null
}

case "$ACTION" in
  start) start_all ;;
  router) start_router "${2:?router config required}" ;;
  reset-hbm) reset_hbm "${2:-false}" ;;
  clear-l1) clear_l1 ;;
  stop) stop_all ;;
  status)
    for pid_file in "$PID_DIR"/*.pid; do
      [[ -e "$pid_file" ]] || continue
      pid="$(<"$pid_file")"
      kill -0 "$pid" 2>/dev/null && state=running || state=stopped
      echo "$(basename "$pid_file" .pid) pid=$pid $state"
    done
    ;;
  *) echo "usage: $0 start|stop|status|router CONFIG|reset-hbm [true|false]|clear-l1" >&2; exit 2 ;;
esac
