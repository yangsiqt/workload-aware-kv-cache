#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507}"
MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B-Instruct-2507}"
PYTHON_BIN="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
ROUTER_BIN="${ROUTER_BIN:-/root/.venvs/vllm-router/bin/vllm-router}"
MOONCAKE_MASTER_BIN="${MOONCAKE_MASTER_BIN:-/root/.venvs/kv-worker/lib/python3.12/site-packages/mooncake/mooncake_master}"
LOG_ROOT="${FOUR_H20_LOG_ROOT:-/root/log/workload-aware-kv-cache/four-h20}"
PID_DIR="$LOG_ROOT/pids"
START_TIMEOUT="${FOUR_H20_START_TIMEOUT:-720}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MOONCAKE_MASTER_METRICS_PORT="${MOONCAKE_MASTER_METRICS_PORT:-9004}"
LMCACHE_HASH_SEED="${LMCACHE_HASH_SEED:-123}"
DRY_RUN=0

usage() {
  echo "usage: $0 [--dry-run] start-kv|start-pd|router CONFIG|reset|clear-l1|status|stop" >&2
}

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
ACTION="${1:-}"
shift || true

mkdir -p "$PID_DIR" "$LOG_ROOT/serving" "$LOG_ROOT/routing" "$LOG_ROOT/components"
source "$PROJECT_ROOT/scripts/activate_four_h20_env.sh"

render() {
  printf 'DRY-RUN:'
  printf ' %q' "$@"
  printf '\n'
}

start_process() {
  local name="$1"
  shift
  local pid_file="$PID_DIR/$name.pid"
  local log_file="$LOG_ROOT/components/$name.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    render "$@"
    return
  fi
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    echo "$name already running" >&2
    return 1
  fi
  nohup "$@" >"$log_file" 2>&1 &
  echo "$!" >"$pid_file"
}

wait_http() {
  local name="$1" url="$2"
  [[ "$DRY_RUN" == "1" ]] && return
  local pid_file="$PID_DIR/$name.pid"
  for ((i=0; i<START_TIMEOUT; i++)); do
    curl -fsS "$url" >/dev/null 2>&1 && return
    if [[ -f "$pid_file" ]] && ! kill -0 "$(<"$pid_file")" 2>/dev/null; then
      tail -80 "$LOG_ROOT/components/$name.log" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "timeout waiting for $name at $url" >&2
  return 1
}

wait_port() {
  local name="$1" host="$2" port="$3"
  [[ "$DRY_RUN" == "1" ]] && return
  local pid_file="$PID_DIR/$name.pid"
  for ((i=0; i<START_TIMEOUT; i++)); do
    if "$PYTHON_BIN" - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.5):
    pass
PY
    then
      return
    fi
    if [[ -f "$pid_file" ]] && ! kill -0 "$(<"$pid_file")" 2>/dev/null; then
      tail -80 "$LOG_ROOT/components/$name.log" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "timeout waiting for $name at $host:$port" >&2
  return 1
}

check_preflight() {
  test -d "$MODEL_PATH"
  test -x "$PYTHON_BIN"
  test -x "$ROUTER_BIN"
  test -x /root/.venvs/kv-worker/bin/lmcache_controller
  test -x "$MOONCAKE_MASTER_BIN"
  test -x /root/.venvs/kv-worker/bin/mooncake_http_metadata_server
  "$PROJECT_ROOT/scripts/verify_four_h20_environment.sh"
  if [[ "$DRY_RUN" == "0" ]]; then
    [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 4 ]]
    local available_kb
    available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    [[ "$available_kb" -ge 12582912 ]]
  fi
}

start_mooncake_and_controller() {
  start_process mooncake-master "$MOONCAKE_MASTER_BIN" -v=1 \
    --metrics_port="$MOONCAKE_MASTER_METRICS_PORT"
  start_process mooncake-metadata /root/.venvs/kv-worker/bin/mooncake_http_metadata_server --port 8005
  start_process lmcache-controller env "PYTHONHASHSEED=$LMCACHE_HASH_SEED" \
    /root/.venvs/kv-worker/bin/lmcache_controller \
    --host 127.0.0.1 --port 9000 \
    --monitor-ports '{"pull":9101,"reply":9102,"heartbeat":9103}'
  wait_port mooncake-master 127.0.0.1 50051
  wait_port mooncake-metadata 127.0.0.1 8005
  wait_http lmcache-controller http://127.0.0.1:9000/
}

vllm_base=(
  "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL_PATH"
  --served-model-name "$MODEL_NAME"
  --host 127.0.0.1
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --enable-chunked-prefill
)

start_kv() {
  check_preflight
  if [[ "$DRY_RUN" == "0" ]]; then
    for gpu in 0 1 2 3; do
      local trace actual_trace
      trace="$LOG_ROOT/serving/backend-$gpu.connector-trace.jsonl"
      actual_trace="$LOG_ROOT/serving/backend-$gpu.connector-actual-trace.jsonl"
      for candidate in "$trace" "$actual_trace"; do
        if [[ -f "$candidate" ]]; then
          mv "$candidate" "$candidate.previous.$(date -u +%Y%m%dT%H%M%SZ)"
        fi
      done
    done
  fi
  start_mooncake_and_controller
  for gpu in 0 1 2 3; do
    local port=$((8000 + gpu)) kv_event_port=$((9400 + gpu)) kv_events_config
    kv_events_config="{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$kv_event_port\",\"topic\":\"workload-aware-kv\"}"
    start_process "backend-$gpu" env \
      "CUDA_VISIBLE_DEVICES=$gpu" \
      "LMCACHE_CONFIG_FILE=$PROJECT_ROOT/configs/four_h20/lmcache-backend-$gpu.yaml" \
      "LMCACHE_WORKLOAD_AWARE_TRACE_PATH=$LOG_ROOT/serving/backend-$gpu.connector-trace.jsonl" \
      "LMCACHE_WORKLOAD_AWARE_ACTUAL_TRACE_PATH=$LOG_ROOT/serving/backend-$gpu.connector-actual-trace.jsonl" \
      VLLM_SERVER_DEV_MODE=1 \
      "PYTHONHASHSEED=$LMCACHE_HASH_SEED" \
      "${vllm_base[@]}" --port "$port" \
      --kv-events-config "$kv_events_config" \
      --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
    wait_http "backend-$gpu" "http://127.0.0.1:$port/v1/models"
    wait_http "backend-$gpu" "http://127.0.0.1:$((9300 + gpu))/health"
  done
  start_router "$PROJECT_ROOT/configs/four_h20/agent-slo-kv-adaptive.yaml" kv
}

start_pd() {
  check_preflight
  for gpu in 0 1; do
    local port=$((8000 + gpu))
    start_process "backend-$gpu" env "CUDA_VISIBLE_DEVICES=$gpu" VLLM_SERVER_DEV_MODE=1 \
      "${vllm_base[@]}" --port "$port"
    wait_http "backend-$gpu" "http://127.0.0.1:$port/v1/models"
  done
  start_process backend-2 env CUDA_VISIBLE_DEVICES=2 VLLM_SERVER_DEV_MODE=1 VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
    "${vllm_base[@]}" --port 8002 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"tcp"}}'
  wait_http backend-2 http://127.0.0.1:8002/v1/models
  start_process backend-3 env CUDA_VISIBLE_DEVICES=3 VLLM_SERVER_DEV_MODE=1 \
    "${vllm_base[@]}" --port 8003 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"tcp"}}'
  wait_http backend-3 http://127.0.0.1:8003/v1/models
  start_router "$PROJECT_ROOT/configs/four_h20/agent-slo-pd-adaptive.yaml" pd
}

stop_one() {
  local pid_file="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    render stop-pid-file "$pid_file"
    return 0
  fi
  if [[ ! -f "$pid_file" ]]; then
    return 0
  fi
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

start_router() {
  local config="$1" topology="${2:-kv}"
  stop_one "$PID_DIR/router.pid"
  local labels="monolithic,monolithic,monolithic,monolithic"
  [[ "$topology" == "pd" ]] && labels="monolithic,monolithic,prefill,decode"
  start_process router "$ROUTER_BIN" \
    --host 127.0.0.1 --port 9003 \
    --service-discovery static \
    --static-backends http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003 \
    --static-models "$MODEL_NAME,$MODEL_NAME,$MODEL_NAME,$MODEL_NAME" \
    --static-model-labels "$labels" \
    --routing-logic agent_slo_aware \
    --engine-stats-interval 0.25 \
    --max-instance-failover-reroute-attempts 1 \
    --agent-slo-config "$config" \
    --agent-slo-trace-path "${FOUR_H20_ROUTER_TRACE:-$LOG_ROOT/routing/router-trace.jsonl}" \
    --session-key X-Session-ID
  wait_http router http://127.0.0.1:9003/health
}

reset_caches() {
  local reset_external="${RESET_EXTERNAL:-true}"
  for port in 8000 8001 8002 8003; do
    if [[ "$DRY_RUN" == "1" ]]; then
      render curl -fsS -X POST "http://127.0.0.1:$port/reset_prefix_cache?reset_external=$reset_external"
    else
      curl -fsS -X POST "http://127.0.0.1:$port/reset_prefix_cache?reset_external=$reset_external" >/dev/null
    fi
  done
}

clear_l1() {
  for gpu in 0 1 2 3; do
    local payload
    payload="{\"instance_id\":\"four-h20-backend-$gpu\",\"location\":\"LocalCPUBackend\"}"
    if [[ "$DRY_RUN" == "1" ]]; then
      render curl -fsS -X POST -H Content-Type:application/json -d "$payload" http://127.0.0.1:9000/clear
    else
      curl -fsS -X POST -H 'Content-Type: application/json' -d "$payload" http://127.0.0.1:9000/clear >/dev/null
    fi
  done
}

stop_all() {
  for name in router backend-3 backend-2 backend-1 backend-0 lmcache-controller mooncake-metadata mooncake-master; do
    stop_one "$PID_DIR/$name.pid"
  done
}

case "$ACTION" in
  start-kv) start_kv ;;
  start-pd) start_pd ;;
  router) start_router "${1:?router config required}" "${2:-kv}" ;;
  reset) reset_caches ;;
  clear-l1) clear_l1 ;;
  status)
    for file in "$PID_DIR"/*.pid; do
      [[ -e "$file" ]] || continue
      pid="$(<"$file")"
      kill -0 "$pid" 2>/dev/null && state=running || state=stopped
      echo "$(basename "$file" .pid) pid=$pid $state"
    done
    ;;
  stop) stop_all ;;
  *) usage; exit 2 ;;
esac
