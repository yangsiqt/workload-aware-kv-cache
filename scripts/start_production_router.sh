#!/usr/bin/env bash
set -euo pipefail

ROUTER_BIN="${ROUTER_BIN:-/root/.venvs/vllm-router/bin/vllm-router}"
PORT="${PORT:-9003}"
BACKENDS="${BACKENDS:-http://127.0.0.1:9101,http://127.0.0.1:9102}"
MODEL="${MODEL:-Qwen3-30B-A3B-Instruct-2507}"
POLICY="${POLICY:-agent_slo_aware}"
SESSION_KEY="${SESSION_KEY:-X-Session-ID}"
PREFIX_MIN_MATCH_LENGTH="${PREFIX_MIN_MATCH_LENGTH:-128}"
POLICY_CONFIG="${POLICY_CONFIG:-/root/workload-aware-kv-cache/configs/production_stack/agent-slo-pr3.yaml}"
LOG_DIR="${LOG_DIR:-/root/log/workload-aware-kv-cache/routing}"
TRACE_PATH="${TRACE_PATH:-$LOG_DIR/production-router-trace.jsonl}"
PID_FILE="${PID_FILE:-$LOG_DIR/production-router.pid}"

mkdir -p "$LOG_DIR"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "production router already running: $(cat "$PID_FILE")" >&2
  exit 1
fi

IFS=',' read -r -a backend_array <<<"$BACKENDS"
models=""
for _ in "${backend_array[@]}"; do
  models+="${models:+,}$MODEL"
done

args=(
  --host 127.0.0.1
  --port "$PORT"
  --service-discovery static
  --static-backends "$BACKENDS"
  --static-models "$models"
  --routing-logic "$POLICY"
  --engine-stats-interval 0.25
  --max-instance-failover-reroute-attempts 1
)
if [[ "$POLICY" == "agent_slo_aware" ]]; then
  args+=(
    --agent-slo-config "$POLICY_CONFIG"
    --agent-slo-trace-path "$TRACE_PATH"
    --session-key "$SESSION_KEY"
  )
fi
if [[ "$POLICY" == "session" ]]; then
  args+=(--session-key "$SESSION_KEY")
fi
if [[ "$POLICY" == "prefixaware" ]]; then
  args+=(--prefix-min-match-length "$PREFIX_MIN_MATCH_LENGTH")
fi

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nohup "$ROUTER_BIN" "${args[@]}" >"$LOG_DIR/production-router.log" 2>&1 &
echo $! >"$PID_FILE"

for _ in $(seq 1 80); do
  curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 0.25
done
curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null
echo "production router ready: http://127.0.0.1:$PORT pid=$(cat "$PID_FILE")"
