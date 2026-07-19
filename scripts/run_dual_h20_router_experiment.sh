#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 RUN_ID POLICY WORKLOAD CONCURRENCY MODE [ARRIVAL_TRACE]" >&2
  exit 2
fi

RUN_ID="$1"
POLICY="$2"
WORKLOAD="$3"
CONCURRENCY="$4"
MODE="$5"
ARRIVAL_TRACE="${6:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
PRODUCTION_STACK_ROOT="${PRODUCTION_STACK_ROOT:-/root/production-stack}"
RUN_ROOT="${RUN_ROOT:-/root/workload-aware-kv-cache-data/runs}"
LOG_ROOT="${LOG_ROOT:-/root/log/workload-aware-kv-cache}"
ROUTER_CONFIG="${ROUTER_CONFIG:-$PROJECT_ROOT/configs/production_stack/agent-slo-dual-h20.yaml}"
MODEL="${MODEL:-Qwen3-30B-A3B-Instruct-2507}"
BACKENDS=("http://127.0.0.1:8000" "http://127.0.0.1:8001")
ROUTER_URL="http://127.0.0.1:9003"

mkdir -p "$LOG_ROOT/routing" "$LOG_ROOT/benchmark" "$RUN_ROOT"
ROUTER_LOG="$LOG_ROOT/routing/${RUN_ID}.router.log"
ROUTER_TRACE="$LOG_ROOT/routing/${RUN_ID}.trace.jsonl"
METRICS_LOG="$LOG_ROOT/benchmark/${RUN_ID}.metrics.jsonl"
router_pid=""
metrics_pid=""

cleanup() {
  if [[ -n "$metrics_pid" ]]; then
    kill -INT "$metrics_pid" 2>/dev/null || true
    wait "$metrics_pid" 2>/dev/null || true
  fi
  if [[ -n "$router_pid" ]]; then
    kill -INT "$router_pid" 2>/dev/null || true
    wait "$router_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  idle=1
  for backend in "${BACKENDS[@]}"; do
    values="$(curl -fsS "$backend/metrics" | awk '
      /^vllm:num_requests_running\{/ {running += $NF}
      /^vllm:num_requests_waiting\{/ {waiting += $NF}
      END {print running + waiting}')"
    if [[ "$values" != "0" ]]; then
      idle=0
    fi
  done
  [[ "$idle" == "1" ]] && break
  sleep 0.25
done
[[ "$idle" == "1" ]] || { echo "backends did not become idle" >&2; exit 1; }

for backend in "${BACKENDS[@]}"; do
  curl -fsS -X POST "$backend/reset_prefix_cache" >/dev/null
done

router_cmd=(
  /root/.venvs/vllm-router/bin/vllm-router
  --host 127.0.0.1 --port 9003
  --service-discovery static
  --static-backends "${BACKENDS[0]},${BACKENDS[1]}"
  --static-models "$MODEL,$MODEL"
  --routing-logic "$POLICY"
  --engine-stats-interval 0.25
  --max-instance-failover-reroute-attempts 1
)
if [[ "$POLICY" == "prefixaware" ]]; then
  router_cmd+=(--prefix-min-match-length 128)
elif [[ "$POLICY" == "agent_slo_aware" ]]; then
  router_cmd+=(
    --agent-slo-config "$ROUTER_CONFIG"
    --agent-slo-trace-path "$ROUTER_TRACE"
    --session-key X-Session-ID
  )
fi

(
  cd "$PRODUCTION_STACK_ROOT"
  exec "${router_cmd[@]}"
) >"$ROUTER_LOG" 2>&1 &
router_pid=$!

for _ in $(seq 1 40); do
  curl -fsS "$ROUTER_URL/health" >/dev/null 2>&1 && break
  kill -0 "$router_pid" 2>/dev/null || { tail -50 "$ROUTER_LOG" >&2; exit 1; }
  sleep 0.25
done
curl -fsS "$ROUTER_URL/health" >/dev/null

(
  cd "$PROJECT_ROOT"
  exec python -m benchmarks.sample_backend_metrics \
    --backend "gpu0=${BACKENDS[0]}" \
    --backend "gpu1=${BACKENDS[1]}" \
    --output "$METRICS_LOG" --interval 0.25
) &
metrics_pid=$!

benchmark_cmd=(
  python -m benchmarks.run_benchmark
  --config configs/benchmark-dual-h20-router.yaml
  --mode "$MODE" --concurrency "$CONCURRENCY"
  --route-policy "$POLICY" --run-id "$RUN_ID"
  --output-root "$RUN_ROOT"
  --launch-command "$0 $*"
)
if [[ "$POLICY" == "agent_slo_aware" ]]; then
  benchmark_cmd+=(--router-config "$ROUTER_CONFIG")
fi
if [[ -n "$ARRIVAL_TRACE" ]]; then
  benchmark_cmd+=(--arrival-trace "$ARRIVAL_TRACE")
fi
benchmark_cmd+=("$WORKLOAD")

(
  cd "$PROJECT_ROOT"
  "${benchmark_cmd[@]}"
)

cleanup
trap - EXIT
router_pid=""
metrics_pid=""

python -m benchmarks.analyze_dual_h20 \
  --requests "$RUN_ROOT/$RUN_ID/requests.jsonl" \
  --metrics "$METRICS_LOG" \
  --output "$RUN_ROOT/$RUN_ID/dual_metrics.json"
echo "$RUN_ROOT/$RUN_ID"
