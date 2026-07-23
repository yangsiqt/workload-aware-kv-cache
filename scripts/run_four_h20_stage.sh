#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 [--dry-run] RUN_ID TOPOLOGY ROUTER_CONFIG WORKLOAD MODE CONCURRENCY [ARRIVAL_TRACE]" >&2
  exit 2
fi

RUN_ID="$1"
TOPOLOGY="$2"
ROUTER_CONFIG="$3"
WORKLOAD="$4"
MODE="$5"
CONCURRENCY="$6"
ARRIVAL_TRACE="${7:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
RUN_ROOT="${RUN_ROOT:-/root/workload-aware-kv-cache-data/runs/four_h20}"
LOG_ROOT="${FOUR_H20_LOG_ROOT:-/root/log/workload-aware-kv-cache/four-h20}"
STACK="$PROJECT_ROOT/scripts/four_h20_stack.sh"
TRACE="$LOG_ROOT/routing/$RUN_ID.trace.jsonl"
METRICS="$LOG_ROOT/benchmark/$RUN_ID.metrics.jsonl"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.99}"
REQUIRE_SELECTED_KV_PATHS="${REQUIRE_SELECTED_KV_PATHS:-}"
REQUIRE_ACTUAL_KV_PATHS="${REQUIRE_ACTUAL_KV_PATHS:-}"
REQUIRE_EXECUTION_MODES="${REQUIRE_EXECUTION_MODES:-}"
REQUIRE_V2_1_WORKER_LIFECYCLE="${REQUIRE_V2_1_WORKER_LIFECYCLE:-false}"
metrics_pid=""
declare -a connector_offsets actual_offsets

join_command() {
  local -a command=(
    python -m benchmarks.join_traces
    "$RUN_ROOT/$RUN_ID/requests.jsonl" "$TRACE"
    "$RUN_ROOT/$RUN_ID/joined_trace.jsonl"
  )
  if [[ "$REQUIRE_V2_1_WORKER_LIFECYCLE" == "true" ]]; then
    for gpu in 0 1 2 3; do
      command+=(
        --worker-trace "$RUN_ROOT/$RUN_ID/connector_trace_gpu$gpu.jsonl"
        --worker-trace "$RUN_ROOT/$RUN_ID/connector_actual_trace_gpu$gpu.jsonl"
      )
    done
  fi
  "${command[@]}"
}

validate_command() {
  local -a command=(
    python -m benchmarks.validate_four_h20_run "$RUN_ROOT/$RUN_ID" "$WORKLOAD"
    --min-success-rate "$MIN_SUCCESS_RATE"
    --require-selected-kv-paths "$REQUIRE_SELECTED_KV_PATHS"
    --require-actual-kv-paths "$REQUIRE_ACTUAL_KV_PATHS"
    --require-execution-modes "$REQUIRE_EXECUTION_MODES"
  )
  if [[ "$REQUIRE_V2_1_WORKER_LIFECYCLE" == "true" ]]; then
    command+=(--require-v2-1-worker-lifecycle)
  fi
  "${command[@]}"
}

print_command() {
  printf 'DRY-RUN:'
  printf ' %q' "$@"
  printf '\n'
}

cleanup() {
  if [[ -n "$metrics_pid" ]] && kill -0 "$metrics_pid" 2>/dev/null; then
    kill -TERM "$metrics_pid" 2>/dev/null || true
    wait "$metrics_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

benchmark=(
  python -m benchmarks.run_benchmark "$WORKLOAD"
  --config "$PROJECT_ROOT/configs/benchmark-four-h20-router.yaml"
  --output-root "$RUN_ROOT"
  --mode "$MODE"
  --concurrency "$CONCURRENCY"
  --route-policy agent_slo_aware
  --run-id "$RUN_ID"
  --router-config "$ROUTER_CONFIG"
  --launch-command "$0 $*"
)
if [[ -n "$ARRIVAL_TRACE" ]]; then
  benchmark+=(--arrival-trace "$ARRIVAL_TRACE")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  FOUR_H20_ROUTER_TRACE="$TRACE" "$STACK" --dry-run router "$ROUTER_CONFIG" "$TOPOLOGY"
  "$STACK" --dry-run reset
  print_command python -m benchmarks.sample_backend_metrics --output "$METRICS" --interval 0.25 \
    --backend gpu0=http://127.0.0.1:8000 --backend gpu1=http://127.0.0.1:8001 \
    --backend gpu2=http://127.0.0.1:8002 --backend gpu3=http://127.0.0.1:8003
  print_command "${benchmark[@]}"
  print_command join_command
  print_command validate_command
  exit 0
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT/routing" "$LOG_ROOT/benchmark"
if [[ -d "$RUN_ROOT/$RUN_ID" ]]; then
  mv "$RUN_ROOT/$RUN_ID" "$RUN_ROOT/$RUN_ID.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
fi
rm -f "$TRACE" "$METRICS"
FOUR_H20_ROUTER_TRACE="$TRACE" "$STACK" router "$ROUTER_CONFIG" "$TOPOLOGY"
"$STACK" reset
for gpu in 0 1 2 3; do
  connector_trace="$LOG_ROOT/serving/backend-$gpu.connector-trace.jsonl"
  actual_trace="$LOG_ROOT/serving/backend-$gpu.connector-actual-trace.jsonl"
  connector_offsets[$gpu]="$(stat -c %s "$connector_trace" 2>/dev/null || echo 0)"
  actual_offsets[$gpu]="$(stat -c %s "$actual_trace" 2>/dev/null || echo 0)"
done

python -m benchmarks.sample_backend_metrics \
  --backend gpu0=http://127.0.0.1:8000 \
  --backend gpu1=http://127.0.0.1:8001 \
  --backend gpu2=http://127.0.0.1:8002 \
  --backend gpu3=http://127.0.0.1:8003 \
  --output "$METRICS" --interval 0.25 &
metrics_pid=$!

(cd "$PROJECT_ROOT" && "${benchmark[@]}")
cleanup
metrics_pid=""
sleep 1
for gpu in 0 1 2 3; do
  connector_trace="$LOG_ROOT/serving/backend-$gpu.connector-trace.jsonl"
  actual_trace="$LOG_ROOT/serving/backend-$gpu.connector-actual-trace.jsonl"
  if [[ -f "$connector_trace" ]]; then
    tail -c "+$(( connector_offsets[$gpu] + 1 ))" "$connector_trace" \
      >"$RUN_ROOT/$RUN_ID/connector_trace_gpu$gpu.jsonl"
  fi
  if [[ -f "$actual_trace" ]]; then
    tail -c "+$(( actual_offsets[$gpu] + 1 ))" "$actual_trace" \
      >"$RUN_ROOT/$RUN_ID/connector_actual_trace_gpu$gpu.jsonl"
  fi
done
for _ in $(seq 1 40); do
  if join_command >/dev/null 2>&1; then
    cp "$TRACE" "$RUN_ROOT/$RUN_ID/router_trace.jsonl"
    cp "$METRICS" "$RUN_ROOT/$RUN_ID/backend_metrics.jsonl"
    validate_command
    echo "$RUN_ROOT/$RUN_ID"
    exit 0
  fi
  sleep 0.25
done
join_command
cp "$TRACE" "$RUN_ROOT/$RUN_ID/router_trace.jsonl"
cp "$METRICS" "$RUN_ROOT/$RUN_ID/backend_metrics.jsonl"
validate_command
