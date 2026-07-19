#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKLOAD="${WORKLOAD:-/root/workload-aware-kv-cache-data/processed/small/controlled.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/workload-aware-kv-cache-data/runs}"
RESULT_ROOT="${RESULT_ROOT:-/root/performance-results/workload-aware-kv-cache/simulated}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
POLICIES="${POLICIES:-roundrobin,session,prefixaware,agent_slo_aware}"
CONCURRENCY="${CONCURRENCY:-2}"
BASE_LOG_DIR="${BASE_LOG_DIR:-/root/log/workload-aware-kv-cache/routing/policy-matrix-$RUN_TAG}"
STACK_PID=""
run_dirs=()

cleanup() {
  if [[ -n "$STACK_PID" ]] && kill -0 "$STACK_PID" 2>/dev/null; then
    kill -TERM "$STACK_PID" 2>/dev/null || true
    wait "$STACK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$BASE_LOG_DIR" "$RESULT_ROOT"
IFS=',' read -r -a policy_array <<<"$POLICIES"
for policy in "${policy_array[@]}"; do
  policy_log_dir="$BASE_LOG_DIR/$policy"
  trace_path="$policy_log_dir/router-trace.jsonl"
  pid_file="$policy_log_dir/router.pid"
  mkdir -p "$policy_log_dir"

  POLICY="$policy" LOG_DIR="$policy_log_dir" TRACE_PATH="$trace_path" \
    PID_FILE="$pid_file" "$ROOT/scripts/run_production_mock_stack.sh" \
    >"$policy_log_dir/stack.log" 2>&1 &
  STACK_PID=$!

  ready=false
  for _ in $(seq 1 120); do
    if curl --fail --silent http://127.0.0.1:9003/health >/dev/null 2>&1; then
      ready=true
      break
    fi
    if ! kill -0 "$STACK_PID" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if [[ "$ready" != true ]]; then
    echo "policy stack failed to start: $policy" >&2
    exit 1
  fi

  run_id="sim-production-${policy}-c${CONCURRENCY}-${RUN_TAG}"
  manifest_args=(
    --launch-command "POLICY=$policy scripts/run_production_mock_stack.sh"
  )
  if [[ "$policy" == "agent_slo_aware" ]]; then
    manifest_args+=(
      --router-config "$ROOT/configs/production_stack/agent-slo-pr3.yaml"
    )
  fi
  python -m benchmarks.run_benchmark "$WORKLOAD" \
    --config "$ROOT/configs/benchmark-production-mock.yaml" \
    --output-root "$OUTPUT_ROOT" \
    --mode closed_loop \
    --concurrency "$CONCURRENCY" \
    --route-policy "$policy" \
    --run-id "$run_id" \
    --simulated \
    "${manifest_args[@]}"
  run_dirs+=("$OUTPUT_ROOT/$run_id")

  kill -TERM "$STACK_PID" 2>/dev/null || true
  wait "$STACK_PID" 2>/dev/null || true
  STACK_PID=""
  for _ in $(seq 1 40); do
    remaining=false
    for url in \
      http://127.0.0.1:9003/health \
      http://127.0.0.1:9101/health \
      http://127.0.0.1:9102/health; do
      if curl --silent --max-time 0.2 "$url" >/dev/null 2>&1; then
        remaining=true
      fi
    done
    [[ "$remaining" == false ]] && break
    sleep 0.25
  done
  if [[ "$remaining" != false ]]; then
    echo "stack process remained after stop: $policy" >&2
    exit 1
  fi
done

comparison_dir="$RESULT_ROOT/${RUN_TAG}__production-router-policy-matrix__SIMULATED"
python -m benchmarks.compare_runs "${run_dirs[@]}" --output-dir "$comparison_dir"
printf '%s\n' "$comparison_dir"
