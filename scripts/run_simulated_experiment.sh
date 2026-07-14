#!/usr/bin/env bash
set -euo pipefail

WORKLOAD="${WORKLOAD:-/root/workload-aware-kv-cache-data/processed/small/controlled.jsonl}"
RUN_ROOT="${RUN_ROOT:-/root/workload-aware-kv-cache-data/runs}"
REPORT_DIR="${REPORT_DIR:-/root/workload-aware-kv-cache-data/reports/simulated-small}"
CONCURRENCY="${CONCURRENCY:-4}"

for policy in random prefix_affinity session_affinity; do
  run_id="simulated-${policy}-c${CONCURRENCY}"
  rm -rf "$RUN_ROOT/$run_id"
  curl --fail --silent -X POST http://127.0.0.1:9000/reset >/dev/null
  python -m benchmarks.run_benchmark "$WORKLOAD" \
    --mode closed_loop --concurrency "$CONCURRENCY" --route-policy "$policy" \
    --run-id "$run_id" --simulated
done

python -m benchmarks.compare_runs \
  "$RUN_ROOT/simulated-random-c${CONCURRENCY}" \
  "$RUN_ROOT/simulated-prefix_affinity-c${CONCURRENCY}" \
  "$RUN_ROOT/simulated-session_affinity-c${CONCURRENCY}" \
  --output-dir "$REPORT_DIR"
