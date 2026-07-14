#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/root/log/workload-aware-kv-cache/routing}"
mkdir -p "$LOG_DIR"
pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

BACKEND_ID=mock-a PORT=9101 python -m router.mock_backend --port 9101 >"$LOG_DIR/mock-a.log" 2>&1 & pids+=("$!")
BACKEND_ID=mock-b PORT=9102 python -m router.mock_backend --port 9102 >"$LOG_DIR/mock-b.log" 2>&1 & pids+=("$!")
python -m router.app --port 9000 >"$LOG_DIR/router.log" 2>&1 & pids+=("$!")

for url in http://127.0.0.1:9101/health http://127.0.0.1:9102/health http://127.0.0.1:9000/health; do
  for _ in $(seq 1 40); do
    curl --fail --silent "$url" >/dev/null && break
    sleep 0.25
  done
  curl --fail --silent "$url" >/dev/null
done
echo "mock stack ready: router=http://127.0.0.1:9000"
wait
