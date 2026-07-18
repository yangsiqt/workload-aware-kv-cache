#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-/root/log/workload-aware-kv-cache/routing}"
mkdir -p "$LOG_DIR"
pids=()

cleanup() {
  "$ROOT/scripts/stop_production_router.sh" || true
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

cd "$ROOT"
BACKEND_ID=mock-a PORT=9101 python -m router.mock_backend --port 9101 \
  >"$LOG_DIR/production-mock-a.log" 2>&1 &
pids+=("$!")
BACKEND_ID=mock-b PORT=9102 python -m router.mock_backend --port 9102 \
  >"$LOG_DIR/production-mock-b.log" 2>&1 &
pids+=("$!")

for url in http://127.0.0.1:9101/health http://127.0.0.1:9102/health; do
  for _ in $(seq 1 40); do
    curl --fail --silent "$url" >/dev/null 2>&1 && break
    sleep 0.25
  done
  curl --fail --silent "$url" >/dev/null
done

"$ROOT/scripts/start_production_router.sh"
echo "production mock stack ready: http://127.0.0.1:9003"
wait
