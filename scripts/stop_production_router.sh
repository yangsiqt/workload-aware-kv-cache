#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${PID_FILE:-/root/log/workload-aware-kv-cache/routing/production-router.pid}"
if [[ ! -f "$PID_FILE" ]]; then
  exit 0
fi
pid="$(cat "$PID_FILE")"
kill "$pid" 2>/dev/null || true
for _ in $(seq 1 40); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.25
done
kill -9 "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
