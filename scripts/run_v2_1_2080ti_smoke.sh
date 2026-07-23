#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
STACK="$ROOT/scripts/v2_1_2080ti_stack.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cleanup() {
  [[ "${KEEP_V2_1_STACK:-false}" == "true" ]] || "$STACK" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

"$STACK" start
/root/.venvs/kv-worker/bin/python -m benchmarks.smoke_v2_1_2080ti "$@"
