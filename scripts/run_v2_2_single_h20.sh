#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec /root/.venvs/kv-worker/bin/python -m benchmarks.single_h20_v2_2 "$@"
