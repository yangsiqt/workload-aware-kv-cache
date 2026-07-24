#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT/scripts/activate_four_h20_env.sh"

if [[ "${1:-}" == "--dry-run" ]]; then
  python -m benchmarks.single_h20_v2_1 --dry-run "${@:2}"
else
  python -m benchmarks.single_h20_v2_1 "$@"
fi
