#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
LOG="${READINESS_LOG:-/root/log/workload-aware-kv-cache/four-h20/readiness.log}"
mkdir -p "$(dirname "$LOG")"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec > >(tee "$LOG") 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ'
"$ROOT/scripts/verify_four_h20_environment.sh"
bash -n \
  "$ROOT/scripts/four_h20_stack.sh" \
  "$ROOT/scripts/run_four_h20_stage.sh" \
  "$ROOT/scripts/run_four_h20_kv_window.sh" \
  "$ROOT/scripts/run_four_h20_pd_window.sh"
FOUR_H20_RUN_TAG=readiness "$ROOT/scripts/run_four_h20_kv_window.sh" --dry-run >/tmp/four-h20-kv-readiness.log
FOUR_H20_RUN_TAG=readiness "$ROOT/scripts/run_four_h20_pd_window.sh" --dry-run >/tmp/four-h20-pd-readiness.log
python -m benchmarks.check_four_h20_readiness
