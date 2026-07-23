#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; shift; }

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
STACK="$ROOT/scripts/four_h20_stack.sh"
STAGE="$ROOT/scripts/run_four_h20_stage.sh"
DATA="${FOUR_H20_DATA_ROOT:-/root/workload-aware-kv-cache-data/processed/four_h20}"
RUN_ROOT="${RUN_ROOT:-/root/workload-aware-kv-cache-data/runs/four_h20}"
RESULT_ROOT="${RESULT_ROOT:-/root/performance-results/workload-aware-kv-cache/four-h20/adaptive-kv-v2}"
TAG="${FOUR_H20_RUN_TAG:-v2-$(date -u +%Y%m%dT%H%M%SZ)}"
RPS="${FORMAL_RPS:-4}"
TRACE="/root/workload-aware-kv-cache-data/traces/four_h20/${RPS}rps/swe-final-poisson-${RPS}rps-r2.jsonl"
FIXED="$ROOT/configs/four_h20/agent-slo-kv-fixed-4096.yaml"
ADAPTIVE="$ROOT/configs/four_h20/agent-slo-kv-adaptive-v2.yaml"
HARD_LIMIT_S="${FOUR_H20_HARD_LIMIT_S:-18000}"

[[ "$DRY_RUN" == 1 || -f "$TRACE" ]] || {
  echo "missing frozen independent arrival trace: $TRACE" >&2
  exit 1
}

run_stage() {
  if [[ "$DRY_RUN" == 1 ]]; then
    "$STAGE" --dry-run "$@"
  else
    timeout --signal=TERM --kill-after=30s "${HARD_LIMIT_S}s" "$STAGE" "$@"
  fi
}

cleanup() {
  [[ "$DRY_RUN" == 1 ]] || "$STACK" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ "$DRY_RUN" == 1 ]]; then
  "$STACK" --dry-run start-kv
else
  "$STACK" start-kv
fi

MIN_SUCCESS_RATE=1 run_stage \
  "V2K01-smoke-$TAG" kv "$ADAPTIVE" \
  "$DATA/profiles/four_h20_smoke.jsonl" closed_loop 4

run_stage \
  "V2K02-fixed-before-$TAG" kv "$FIXED" \
  "$DATA/swebench.jsonl" poisson 64 "$TRACE"

run_stage \
  "V2K03-adaptive-after-$TAG" kv "$ADAPTIVE" \
  "$DATA/swebench.jsonl" poisson 64 "$TRACE"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY-RUN: analyze V2K02/V2K03 into $RESULT_ROOT/$TAG"
else
  python -m benchmarks.analyze_four_h20 \
    --run "fixed-4096=$RUN_ROOT/V2K02-fixed-before-$TAG" \
    --run "adaptive-v2=$RUN_ROOT/V2K03-adaptive-after-$TAG" \
    --output-dir "$RESULT_ROOT/$TAG" \
    --title "Adaptive KV V2 Independent Trace"
fi
