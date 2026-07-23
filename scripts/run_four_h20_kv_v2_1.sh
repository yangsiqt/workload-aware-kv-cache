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
RESULT_ROOT="${RESULT_ROOT:-/root/performance-results/workload-aware-kv-cache/four-h20/adaptive-kv-v2-1}"
TAG="${FOUR_H20_RUN_TAG:-v2-1-$(date -u +%Y%m%dT%H%M%SZ)}"
HARD_LIMIT_S="${FOUR_H20_HARD_LIMIT_S:-18000}"
STARTED_AT="$(date +%s)"
DEADLINE=$((STARTED_AT + HARD_LIMIT_S))
FIXED="$ROOT/configs/four_h20/agent-slo-kv-fixed-4096-v2-1.yaml"
ADAPTIVE="$ROOT/configs/four_h20/agent-slo-kv-adaptive-v2-1.yaml"
FINAL_RPS=""
FINAL_FIXED_RUN=""
FINAL_ADAPTIVE_RUN=""

remaining_seconds() {
  local remaining=$((DEADLINE - $(date +%s)))
  if ((remaining <= 0)); then
    echo "V2.1 cumulative five-hour hard limit reached" >&2
    return 1
  fi
  echo "$remaining"
}

run_stage() {
  if [[ "$DRY_RUN" == 1 ]]; then
    "$STAGE" --dry-run "$@"
  else
    local remaining
    remaining="$(remaining_seconds)"
    timeout --signal=TERM --kill-after=30s "${remaining}s" "$STAGE" "$@"
  fi
}

cleanup() {
  [[ "$DRY_RUN" == 1 ]] || "$STACK" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

trace_for_rps() {
  local rps="$1"
  echo "/root/workload-aware-kv-cache-data/traces/four_h20/${rps}rps/swe-final-poisson-${rps}rps-r2.jsonl"
}

run_formal_pair() {
  local rps="$1" trace fixed_run adaptive_run
  trace="$(trace_for_rps "$rps")"
  fixed_run="V21K04-fixed-before-${rps}rps-$TAG"
  adaptive_run="V21K05-adaptive-after-${rps}rps-$TAG"
  [[ "$DRY_RUN" == 1 || -f "$trace" ]] || {
    echo "missing frozen V2.1 arrival trace: $trace" >&2
    return 1
  }

  REQUIRE_V2_1_WORKER_LIFECYCLE=true run_stage \
    "$fixed_run" kv "$FIXED" "$DATA/swebench.jsonl" poisson 64 "$trace" || return 1
  REQUIRE_V2_1_WORKER_LIFECYCLE=true run_stage \
    "$adaptive_run" kv "$ADAPTIVE" "$DATA/swebench.jsonl" poisson 64 "$trace" || return 1

  FINAL_RPS="$rps"
  FINAL_FIXED_RUN="$fixed_run"
  FINAL_ADAPTIVE_RUN="$adaptive_run"
}

if [[ "$DRY_RUN" == 1 ]]; then
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-12}" "$STACK" --dry-run start-kv
else
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-12}" "$STACK" start-kv
fi

# Cold write -> clear only HBM and prove strict L1 -> clear HBM+L1 and prove L2.
MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true \
  REQUIRE_SELECTED_KV_PATHS=recompute run_stage \
  "V21K01A-cold-$TAG" kv "$ROOT/configs/four_h20/agent-slo-kv-recompute.yaml" \
  "$DATA/profiles/four_h20_smoke.jsonl" closed_loop 4

if [[ "$DRY_RUN" == 1 ]]; then
  RESET_EXTERNAL=false "$STACK" --dry-run reset
else
  RESET_EXTERNAL=false "$STACK" reset
fi
RESET_EXTERNAL=false MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true \
  REQUIRE_SELECTED_KV_PATHS=lmcache_l1 REQUIRE_ACTUAL_KV_PATHS=lmcache_l1 run_stage \
  "V21K01B-l1-$TAG" kv "$ROOT/configs/four_h20/agent-slo-kv-force-l1.yaml" \
  "$DATA/profiles/four_h20_smoke.jsonl" closed_loop 4

if [[ "$DRY_RUN" == 1 ]]; then
  "$STACK" --dry-run clear-l1
else
  "$STACK" clear-l1
fi
if [[ "$DRY_RUN" == 1 ]]; then
  RESET_EXTERNAL=false "$STACK" --dry-run reset
else
  RESET_EXTERNAL=false "$STACK" reset
fi
RESET_EXTERNAL=false MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true \
  REQUIRE_SELECTED_KV_PATHS=mooncake_l2 REQUIRE_ACTUAL_KV_PATHS=mooncake_l2 run_stage \
  "V21K01C-l2-$TAG" kv "$ROOT/configs/four_h20/agent-slo-kv-force-l2.yaml" \
  "$DATA/profiles/four_h20_smoke.jsonl" closed_loop 4

FORMAL_RPS="${FORMAL_RPS:-4}"
if ! run_formal_pair "$FORMAL_RPS"; then
  if [[ "$FORMAL_RPS" != 4 ]]; then
    echo "V2.1 formal pair failed at explicitly selected ${FORMAL_RPS} RPS" >&2
    exit 1
  fi
  echo "4 RPS pair failed; preserving it and restarting both sides on the same 2 RPS Trace" >&2
  run_formal_pair 2
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY-RUN: analyze $FINAL_FIXED_RUN and $FINAL_ADAPTIVE_RUN at $FINAL_RPS RPS"
else
  remaining_seconds >/dev/null
  python -m benchmarks.analyze_four_h20 \
    --run "fixed-4096=$RUN_ROOT/$FINAL_FIXED_RUN" \
    --run "adaptive-v2-1=$RUN_ROOT/$FINAL_ADAPTIVE_RUN" \
    --output-dir "$RESULT_ROOT/$TAG" \
    --title "Adaptive KV V2.1 Independent Trace (${FINAL_RPS} RPS)"
fi
