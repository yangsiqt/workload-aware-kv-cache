#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
FROM="K01"
RESUME=0
RUN_TAG_ARG=""
SKIP_FAILURE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --from) FROM="${2:?stage required}"; shift ;;
    --resume) RESUME=1 ;;
    --run-tag) RUN_TAG_ARG="${2:?run tag required}"; shift ;;
    --include-failure) SKIP_FAILURE=0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
STACK="$ROOT/scripts/four_h20_stack.sh"
STAGE="$ROOT/scripts/run_four_h20_stage.sh"
DATA="${FOUR_H20_DATA_ROOT:-/root/workload-aware-kv-cache-data/processed/four_h20}"
RUN_ROOT="${RUN_ROOT:-/root/workload-aware-kv-cache-data/runs/four_h20}"
RESULT_ROOT="${RESULT_ROOT:-/root/performance-results/workload-aware-kv-cache/four-h20/adaptive-kv}"
if [[ -n "$RUN_TAG_ARG" ]]; then
  TAG="$RUN_TAG_ARG"
elif [[ -n "${FOUR_H20_RUN_TAG:-}" ]]; then
  TAG="$FOUR_H20_RUN_TAG"
elif [[ "$RESUME" == 1 ]]; then
  latest_state="$(find "$RUN_ROOT" -maxdepth 1 -type f -name 'kv-window-*.state' -printf '%T@ %f\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  [[ -n "$latest_state" ]] || { echo "--resume requested but no KV state file exists" >&2; exit 1; }
  TAG="${latest_state#kv-window-}"
  TAG="${TAG%.state}"
else
  TAG="$(date -u +%Y%m%dT%H%M%SZ)"
fi
RPS="${FORMAL_RPS:-4}"
TRACE="/root/workload-aware-kv-cache-data/traces/four_h20/${RPS}rps/swe-final-poisson-${RPS}rps-r1.jsonl"
STATE="$RUN_ROOT/kv-window-$TAG.state"
FROZEN="$RUN_ROOT/kv-window-$TAG.fixed.yaml"
ADAPTIVE_FROZEN="$RUN_ROOT/kv-window-$TAG.adaptive.yaml"
START_EPOCH="$(date +%s)"
HARD_LIMIT_S="${FOUR_H20_HARD_LIMIT_S:-18000}"
SOFT_LIMIT_S="${FOUR_H20_SOFT_LIMIT_S:-14400}"
stages=(K01 K02 K03 K04 K05 K06 K07)

index_of() { local i; for i in "${!stages[@]}"; do [[ "${stages[$i]}" == "$1" ]] && { echo "$i"; return; }; done; return 1; }
from_index="$(index_of "$FROM")" || { echo "invalid --from stage: $FROM" >&2; exit 2; }
completed() { [[ -f "$STATE" ]] && grep -qx "$1" "$STATE"; }
mark() { [[ "$DRY_RUN" == 1 ]] || echo "$1" >>"$STATE"; }
should_run() { local idx; idx="$(index_of "$1")"; (( idx >= from_index )) && { [[ "$RESUME" == 0 ]] || ! completed "$1"; }; }
run_stage() {
  if [[ "$DRY_RUN" == 1 ]]; then
    "$STAGE" --dry-run "$@"
  else
    local elapsed remaining
    elapsed=$(( $(date +%s) - START_EPOCH ))
    remaining=$(( HARD_LIMIT_S - elapsed ))
    (( remaining > 0 )) || { echo "four-H20 KV hard time limit reached" >&2; return 124; }
    timeout --signal=TERM --kill-after=30s "${remaining}s" "$STAGE" "$@"
  fi
}
past_soft_limit() { (( $(date +%s) - START_EPOCH >= SOFT_LIMIT_S )); }
start_stack() { if [[ "$DRY_RUN" == 1 ]]; then "$STACK" --dry-run start-kv; else "$STACK" start-kv; fi; }
ensure_stack() {
  local healthy=1 gpu pid_file
  for gpu in 0 1 2 3; do
    pid_file="/root/log/workload-aware-kv-cache/four-h20/pids/backend-$gpu.pid"
    [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null || healthy=0
  done
  if [[ "$DRY_RUN" == 1 || "$healthy" == 0 ]]; then
    [[ "$DRY_RUN" == 1 ]] || "$STACK" stop
    start_stack
  fi
}

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"
trap '[[ "$DRY_RUN" == 1 ]] || "$STACK" stop >/dev/null 2>&1 || true' EXIT INT TERM

if (( from_index > 0 && from_index < 6 )); then
  ensure_stack
fi

if should_run K01; then
  start_stack
  MIN_SUCCESS_RATE=1 run_stage "K01-smoke-$TAG" kv "$ROOT/configs/four_h20/agent-slo-kv-adaptive.yaml" "$DATA/profiles/four_h20_smoke.jsonl" closed_loop 4
  mark K01
fi
if should_run K02; then
  recompute_id="K02-recompute-$TAG"
  l1_id="K02-force-l1-$TAG"
  l2_id="K02-force-l2-$TAG"
  MIN_SUCCESS_RATE=1 REQUIRE_SELECTED_KV_PATHS=recompute run_stage "$recompute_id" kv "$ROOT/configs/four_h20/agent-slo-kv-recompute.yaml" "$DATA/profiles/kv_cost_controlled.jsonl" closed_loop 1
  MIN_SUCCESS_RATE=1 REQUIRE_SELECTED_KV_PATHS=lmcache_l1 REQUIRE_ACTUAL_KV_PATHS=lmcache_l1 RESET_EXTERNAL=false run_stage "$l1_id" kv "$ROOT/configs/four_h20/agent-slo-kv-force-l1.yaml" "$DATA/profiles/kv_cost_controlled.jsonl" closed_loop 1
  if [[ "$DRY_RUN" == 1 ]]; then "$STACK" --dry-run clear-l1; else "$STACK" clear-l1; fi
  [[ "$DRY_RUN" == 1 ]] || sleep 2
  MIN_SUCCESS_RATE=1 REQUIRE_SELECTED_KV_PATHS=mooncake_l2 REQUIRE_ACTUAL_KV_PATHS=mooncake_l2 RESET_EXTERNAL=false run_stage "$l2_id" kv "$ROOT/configs/four_h20/agent-slo-kv-force-l2.yaml" "$DATA/profiles/kv_cost_controlled.jsonl" closed_loop 1
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: fit measured KV costs to $ADAPTIVE_FROZEN"
  else
    python -m benchmarks.fit_four_h20_costs kv \
      --recompute-run "$RUN_ROOT/$recompute_id" \
      --l1-run "$RUN_ROOT/$l1_id" --l2-run "$RUN_ROOT/$l2_id" \
      --template "$ROOT/configs/four_h20/agent-slo-kv-adaptive.yaml" \
      --output-config "$ADAPTIVE_FROZEN" \
      --report "$RUN_ROOT/kv-window-$TAG.cost-fit.json"
  fi
  mark K02
fi
if should_run K03; then
  candidates=()
  for threshold in 256 1024 4096; do
    config="$ROOT/configs/four_h20/agent-slo-kv-fixed-$threshold.yaml"
    run_id="K03-fixed-$threshold-$TAG"
    run_stage "$run_id" kv "$config" "$DATA/profiles/kv_threshold_screening.jsonl" closed_loop 8
    candidates+=(--candidate "fixed-$threshold=$config=$RUN_ROOT/$run_id")
  done
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: python -m benchmarks.freeze_four_h20_config kv ${candidates[*]} --output-config $FROZEN"
  else
    python -m benchmarks.freeze_four_h20_config kv "${candidates[@]}" \
      --output-config "$FROZEN" --report "$RUN_ROOT/kv-window-$TAG.selection.json"
  fi
  mark K03
fi
if should_run K04; then
  config="${KV_FIXED_CONFIG:-$FROZEN}"
  [[ "$DRY_RUN" == 1 || -f "$config" ]] || { echo "missing frozen KV config: $config" >&2; exit 1; }
  run_stage "K04-fixed-before-$TAG" kv "$config" "$DATA/swebench.jsonl" poisson 64 "$TRACE"
  mark K04
fi
if should_run K05; then
  [[ "$DRY_RUN" == 1 || -f "$ADAPTIVE_FROZEN" ]] || { echo "missing measured Adaptive KV config: $ADAPTIVE_FROZEN" >&2; exit 1; }
  run_stage "K05-adaptive-after-$TAG" kv "$ADAPTIVE_FROZEN" "$DATA/swebench.jsonl" poisson 64 "$TRACE"
  mark K05
fi
if should_run K06; then
  if [[ "$SKIP_FAILURE" == 1 ]] || past_soft_limit; then
    echo "K06 skipped (default/timebox); use --include-failure to enable"
  elif [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: inject backend-3 TERM during failure profile"
    run_stage "K06-failure-$TAG" kv "$ROOT/configs/four_h20/agent-slo-kv-adaptive.yaml" "$DATA/profiles/failure.jsonl" closed_loop 4
  else
    (sleep 2; [[ -f /root/log/workload-aware-kv-cache/four-h20/pids/backend-3.pid ]] && kill -TERM "$(</root/log/workload-aware-kv-cache/four-h20/pids/backend-3.pid)" 2>/dev/null || true) &
    run_stage "K06-failure-$TAG" kv "$ROOT/configs/four_h20/agent-slo-kv-adaptive.yaml" "$DATA/profiles/failure.jsonl" closed_loop 4
  fi
  mark K06
fi
if should_run K07; then
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: analyze K04/K05 and stop stack"
  else
    python -m benchmarks.analyze_four_h20 \
      --run "fixed=$RUN_ROOT/K04-fixed-before-$TAG" \
      --run "adaptive=$RUN_ROOT/K05-adaptive-after-$TAG" \
      --output-dir "$RESULT_ROOT/$TAG" --title "Adaptive KV Path"
    "$STACK" stop
  fi
  mark K07
fi
