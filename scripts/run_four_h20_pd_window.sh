#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
FROM="P01"
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
RESULT_ROOT="${RESULT_ROOT:-/root/performance-results/workload-aware-kv-cache/four-h20/adaptive-pd}"
if [[ -n "$RUN_TAG_ARG" ]]; then
  TAG="$RUN_TAG_ARG"
elif [[ -n "${FOUR_H20_RUN_TAG:-}" ]]; then
  TAG="$FOUR_H20_RUN_TAG"
elif [[ "$RESUME" == 1 ]]; then
  latest_state="$(find "$RUN_ROOT" -maxdepth 1 -type f -name 'pd-window-*.state' -printf '%T@ %f\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  [[ -n "$latest_state" ]] || { echo "--resume requested but no PD state file exists" >&2; exit 1; }
  TAG="${latest_state#pd-window-}"
  TAG="${TAG%.state}"
else
  TAG="$(date -u +%Y%m%dT%H%M%SZ)"
fi
RPS="${FORMAL_RPS:-4}"
TRACE="/root/workload-aware-kv-cache-data/traces/four_h20/${RPS}rps/swe-final-poisson-${RPS}rps-r1.jsonl"
STATE="$RUN_ROOT/pd-window-$TAG.state"
FROZEN="$RUN_ROOT/pd-window-$TAG.fixed.yaml"
ADAPTIVE_FROZEN="$RUN_ROOT/pd-window-$TAG.adaptive.yaml"
START_EPOCH="$(date +%s)"
HARD_LIMIT_S="${FOUR_H20_HARD_LIMIT_S:-18000}"
SOFT_LIMIT_S="${FOUR_H20_SOFT_LIMIT_S:-14400}"
stages=(P01 P02 P03 P04 P05 P06)

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
    (( remaining > 0 )) || { echo "four-H20 PD hard time limit reached" >&2; return 124; }
    timeout --signal=TERM --kill-after=30s "${remaining}s" "$STAGE" "$@"
  fi
}
past_soft_limit() { (( $(date +%s) - START_EPOCH >= SOFT_LIMIT_S )); }
start_stack() { if [[ "$DRY_RUN" == 1 ]]; then "$STACK" --dry-run start-pd; else "$STACK" start-pd; fi; }
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

if (( from_index > 0 && from_index < 5 )); then
  ensure_stack
fi

if should_run P01; then
  start_stack
  MIN_SUCCESS_RATE=1 REQUIRE_EXECUTION_MODES=pd run_stage "P01-smoke-$TAG" pd "$ROOT/configs/four_h20/agent-slo-pd-always.yaml" "$DATA/profiles/four_h20_smoke.jsonl" closed_loop 4
  mark P01
fi
if should_run P02; then
  mono_id="P02-monolithic-$TAG"
  pd_id="P02-pd-$TAG"
  MIN_SUCCESS_RATE=1 REQUIRE_EXECUTION_MODES=monolithic run_stage "$mono_id" pd "$ROOT/configs/four_h20/agent-slo-pd-monolithic.yaml" "$DATA/profiles/pd_crossover_monolithic.jsonl" closed_loop 4
  MIN_SUCCESS_RATE=1 REQUIRE_EXECUTION_MODES=pd run_stage "$pd_id" pd "$ROOT/configs/four_h20/agent-slo-pd-always.yaml" "$DATA/profiles/pd_crossover_pd.jsonl" closed_loop 4
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: freeze PD crossover config to $FROZEN"
  else
    python -m benchmarks.freeze_four_h20_config pd \
      --monolithic-run "$RUN_ROOT/$mono_id" --pd-run "$RUN_ROOT/$pd_id" \
      --template "$ROOT/configs/four_h20/agent-slo-pd-fixed.yaml" \
      --output-config "$FROZEN" --report "$RUN_ROOT/pd-window-$TAG.selection.json"
    python -m benchmarks.fit_four_h20_costs pd \
      --monolithic-run "$RUN_ROOT/$mono_id" --pd-run "$RUN_ROOT/$pd_id" \
      --template "$ROOT/configs/four_h20/agent-slo-pd-adaptive.yaml" \
      --output-config "$ADAPTIVE_FROZEN" \
      --report "$RUN_ROOT/pd-window-$TAG.cost-fit.json"
  fi
  mark P02
fi
if should_run P03; then
  config="${PD_FIXED_CONFIG:-$FROZEN}"
  [[ "$DRY_RUN" == 1 || -f "$config" ]] || { echo "missing frozen PD config: $config" >&2; exit 1; }
  run_stage "P03-fixed-before-$TAG" pd "$config" "$DATA/swebench.jsonl" poisson 64 "$TRACE"
  mark P03
fi
if should_run P04; then
  [[ "$DRY_RUN" == 1 || -f "$ADAPTIVE_FROZEN" ]] || { echo "missing measured Adaptive PD config: $ADAPTIVE_FROZEN" >&2; exit 1; }
  run_stage "P04-adaptive-after-$TAG" pd "$ADAPTIVE_FROZEN" "$DATA/swebench.jsonl" poisson 64 "$TRACE"
  mark P04
fi
if should_run P05; then
  if [[ "$SKIP_FAILURE" == 1 ]] || past_soft_limit; then
    echo "P05 skipped (default/timebox); use --include-failure to enable"
  elif [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: inject prefill backend TERM during failure profile"
    run_stage "P05-failure-$TAG" pd "$ROOT/configs/four_h20/agent-slo-pd-adaptive.yaml" "$DATA/profiles/failure.jsonl" closed_loop 4
  else
    (sleep 2; [[ -f /root/log/workload-aware-kv-cache/four-h20/pids/backend-2.pid ]] && kill -TERM "$(</root/log/workload-aware-kv-cache/four-h20/pids/backend-2.pid)" 2>/dev/null || true) &
    run_stage "P05-failure-$TAG" pd "$ROOT/configs/four_h20/agent-slo-pd-adaptive.yaml" "$DATA/profiles/failure.jsonl" closed_loop 4
  fi
  mark P05
fi
if should_run P06; then
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: analyze P03/P04 and stop stack"
  else
    python -m benchmarks.analyze_four_h20 \
      --run "fixed=$RUN_ROOT/P03-fixed-before-$TAG" \
      --run "adaptive=$RUN_ROOT/P04-adaptive-after-$TAG" \
      --output-dir "$RESULT_ROOT/$TAG" --title "Hybrid Adaptive PD"
    "$STACK" stop
  fi
  mark P06
fi
