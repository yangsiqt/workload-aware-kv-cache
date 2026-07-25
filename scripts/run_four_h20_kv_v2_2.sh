#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
TAG="${FOUR_H20_RUN_TAG:-}"
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --resume) RESUME=1 ;;
    --run-tag) shift; TAG="${1:?--run-tag requires a value}" ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
TAG="${TAG:-v2-2-$(date -u +%Y%m%dT%H%M%SZ)}"

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
STACK="$ROOT/scripts/four_h20_stack.sh"
STAGE="$ROOT/scripts/run_four_h20_stage.sh"
RUN_ROOT="${RUN_ROOT:-/root/workload-aware-kv-cache-data/runs/four_h20}"
TRACE_ROOT="${FOUR_H20_TRACE_ROOT:-/root/workload-aware-kv-cache-data/traces/four_h20/v2_2}"
DATA="${FOUR_H20_DATA_ROOT:-/root/workload-aware-kv-cache-data/processed/four_h20}"
RESULT_ROOT="${RESULT_ROOT:-/root/performance-results/workload-aware-kv-cache/four-h20/adaptive-kv-v2-2}"
CONTROL="$RUN_ROOT/V22-window-$TAG"
COMPLETED="$CONTROL/completed"
ACTIVE_FILE="$CONTROL/active-seconds"
PAIR_FILE="$CONTROL/formal-pair-attempt"
HARD_LIMIT_S="${FOUR_H20_HARD_LIMIT_S:-18000}"
STARTED="$(date +%s)"
BASE_ACTIVE=0
PAIR_ATTEMPT=1
ADAPTIVE_TEMPLATE="$ROOT/configs/four_h20/agent-slo-kv-adaptive-v2-2.yaml"
FIXED_CONFIG="$ROOT/configs/four_h20/agent-slo-kv-fixed-4096-v2-2.yaml"
ADAPTIVE_FROZEN="$CONTROL/adaptive-v2-2-frozen.yaml"
THRESHOLD_REPORT="$CONTROL/threshold-selection.json"
CAL_WORKLOAD="$TRACE_ROOT/v2-2-calibration-workload.jsonl"
CAL_TRACE="$TRACE_ROOT/v2-2-calibration-wave-bursty-2.5rps.jsonl"
FORMAL_TRACE="$TRACE_ROOT/v2-2-formal-wave-bursty-2.5rps.jsonl"
WORKLOAD="$DATA/swebench.jsonl"
STACK_STARTED=0
CURRENT_STAGE=""

if [[ "$DRY_RUN" == 0 ]]; then
  if [[ -d "$CONTROL" && "$RESUME" == 0 ]]; then
    echo "control directory exists; use --resume or a new tag: $CONTROL" >&2
    exit 1
  fi
  mkdir -p "$COMPLETED"
  [[ -f "$ACTIVE_FILE" ]] && BASE_ACTIVE="$(<"$ACTIVE_FILE")"
  [[ -f "$PAIR_FILE" ]] && PAIR_ATTEMPT="$(<"$PAIR_FILE")"
fi

elapsed() { echo $((BASE_ACTIVE + $(date +%s) - STARTED)); }
remaining() {
  local value=$((HARD_LIMIT_S - $(elapsed)))
  ((value > 0)) || { echo "V2.2 cumulative five-hour limit reached" >&2; return 1; }
  echo "$value"
}
done_stage() { [[ "$DRY_RUN" == 0 && -f "$COMPLETED/$1" ]]; }
mark_stage() {
  [[ "$DRY_RUN" == 1 ]] || touch "$COMPLETED/$1"
  echo "V2.2 stage $1 PASS: $2"
}
run_stage() {
  if [[ "$DRY_RUN" == 1 ]]; then
    "$STAGE" --dry-run "$@"
  else
    timeout --signal=TERM --kill-after=30s "$(remaining)s" "$STAGE" "$@"
  fi
}
ensure_stack() {
  ((STACK_STARTED == 1)) && return
  if [[ "$DRY_RUN" == 1 ]]; then
    MAX_NUM_SEQS=12 "$STACK" --dry-run start-kv
  else
    MAX_NUM_SEQS=12 "$STACK" start-kv
  fi
  STACK_STARTED=1
}
finish() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$DRY_RUN" == 0 ]]; then
    echo "$(elapsed)" >"$ACTIVE_FILE"
    if ((status != 0)) && [[ "$CURRENT_STAGE" =~ ^K0[45]$ ]]; then
      rm -f "$COMPLETED/K04" "$COMPLETED/K05"
      PAIR_ATTEMPT=$((PAIR_ATTEMPT + 1))
      echo "$PAIR_ATTEMPT" >"$PAIR_FILE"
    fi
    "$STACK" stop >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap finish EXIT INT TERM

if ! done_stage F00; then
  CURRENT_STAGE=F00
  if [[ "$DRY_RUN" == 1 ]]; then
    "$ROOT/scripts/check_v2_2_readiness.sh"
  else
    "$ROOT/scripts/check_v2_2_readiness.sh" --require-gpu \
      --output "$CONTROL/readiness.json"
  fi
  ensure_stack
  mark_stage F00 "四Backend启动；max_num_seqs=12；V2.2静态与GPU门禁通过。"
fi

CAL_DEFAULT="V22K01-calibration-default-$TAG"
if ! done_stage K01; then
  CURRENT_STAGE=K01
  ensure_stack
  REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$CAL_DEFAULT" kv \
    "$ADAPTIVE_TEMPLATE" "$CAL_WORKLOAD" poisson 64 "$CAL_TRACE"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: select one of 500ms/15%, 250ms/10%, 750ms/20% and freeze config"
    ADAPTIVE_FROZEN="$ADAPTIVE_TEMPLATE"
  else
    python -m benchmarks.select_v2_2_thresholds \
      "$RUN_ROOT/$CAL_DEFAULT/joined_trace.jsonl" \
      --template "$ADAPTIVE_TEMPLATE" --output-config "$ADAPTIVE_FROZEN" \
      --report "$THRESHOLD_REPORT"
    recalibrate="$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["requires_one_recalibration"]).lower())' "$THRESHOLD_REPORT")"
    validation_run="$CAL_DEFAULT"
    if [[ "$recalibrate" == true ]]; then
      validation_run="V22K01-calibration-recalibrated-$TAG"
      REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$validation_run" kv \
        "$ADAPTIVE_FROZEN" "$CAL_WORKLOAD" poisson 64 "$CAL_TRACE"
    fi
    python -m benchmarks.validate_v2_2_activation \
      "$RUN_ROOT/$validation_run/joined_trace.jsonl" --expected-rows 180 \
      --min-overrides 9 --min-path-changes 4 --min-external-hit-rate 0.95 \
      --min-external-overrides 1 \
      --output "$CONTROL/calibration-activation.json"
  fi
  mark_stage K01 "180请求独立Trace校准完成；阈值最多调整一次并冻结。"
fi

if [[ "$DRY_RUN" == 0 && ! -f "$ADAPTIVE_FROZEN" ]]; then
  echo "missing frozen V2.2 adaptive config: $ADAPTIVE_FROZEN" >&2
  exit 1
fi
[[ "$DRY_RUN" == 1 ]] && ADAPTIVE_FROZEN="$ADAPTIVE_TEMPLATE"

FIXED_RUN="V22K04-fixed-before-2.5rps-$TAG-p$PAIR_ATTEMPT"
if ! done_stage K04; then
  CURRENT_STAGE=K04
  ensure_stack
  REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$FIXED_RUN" kv \
    "$FIXED_CONFIG" "$WORKLOAD" poisson 64 "$FORMAL_TRACE"
  mark_stage K04 "fixed-4096 Before 1200请求完成；不单独形成结论。"
fi

ADAPTIVE_RUN="V22K05-adaptive-after-2.5rps-$TAG-p$PAIR_ATTEMPT"
if ! done_stage K05; then
  CURRENT_STAGE=K05
  ensure_stack
  REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$ADAPTIVE_RUN" kv \
    "$ADAPTIVE_FROZEN" "$WORKLOAD" poisson 64 "$FORMAL_TRACE"
  mark_stage K05 "Adaptive V2.2 After 1200请求完成；等待配对门禁。"
fi

if ! done_stage K07; then
  CURRENT_STAGE=K07
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: validate activation and analyze paired formal results"
  else
    output="$RESULT_ROOT/$TAG/v2-2-pair-report.json"
    mkdir -p "$(dirname "$output")"
    python -m benchmarks.analyze_v2_2_pair \
      --fixed "$RUN_ROOT/$FIXED_RUN" --adaptive "$RUN_ROOT/$ADAPTIVE_RUN" \
      --output "$output"
  fi
  mark_stage K07 "公平配对、激活和性能门禁完成；结论以四卡报告为准。"
fi
