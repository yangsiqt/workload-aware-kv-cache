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
VERSION_LABEL="${FOUR_H20_VERSION_LABEL:-V2.2}"
RUN_NAME_VERSION="${FOUR_H20_RUN_NAME_VERSION:-V22}"
TAG_PREFIX="${FOUR_H20_TAG_PREFIX:-v2-2}"
TAG="${TAG:-$TAG_PREFIX-$(date -u +%Y%m%dT%H%M%SZ)}"

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
STACK="$ROOT/scripts/four_h20_stack.sh"
STAGE="$ROOT/scripts/run_four_h20_stage.sh"
RUN_ROOT="${RUN_ROOT:-/root/workload-aware-kv-cache-data/runs/four_h20}"
TRACE_ROOT="${FOUR_H20_TRACE_ROOT:-/root/workload-aware-kv-cache-data/traces/four_h20/v2_2}"
DATA="${FOUR_H20_DATA_ROOT:-/root/workload-aware-kv-cache-data/processed/four_h20}"
RESULT_ROOT="${RESULT_ROOT:-/root/performance-results/workload-aware-kv-cache/four-h20/adaptive-kv-v2-2}"
LOG_ROOT="${FOUR_H20_LOG_ROOT:-/root/log/workload-aware-kv-cache/four-h20}"
CONTROL_PREFIX="${FOUR_H20_CONTROL_PREFIX:-V22-window}"
CONTROL="$RUN_ROOT/$CONTROL_PREFIX-$TAG"
COMPLETED="$CONTROL/completed"
ACTIVE_FILE="$CONTROL/active-seconds"
PAIR_FILE="$CONTROL/formal-pair-attempt"
HARD_LIMIT_S="${FOUR_H20_HARD_LIMIT_S:-18000}"
STARTED="$(date +%s)"
BASE_ACTIVE=0
PAIR_ATTEMPT=1
ADAPTIVE_TEMPLATE="${FOUR_H20_ADAPTIVE_TEMPLATE:-$ROOT/configs/four_h20/agent-slo-kv-adaptive-v2-2.yaml}"
FIXED_CONFIG="${FOUR_H20_FIXED_CONFIG:-$ROOT/configs/four_h20/agent-slo-kv-fixed-4096-v2-2.yaml}"
ADAPTIVE_FROZEN_NAME="${FOUR_H20_ADAPTIVE_FROZEN_NAME:-adaptive-v2-2-frozen.yaml}"
ADAPTIVE_FROZEN="$CONTROL/$ADAPTIVE_FROZEN_NAME"
READINESS_SCRIPT="${FOUR_H20_READINESS_SCRIPT:-$ROOT/scripts/check_v2_2_readiness.sh}"
RECORD_MODULE="${FOUR_H20_RECORD_MODULE:-benchmarks.record_v2_2_stage}"
ANALYZE_MODULE="${FOUR_H20_ANALYZE_MODULE:-benchmarks.analyze_v2_2_pair}"
PAIR_REPORT_NAME="${FOUR_H20_PAIR_REPORT_NAME:-v2-2-pair-report.json}"
EXTRA_VLLM_METRIC="${FOUR_H20_EXTRA_VLLM_METRIC:-}"
THRESHOLD_REPORT="$CONTROL/threshold-selection.json"
CAL_WORKLOAD="$TRACE_ROOT/v2-2-calibration-workload-cohort30.jsonl"
CAL_TRACE="$TRACE_ROOT/v2-2-calibration-cohort30-bursty-2.5rps.jsonl"
FORMAL_TRACE="$TRACE_ROOT/v2-2-formal-cohort30-bursty-2.5rps.jsonl"
REPLICATE_TRACE="${FOUR_H20_REPLICATE_TRACE:-}"
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
  ((value > 0)) || { echo "$VERSION_LABEL cumulative five-hour limit reached" >&2; return 1; }
  echo "$value"
}
done_stage() { [[ "$DRY_RUN" == 0 && -f "$COMPLETED/$1" ]]; }
mark_stage() {
  [[ "$DRY_RUN" == 1 ]] || touch "$COMPLETED/$1"
  if [[ "$DRY_RUN" == 0 ]]; then
    python -m "$RECORD_MODULE" --tag "$TAG" --stage "$1" \
      --status PASS --detail "$2"
  fi
  echo "$VERSION_LABEL stage $1 PASS: $2"
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
validate_live_stack() {
  local gpu pid metrics metric cmdline
  local -a vllm_metrics=(
    "vllm:num_requests_running"
    "vllm:num_requests_waiting"
    "vllm:waiting_prefill_tokens"
    "vllm:running_prefill_tokens"
    "vllm:active_decode_sequences"
    "vllm:scheduled_prefill_tokens"
    "vllm:scheduled_decode_tokens"
    "vllm:skipped_waiting_prefill_tokens"
    "vllm:kv_cache_free_blocks"
    "vllm:kv_cache_total_blocks"
  )
  [[ -n "$EXTRA_VLLM_METRIC" ]] && vllm_metrics+=("$EXTRA_VLLM_METRIC")
  local -a mooncake_metrics=(
    "mooncake_transfer_inflight_read_operations"
    "mooncake_transfer_inflight_read_bytes"
    "mooncake_transfer_read_failures"
    "mooncake_transfer_read_misses"
  )
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: validate four live vLLM/Mooncake endpoints, KV-event ports and max_num_seqs=12"
    return
  fi
  for gpu in 0 1 2 3; do
    metrics="$(curl -fsS "http://127.0.0.1:$((8000 + gpu))/metrics")"
    for metric in "${vllm_metrics[@]}"; do
      grep -Fq "$metric" <<<"$metrics" || {
        echo "backend-$gpu missing required metric: $metric" >&2
        return 1
      }
    done
    pid="$(<"$LOG_ROOT/pids/backend-$gpu.pid")"
    cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
    grep -Fq -- "--max-num-seqs 12" <<<"$cmdline" || return 1
    grep -Fq -- "tcp://*:$((9400 + gpu))" <<<"$cmdline" || return 1
    timeout 1 bash -c "</dev/tcp/127.0.0.1/$((9400 + gpu))"
    curl -fsS "http://127.0.0.1:$((9300 + gpu))/health" >/dev/null
    metrics="$(curl -fsS "http://127.0.0.1:$((9300 + gpu))/metrics")"
    for metric in "${mooncake_metrics[@]}"; do
      grep -Fq "$metric" <<<"$metrics" || return 1
    done
  done
}
finish() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$DRY_RUN" == 0 ]]; then
    echo "$(elapsed)" >"$ACTIVE_FILE"
    if ((status != 0)) && [[ -n "$CURRENT_STAGE" ]]; then
      python -m "$RECORD_MODULE" --tag "$TAG" \
        --stage "$CURRENT_STAGE" --status FAIL \
        --detail "节点失败，已停止服务并保留现有证据。" || true
    fi
    if ((status != 0)) && [[ "$CURRENT_STAGE" =~ ^K0[45]R?$ ]]; then
      rm -f "$COMPLETED/K04" "$COMPLETED/K05" \
        "$COMPLETED/K04R" "$COMPLETED/K05R"
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
    "$READINESS_SCRIPT"
  else
    "$READINESS_SCRIPT" --require-gpu \
      --output "$CONTROL/readiness.json"
  fi
  ensure_stack
  validate_live_stack
  mark_stage F00 "四Backend启动；max_num_seqs=12；$VERSION_LABEL静态与GPU门禁通过。"
fi

CAL_DEFAULT="${RUN_NAME_VERSION}K01-calibration-default-$TAG"
if ! done_stage K01; then
  CURRENT_STAGE=K01
  ensure_stack
  MIN_SUCCESS_RATE=1 REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$CAL_DEFAULT" kv \
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
      validation_run="${RUN_NAME_VERSION}K01-calibration-recalibrated-$TAG"
      MIN_SUCCESS_RATE=1 REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$validation_run" kv \
        "$ADAPTIVE_FROZEN" "$CAL_WORKLOAD" poisson 64 "$CAL_TRACE"
    fi
    python -m benchmarks.validate_v2_2_activation \
      "$RUN_ROOT/$validation_run/joined_trace.jsonl" --expected-rows 180 \
      --min-overrides 9 --min-path-changes 4 --min-external-hit-rate 0.95 \
      --min-external-overrides 4 \
      --output "$CONTROL/calibration-activation.json"
    if [[ "$VERSION_LABEL" == "V2.3" ]]; then
      python -m benchmarks.validate_v2_3_decode_telemetry \
        "$RUN_ROOT/$validation_run/joined_trace.jsonl" --expected-rows 180 \
        --output "$CONTROL/calibration-decode-telemetry.json"
    fi
  fi
  mark_stage K01 "180请求独立Trace校准完成；阈值最多调整一次并冻结。"
fi

if [[ "$DRY_RUN" == 0 && ! -f "$ADAPTIVE_FROZEN" ]]; then
  echo "missing frozen $VERSION_LABEL adaptive config: $ADAPTIVE_FROZEN" >&2
  exit 1
fi
[[ "$DRY_RUN" == 1 ]] && ADAPTIVE_FROZEN="$ADAPTIVE_TEMPLATE"

FIXED_RUN="${RUN_NAME_VERSION}K04-fixed-before-2.5rps-$TAG-p$PAIR_ATTEMPT"
if ! done_stage K04; then
  CURRENT_STAGE=K04
  ensure_stack
  REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$FIXED_RUN" kv \
    "$FIXED_CONFIG" "$WORKLOAD" poisson 64 "$FORMAL_TRACE"
  mark_stage K04 "fixed-4096 Before 1200请求完成；不单独形成结论。"
fi

ADAPTIVE_RUN="${RUN_NAME_VERSION}K05-adaptive-after-2.5rps-$TAG-p$PAIR_ATTEMPT"
if ! done_stage K05; then
  CURRENT_STAGE=K05
  ensure_stack
  REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$ADAPTIVE_RUN" kv \
    "$ADAPTIVE_FROZEN" "$WORKLOAD" poisson 64 "$FORMAL_TRACE"
  mark_stage K05 "Adaptive $VERSION_LABEL After 1200请求完成；等待配对门禁。"
fi

FIXED_REPLICATE_RUN="${RUN_NAME_VERSION}K04R-fixed-replicate-2.5rps-$TAG-p$PAIR_ATTEMPT"
ADAPTIVE_REPLICATE_RUN="${RUN_NAME_VERSION}K05R-adaptive-replicate-2.5rps-$TAG-p$PAIR_ATTEMPT"
if [[ -n "$REPLICATE_TRACE" ]] && ! done_stage K05R; then
  CURRENT_STAGE=K05R
  ensure_stack
  REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$ADAPTIVE_REPLICATE_RUN" kv \
    "$ADAPTIVE_FROZEN" "$WORKLOAD" poisson 64 "$REPLICATE_TRACE"
  mark_stage K05R "Adaptive独立Arrival Trace复验完成；等待双Trace门禁。"
fi
if [[ -n "$REPLICATE_TRACE" ]] && ! done_stage K04R; then
  CURRENT_STAGE=K04R
  ensure_stack
  REQUIRE_V2_2_WORKER_LIFECYCLE=true run_stage "$FIXED_REPLICATE_RUN" kv \
    "$FIXED_CONFIG" "$WORKLOAD" poisson 64 "$REPLICATE_TRACE"
  mark_stage K04R "fixed-4096独立Arrival Trace复验完成；与Adaptive形成反向顺序配对。"
fi

if ! done_stage K07; then
  CURRENT_STAGE=K07
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: validate activation and analyze paired formal results"
  else
    output="$RESULT_ROOT/$TAG/$PAIR_REPORT_NAME"
    mkdir -p "$(dirname "$output")"
    analyze_command=(
      python -m "$ANALYZE_MODULE"
      --fixed "$RUN_ROOT/$FIXED_RUN"
      --adaptive "$RUN_ROOT/$ADAPTIVE_RUN"
      --output "$output"
    )
    if [[ -n "$REPLICATE_TRACE" ]]; then
      analyze_command+=(
        --replicate-fixed "$RUN_ROOT/$FIXED_REPLICATE_RUN"
        --replicate-adaptive "$RUN_ROOT/$ADAPTIVE_REPLICATE_RUN"
      )
    fi
    "${analyze_command[@]}"
  fi
  mark_stage K07 "公平配对、激活和性能门禁完成；结论以四卡报告为准。"
fi
