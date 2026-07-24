#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
RESUME=0
FROM_STAGE="F00"
TAG="${FOUR_H20_RUN_TAG:-}"
while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --resume) RESUME=1 ;;
    --run-tag) shift; TAG="${1:?--run-tag requires a value}" ;;
    --from) shift; FROM_STAGE="${1:?--from requires a stage}" ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
TAG="${TAG:-v2-1-$(date -u +%Y%m%dT%H%M%SZ)}"

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
STACK="$ROOT/scripts/four_h20_stack.sh"
STAGE="$ROOT/scripts/run_four_h20_stage.sh"
DATA="${FOUR_H20_DATA_ROOT:-/root/workload-aware-kv-cache-data/processed/four_h20}"
TRACE_ROOT="${FOUR_H20_TRACE_ROOT:-/root/workload-aware-kv-cache-data/traces/four_h20}"
RUN_ROOT="${RUN_ROOT:-/root/workload-aware-kv-cache-data/runs/four_h20}"
RESULT_ROOT="${RESULT_ROOT:-/root/performance-results/workload-aware-kv-cache/four-h20/adaptive-kv-v2-1}"
LOG_ROOT="${FOUR_H20_LOG_ROOT:-/root/log/workload-aware-kv-cache/four-h20}"
CONTROL="$RUN_ROOT/V21-window-$TAG"
COMPLETED="$CONTROL/completed"
HARD_LIMIT_S="${FOUR_H20_HARD_LIMIT_S:-18000}"
ACTIVE_FILE="$CONTROL/active-seconds"
PAIR_ATTEMPT_FILE="$CONTROL/formal-pair-attempt"
INVOCATION_STARTED="$(date +%s)"
BASE_ACTIVE=0
PAIR_ATTEMPT=1
FIXED_TEMPLATE="$ROOT/configs/four_h20/agent-slo-kv-fixed-4096-v2-1.yaml"
ADAPTIVE_TEMPLATE="$ROOT/configs/four_h20/agent-slo-kv-adaptive-v2-1.yaml"
FIXED_FROZEN="$CONTROL/fixed-4096-frozen.yaml"
ADAPTIVE_FROZEN="$CONTROL/adaptive-v2-1-frozen.yaml"
COST_REPORT="$CONTROL/k02-cost-freeze.json"
CAPACITY_REPORT="$CONTROL/capacity-decision.json"
PROFILE_ROOT="$DATA/profiles"
FINAL_RPS=""
FINAL_FIXED_RUN=""
FINAL_ADAPTIVE_RUN=""
STACK_STARTED=0
CURRENT_STAGE=""
CURRENT_RECORDED=0
CURRENT_RUNS=()
CURRENT_ARTIFACTS=()

STAGES=(F00 K01 K02 K03 K04 K05 K07)
stage_index() {
  local target="$1" index
  for index in "${!STAGES[@]}"; do
    [[ "${STAGES[$index]}" == "$target" ]] && { echo "$index"; return; }
  done
  echo "unknown stage: $target" >&2
  return 1
}
FROM_INDEX="$(stage_index "$FROM_STAGE")"

if [[ "$DRY_RUN" == 0 ]]; then
  if [[ -d "$CONTROL" && "$RESUME" == 0 ]]; then
    echo "control directory already exists; use --resume or a new --run-tag: $CONTROL" >&2
    exit 1
  fi
  mkdir -p "$COMPLETED"
  [[ -f "$ACTIVE_FILE" ]] && BASE_ACTIVE="$(<"$ACTIVE_FILE")"
  [[ -f "$PAIR_ATTEMPT_FILE" ]] && PAIR_ATTEMPT="$(<"$PAIR_ATTEMPT_FILE")"
  for ((index=0; index<FROM_INDEX; index++)); do
    [[ -f "$COMPLETED/${STAGES[$index]}" ]] || {
      echo "cannot start from $FROM_STAGE; ${STAGES[$index]} is not complete" >&2
      exit 1
    }
  done
fi

elapsed_active() {
  echo $((BASE_ACTIVE + $(date +%s) - INVOCATION_STARTED))
}

remaining_seconds() {
  local remaining=$((HARD_LIMIT_S - $(elapsed_active)))
  if ((remaining <= 0)); then
    echo "V2.1 cumulative five-hour active limit reached" >&2
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

should_run() {
  local stage="$1" index
  index="$(stage_index "$stage")"
  ((index >= FROM_INDEX)) || return 1
  [[ "$DRY_RUN" == 1 || ! -f "$COMPLETED/$stage" ]]
}

begin_stage() {
  CURRENT_STAGE="$1"
  CURRENT_RECORDED=0
  CURRENT_RUNS=()
  CURRENT_ARTIFACTS=()
}

record_stage() {
  local status="$1" note="${2:-}" path args stage_label="$CURRENT_STAGE"
  [[ "$DRY_RUN" == 1 ]] && return
  if [[ "$CURRENT_STAGE" == "K04" || "$CURRENT_STAGE" == "K05" ]]; then
    stage_label="${CURRENT_STAGE}-p${PAIR_ATTEMPT}"
  fi
  args=(python -m benchmarks.record_v2_1_four_h20 --stage "$stage_label" --tag "$TAG" --status "$status" --note "$note")
  for path in "${CURRENT_RUNS[@]}"; do
    [[ -d "$RUN_ROOT/$path" ]] && args+=(--run-dir "$RUN_ROOT/$path")
  done
  for path in "${CURRENT_ARTIFACTS[@]}"; do
    [[ -f "$path" ]] && args+=(--artifact "$path")
  done
  "${args[@]}"
  CURRENT_RECORDED=1
}

complete_stage() {
  local note="${1:-}"
  record_stage PASS "$note"
  [[ "$DRY_RUN" == 1 ]] || touch "$COMPLETED/$CURRENT_STAGE"
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
  local -a mooncake_metrics=(
    "mooncake_transfer_inflight_read_operations"
    "mooncake_transfer_inflight_read_bytes"
    "mooncake_transfer_read_failures"
    "mooncake_transfer_read_misses"
  )

  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: validate four live vLLM metric endpoints, four Mooncake metric endpoints and max_num_seqs=12"
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
    grep -Fq -- "--max-num-seqs 12" <<<"$cmdline" || {
      echo "backend-$gpu did not start with max_num_seqs=12" >&2
      return 1
    }

    curl -fsS "http://127.0.0.1:$((9300 + gpu))/health" >/dev/null
    metrics="$(curl -fsS "http://127.0.0.1:$((9300 + gpu))/metrics")"
    for metric in "${mooncake_metrics[@]}"; do
      grep -Fq "$metric" <<<"$metrics" || {
        echo "backend-$gpu Mooncake endpoint missing required metric: $metric" >&2
        return 1
      }
    done
  done
}

trace_for_rps() {
  local rps="$1"
  echo "$TRACE_ROOT/${rps}rps/swe-final-poisson-${rps}rps-r2.jsonl"
}

finish() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$DRY_RUN" == 0 ]]; then
    echo "$(elapsed_active)" >"$ACTIVE_FILE"
    if ((status != 0)) && [[ -n "$CURRENT_STAGE" ]] && ((CURRENT_RECORDED == 0)); then
      record_stage FAIL "阶段异常退出；保留原始run和日志，禁止继续后续节点。" || true
    fi
    if ((status != 0)) && [[ "$CURRENT_STAGE" == "K04" || "$CURRENT_STAGE" == "K05" ]]; then
      rm -f "$COMPLETED/K04" "$COMPLETED/K05"
      PAIR_ATTEMPT=$((PAIR_ATTEMPT + 1))
      echo "$PAIR_ATTEMPT" >"$PAIR_ATTEMPT_FILE"
    fi
    "$STACK" stop >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap finish EXIT INT TERM

if should_run F00; then
  begin_stage F00
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: verify environment, readiness, four GPUs, hashes, disk and ports"
  else
    "$ROOT/scripts/verify_four_h20_environment.sh"
    "$ROOT/scripts/check_four_h20_readiness.sh"
    [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 4 ]]
    [[ "$(nproc)" -ge 64 ]]
    memory_limit="$(cat /sys/fs/cgroup/memory.max)"
    [[ "$memory_limit" == max || "$memory_limit" -ge 644245094400 ]]
    [[ "$(df --output=avail -k / | tail -1)" -ge 8388608 ]]
  fi
  ensure_stack
  validate_live_stack
  complete_stage "公共环境与readiness通过，四Backend按max_num_seqs=12启动且V2.1在线指标完整；尚未产生性能结果。"
fi

if should_run K01; then
  begin_stage K01
  ensure_stack
  strict="$PROFILE_ROOT/v2_1_k01_strict.jsonl"
  target="$PROFILE_ROOT/v2_1_k01_lru_target.jsonl"
  fillers="$PROFILE_ROOT/v2_1_k01_lru_fillers.jsonl"

  a="V21K01A-strict-recompute-$TAG"; CURRENT_RUNS+=("$a")
  MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true REQUIRE_SELECTED_KV_PATHS=recompute \
    run_stage "$a" kv "$ROOT/configs/four_h20/agent-slo-kv-recompute.yaml" "$strict" closed_loop 4
  b="V21K01B-strict-l1-$TAG"; CURRENT_RUNS+=("$b")
  RESET_EXTERNAL=false MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true \
    REQUIRE_SELECTED_KV_PATHS=lmcache_l1 REQUIRE_ACTUAL_KV_PATHS=lmcache_l1 \
    run_stage "$b" kv "$ROOT/configs/four_h20/agent-slo-kv-force-l1.yaml" "$strict" closed_loop 4
  if [[ "$DRY_RUN" == 1 ]]; then "$STACK" --dry-run clear-l1; else "$STACK" clear-l1; fi
  c="V21K01C-strict-l2-$TAG"; CURRENT_RUNS+=("$c")
  RESET_EXTERNAL=false MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true \
    REQUIRE_SELECTED_KV_PATHS=mooncake_l2 REQUIRE_ACTUAL_KV_PATHS=mooncake_l2 \
    run_stage "$c" kv "$ROOT/configs/four_h20/agent-slo-kv-force-l2.yaml" "$strict" closed_loop 4

  d="V21K01D-lru-cold-$TAG"; CURRENT_RUNS+=("$d")
  MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true REQUIRE_SELECTED_KV_PATHS=recompute \
    run_stage "$d" kv "$ROOT/configs/four_h20/agent-slo-kv-recompute.yaml" "$target" closed_loop 1
  for sample in 1 2; do
    run="V21K01E${sample}-adaptive-l1-$TAG"; CURRENT_RUNS+=("$run")
    RESET_EXTERNAL=false ROUTER_REFRESH_EXPECTED_PATH=lmcache_l1 MIN_SUCCESS_RATE=1 \
      REQUIRE_V2_1_WORKER_LIFECYCLE=true REQUIRE_SELECTED_KV_PATHS=lmcache_l1 \
      REQUIRE_ACTUAL_KV_PATHS=lmcache_l1 \
      run_stage "$run" kv "$ADAPTIVE_TEMPLATE" "$target" closed_loop 1
  done
  f="V21K01F-lru-fillers-$TAG"; CURRENT_RUNS+=("$f")
  RESET_EXTERNAL=false BENCHMARK_INTER_REQUEST_DELAY_S=1 \
    MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true \
    REQUIRE_SELECTED_KV_PATHS=recompute \
    run_stage "$f" kv "$ROOT/configs/four_h20/agent-slo-kv-recompute.yaml" "$fillers" closed_loop 1
  g="V21K01G-adaptive-l2-$TAG"; CURRENT_RUNS+=("$g")
  RESET_EXTERNAL=false ROUTER_REFRESH_EXPECTED_PATH=mooncake_l2 MIN_SUCCESS_RATE=1 \
    REQUIRE_V2_1_WORKER_LIFECYCLE=true REQUIRE_SELECTED_KV_PATHS=mooncake_l2 \
    REQUIRE_ACTUAL_KV_PATHS=mooncake_l2 \
    run_stage "$g" kv "$ADAPTIVE_TEMPLATE" "$target" closed_loop 1
  k01_report="$CONTROL/k01-validation.json"; CURRENT_ARTIFACTS+=("$k01_report")
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: aggregate and validate the exact 20-request K01 matrix"
  else
    python -m benchmarks.validate_v2_1_k01 \
      --strict-recompute "$RUN_ROOT/$a" --strict-l1 "$RUN_ROOT/$b" --strict-l2 "$RUN_ROOT/$c" \
      --cold-target "$RUN_ROOT/$d" --adaptive-l1 "$RUN_ROOT/V21K01E1-adaptive-l1-$TAG" \
      --adaptive-l1 "$RUN_ROOT/V21K01E2-adaptive-l1-$TAG" --fillers "$RUN_ROOT/$f" \
      --adaptive-l2 "$RUN_ROOT/$g" --output "$k01_report"
  fi
  complete_stage "20/20严格路径与真实LRU多层可见性Smoke完成。"
fi

if should_run K02; then
  begin_stage K02
  ensure_stack
  profile="$PROFILE_ROOT/v2_1_k02_cost.jsonl"
  recompute="V21K02-recompute-$TAG"; CURRENT_RUNS+=("$recompute")
  BENCHMARK_INTER_REQUEST_DELAY_S=1 MIN_SUCCESS_RATE=1 \
    REQUIRE_V2_1_WORKER_LIFECYCLE=true REQUIRE_SELECTED_KV_PATHS=recompute \
    run_stage "$recompute" kv "$ROOT/configs/four_h20/agent-slo-kv-recompute.yaml" "$profile" closed_loop 1
  l1="V21K02-l1-$TAG"; CURRENT_RUNS+=("$l1")
  RESET_EXTERNAL=false BENCHMARK_INTER_REQUEST_DELAY_S=1 MIN_SUCCESS_RATE=1 \
    REQUIRE_V2_1_WORKER_LIFECYCLE=true \
    REQUIRE_SELECTED_KV_PATHS=lmcache_l1 REQUIRE_ACTUAL_KV_PATHS=lmcache_l1 \
    run_stage "$l1" kv "$ROOT/configs/four_h20/agent-slo-kv-force-l1.yaml" "$profile" closed_loop 1
  if [[ "$DRY_RUN" == 1 ]]; then "$STACK" --dry-run clear-l1; else "$STACK" clear-l1; fi
  l2="V21K02-l2-$TAG"; CURRENT_RUNS+=("$l2")
  RESET_EXTERNAL=false BENCHMARK_INTER_REQUEST_DELAY_S=1 MIN_SUCCESS_RATE=1 \
    REQUIRE_V2_1_WORKER_LIFECYCLE=true \
    REQUIRE_SELECTED_KV_PATHS=mooncake_l2 REQUIRE_ACTUAL_KV_PATHS=mooncake_l2 \
    run_stage "$l2" kv "$ROOT/configs/four_h20/agent-slo-kv-force-l2.yaml" "$profile" closed_loop 1
  CURRENT_ARTIFACTS+=("$COST_REPORT")
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: freeze identical K02 costs into $FIXED_FROZEN and $ADAPTIVE_FROZEN"
  else
    python -m benchmarks.freeze_v2_1_kv_costs \
      --recompute-run "$RUN_ROOT/$recompute" --l1-run "$RUN_ROOT/$l1" --l2-run "$RUN_ROOT/$l2" \
      --profile "$profile" --fixed-template "$FIXED_TEMPLATE" --adaptive-template "$ADAPTIVE_TEMPLATE" \
      --output-dir "$CONTROL" --report "$COST_REPORT"
  fi
  complete_stage "36/36四Backend、三长度、三路径成本校准完成并冻结公平配置。"
fi

if should_run K03; then
  begin_stage K03
  ensure_stack
  [[ "$DRY_RUN" == 1 || -f "$FIXED_FROZEN" ]] || { echo "missing K02 frozen fixed config" >&2; exit 1; }
  capacity="V21K03-capacity-4rps-$TAG"; CURRENT_RUNS+=("$capacity")
  MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true \
    run_stage "$capacity" kv "$FIXED_FROZEN" "$PROFILE_ROOT/v2_1_k03_capacity.jsonl" poisson 64 \
    "$TRACE_ROOT/v2_1_capacity/4rps/swe-final-poisson-4rps-r1.jsonl"
  CURRENT_ARTIFACTS+=("$CAPACITY_REPORT")
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: select formal RPS from 2.0,2.5,3.0,3.5,4.0 using 90% capacity rule"
    FINAL_RPS="4"
  else
    set +e
    python -m benchmarks.select_v2_1_formal_rps "$RUN_ROOT/$capacity" --output "$CAPACITY_REPORT"
    selection_status=$?
    set -e
    if ((selection_status == 3)); then
      confirm="V21K03B-confirm-2rps-$TAG"; CURRENT_RUNS+=("$confirm")
      MIN_SUCCESS_RATE=1 REQUIRE_V2_1_WORKER_LIFECYCLE=true \
        run_stage "$confirm" kv "$FIXED_FROZEN" "$PROFILE_ROOT/v2_1_k03_capacity_confirm.jsonl" poisson 64 \
        "$TRACE_ROOT/v2_1_capacity/2rps/swe-final-poisson-2rps-r1.jsonl"
      python -m benchmarks.select_v2_1_formal_rps "$RUN_ROOT/$capacity" \
        --confirmation-run "$RUN_ROOT/$confirm" --output "$CAPACITY_REPORT"
    elif ((selection_status != 0)); then
      exit "$selection_status"
    fi
    FINAL_RPS="$(python - "$CAPACITY_REPORT" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))["selection"]["formal_rps"]
if value is None:
    raise SystemExit("formal RPS was not selected")
print(f"{float(value):g}")
PY
)"
    trace="$(trace_for_rps "$FINAL_RPS")"
    [[ -f "$trace" ]] || { echo "missing frozen formal Trace: $trace" >&2; exit 1; }
    python - "$CAPACITY_REPORT" "$trace" <<'PY'
import json, sys
from benchmarks.io_utils import sha256_file, write_json
path, trace = sys.argv[1:]
report = json.load(open(path))
report["formal_trace"] = {"path": trace, "sha256": sha256_file(trace)}
write_json(path, report)
PY
  fi
  complete_stage "K03按预注册90%容量规则冻结正式RPS；K04后禁止换档。"
fi

if [[ "$DRY_RUN" == 0 && -f "$CAPACITY_REPORT" ]]; then
  FINAL_RPS="$(python - "$CAPACITY_REPORT" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))["selection"]["formal_rps"]
print(f"{float(value):g}")
PY
)"
fi

if should_run K04; then
  begin_stage K04
  ensure_stack
  trace="$(trace_for_rps "$FINAL_RPS")"
  FINAL_FIXED_RUN="V21K04-fixed-before-${FINAL_RPS}rps-$TAG-p${PAIR_ATTEMPT}"; CURRENT_RUNS+=("$FINAL_FIXED_RUN")
  REQUIRE_V2_1_WORKER_LIFECYCLE=true run_stage "$FINAL_FIXED_RUN" kv "$FIXED_FROZEN" \
    "$DATA/swebench.jsonl" poisson 64 "$trace"
  complete_stage "fixed-4096正式Before完成；不单独解释为V2.1收益。"
fi

if should_run K05; then
  begin_stage K05
  ensure_stack
  trace="$(trace_for_rps "$FINAL_RPS")"
  FINAL_ADAPTIVE_RUN="V21K05-adaptive-after-${FINAL_RPS}rps-$TAG-p${PAIR_ATTEMPT}"; CURRENT_RUNS+=("$FINAL_ADAPTIVE_RUN")
  REQUIRE_V2_1_WORKER_LIFECYCLE=true run_stage "$FINAL_ADAPTIVE_RUN" kv "$ADAPTIVE_FROZEN" \
    "$DATA/swebench.jsonl" poisson 64 "$trace"
  complete_stage "Adaptive V2.1正式After完成；只有K07公平校验后才能形成结论。"
fi

if should_run K07; then
  begin_stage K07
  FINAL_FIXED_RUN="${FINAL_FIXED_RUN:-V21K04-fixed-before-${FINAL_RPS}rps-$TAG-p${PAIR_ATTEMPT}}"
  FINAL_ADAPTIVE_RUN="${FINAL_ADAPTIVE_RUN:-V21K05-adaptive-after-${FINAL_RPS}rps-$TAG-p${PAIR_ATTEMPT}}"
  CURRENT_RUNS+=("$FINAL_FIXED_RUN" "$FINAL_ADAPTIVE_RUN")
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY-RUN: analyze paired formal runs and release GPUs"
  else
    python -m benchmarks.analyze_four_h20 \
      --run "fixed-4096=$RUN_ROOT/$FINAL_FIXED_RUN" \
      --run "adaptive-v2-1=$RUN_ROOT/$FINAL_ADAPTIVE_RUN" \
      --output-dir "$RESULT_ROOT/$TAG" \
      --title "Adaptive KV V2.1 Independent Trace (${FINAL_RPS} RPS)"
  fi
  complete_stage "正式配对分析、证据归档和资源清理完成；结论以真实报告为准。"
fi
