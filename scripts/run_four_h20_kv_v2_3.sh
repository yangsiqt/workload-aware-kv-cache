#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export FOUR_H20_VERSION_LABEL=V2.3
export FOUR_H20_RUN_NAME_VERSION=V23
export FOUR_H20_TAG_PREFIX=v2-3
export FOUR_H20_CONTROL_PREFIX=V23-window
export FOUR_H20_ADAPTIVE_TEMPLATE="$ROOT/configs/four_h20/agent-slo-kv-adaptive-v2-3.yaml"
export FOUR_H20_FIXED_CONFIG="$ROOT/configs/four_h20/agent-slo-kv-fixed-4096-v2-3.yaml"
export FOUR_H20_ADAPTIVE_FROZEN_NAME=adaptive-v2-3-frozen.yaml
export FOUR_H20_READINESS_SCRIPT="$ROOT/scripts/check_v2_3_readiness.sh"
export FOUR_H20_RECORD_MODULE=benchmarks.record_v2_3_stage
export FOUR_H20_ANALYZE_MODULE=benchmarks.analyze_v2_3_pair
export FOUR_H20_PAIR_REPORT_NAME=v2-3-pair-report.json
export FOUR_H20_EXTRA_VLLM_METRIC=vllm:remaining_decode_tokens
export REQUIRE_V2_3_WORKER_LIFECYCLE=true
export FOUR_H20_REPLICATE_TRACE="${FOUR_H20_REPLICATE_TRACE:-/root/workload-aware-kv-cache-data/traces/four_h20/v2_2/v2-2-replicate-cohort30-bursty-2.5rps.jsonl}"
export RESULT_ROOT="${RESULT_ROOT:-/root/performance-results/workload-aware-kv-cache/four-h20/adaptive-kv-v2-3}"

exec "$ROOT/scripts/run_four_h20_kv_v2_2.sh" "$@"
