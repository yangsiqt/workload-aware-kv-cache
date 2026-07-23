#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
export V2_LITE_LOG_ROOT="${V2_1_LOG_ROOT:-/root/log/workload-aware-kv-cache/v2-1-2080ti}"
export SMOKE_LMCACHE_CONFIG="${SMOKE_LMCACHE_CONFIG:-$ROOT/configs/v2_1/lmcache-2080ti.yaml}"
export SMOKE_ROUTER_CONFIG="${SMOKE_ROUTER_CONFIG:-$ROOT/configs/v2_1/agent-slo-2080ti.yaml}"
export SMOKE_LMCACHE_INSTANCE_ID="${SMOKE_LMCACHE_INSTANCE_ID:-v2-1-2080ti}"
export SMOKE_MOONCAKE_CLIENT_METRICS_PORT="${SMOKE_MOONCAKE_CLIENT_METRICS_PORT:-9300}"
exec "$ROOT/scripts/v2_lite_2080ti_stack.sh" "$@"
