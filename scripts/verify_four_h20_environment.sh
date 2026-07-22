#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
ROUTER_PYTHON="${ROUTER_PYTHON:-/root/.venvs/vllm-router/bin/python}"
source "$PROJECT_ROOT/scripts/activate_four_h20_env.sh"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -c 'import lmcache, mooncake.store, nixl_cu13, torch, vllm; from lmcache.integration.vllm.workload_aware import WorkloadAwareRequest; print(f"torch={torch.__version__} vllm={vllm.__version__} mooncake=OK lmcache_patched=OK nixl=OK")'
"$ROUTER_PYTHON" -c 'import transformers, vllm_router; print(f"router_transformers={transformers.__version__} vllm_router=OK")'
ldd "$($PYTHON_BIN -c 'import mooncake.store; print(mooncake.store.__file__)')" | grep 'libcudart.so.13'
