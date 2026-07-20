#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
source "$PROJECT_ROOT/scripts/activate_four_h20_env.sh"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -c 'import lmcache, mooncake.store, torch, vllm; print(f"torch={torch.__version__} vllm={vllm.__version__} mooncake=OK lmcache=OK")'
ldd "$($PYTHON_BIN -c 'import mooncake.store; print(mooncake.store.__file__)')" | grep 'libcudart.so.13'
