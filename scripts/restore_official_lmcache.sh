#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
BASE_WHEEL="${BASE_LMCACHE_WHEEL:-/root/wheels/workload-aware-kv-cache/lmcache-0.5.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl}"

test -f "$BASE_WHEEL"
"$PYTHON_BIN" -m pip install --no-deps --force-reinstall "$BASE_WHEEL"
"$PYTHON_BIN" -m pip check
