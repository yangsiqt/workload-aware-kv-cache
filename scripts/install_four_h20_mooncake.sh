#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
WHEEL="/root/wheels/workload-aware-kv-cache/mooncake_transfer_engine_cuda13-0.3.11.post1-cp312-cp312-manylinux_2_35_x86_64.whl"
EXPECTED_SHA256="1f0b62ef625bf017eb4f1717d7240bf72ba22cb9c0abd93fcd28ed53bbb492b9"

test -x "$PYTHON_BIN"
test -f "$WHEEL"
test "$(sha256sum "$WHEEL" | awk '{print $1}')" = "$EXPECTED_SHA256"
"$PYTHON_BIN" -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13 || true
"$PYTHON_BIN" -m pip install --no-deps "$WHEEL"
source "$PROJECT_ROOT/scripts/activate_four_h20_env.sh"
"$PYTHON_BIN" -c 'import mooncake.store; print("MOONCAKE_STORE_OK")'
