#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
WHEEL_DIR="${PATCHED_WHEEL_DIR:-/root/wheels/workload-aware-kv-cache/patched}"
SHA_FILE="$WHEEL_DIR/lmcache-0.5.1-workload-aware.sha256"

test -f "$SHA_FILE"
read -r expected_sha wheel_path <"$SHA_FILE"
if [[ ! -f "$wheel_path" ]]; then
  wheel_path="$WHEEL_DIR/$(basename "$wheel_path")"
fi
test -f "$wheel_path"
test "$(sha256sum "$wheel_path" | awk '{print $1}')" = "$expected_sha"

"$PYTHON_BIN" -m pip install --no-deps --force-reinstall "$wheel_path"
source "$PROJECT_ROOT/scripts/activate_four_h20_env.sh"
"$PYTHON_BIN" -c 'from lmcache.integration.vllm.workload_aware import WorkloadAwareRequest; print("PATCHED_LMCACHE_OK")'
