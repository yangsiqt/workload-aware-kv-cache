#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
VLLM_SOURCE="${VLLM_SOURCE:-/root/vllm}"
WORKER_PYTHON="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
UV_BIN="${UV_BIN:-/root/.local/bin/uv}"
VLLM_INSTALLED="$($WORKER_PYTHON -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).parent)')"
BACKUP_ROOT="/root/.venvs/kv-worker/lib/python3.12/site-packages/.v2-1-backup/vllm"

ensure_wheel() {
  local manifest="$1"
  local build_script="$2"
  local expected_sha wheel_path
  if [[ -f "$manifest" ]]; then
    read -r expected_sha wheel_path < "$manifest"
    if [[ -f "$wheel_path" ]] && \
      [[ "$(sha256sum "$wheel_path" | awk '{print $1}')" = "$expected_sha" ]]; then
      printf 'Reusing verified wheel: %s\n' "$wheel_path"
      return
    fi
  fi
  "$build_script"
}

ensure_wheel \
  /root/wheels/workload-aware-kv-cache/v2-1/lmcache/lmcache-v2-1.sha256 \
  "$ROOT/scripts/build_v2_1_lmcache_wheel.sh"
ensure_wheel \
  /root/wheels/workload-aware-kv-cache/v2-1/mooncake/mooncake-v2-1.sha256 \
  "$ROOT/scripts/build_v2_1_mooncake_wheel.sh"
read -r lmcache_sha lmcache_wheel < /root/wheels/workload-aware-kv-cache/v2-1/lmcache/lmcache-v2-1.sha256
read -r mooncake_sha mooncake_wheel < /root/wheels/workload-aware-kv-cache/v2-1/mooncake/mooncake-v2-1.sha256
test "$(sha256sum "$lmcache_wheel" | awk '{print $1}')" = "$lmcache_sha"
test "$(sha256sum "$mooncake_wheel" | awk '{print $1}')" = "$mooncake_sha"

"$UV_BIN" pip install --python "$WORKER_PYTHON" --reinstall --no-deps "$lmcache_wheel"
"$WORKER_PYTHON" -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13 mooncake-transfer-engine-non-cuda || true
if ! "$WORKER_PYTHON" -c 'import msgpack' >/dev/null 2>&1; then
  "$UV_BIN" pip install --python "$WORKER_PYTHON" msgpack
fi
"$WORKER_PYTHON" -m pip install --no-deps "$mooncake_wheel"

for relative in \
  v1/core/sched/scheduler.py \
  v1/metrics/stats.py \
  v1/metrics/loggers.py
do
  source_file="$VLLM_SOURCE/vllm/$relative"
  installed_file="$VLLM_INSTALLED/$relative"
  backup_file="$BACKUP_ROOT/$relative"
  test -f "$source_file"
  test -f "$installed_file"
  if [[ ! -f "$backup_file" ]]; then
    mkdir -p "$(dirname "$backup_file")"
    cp -a "$installed_file" "$backup_file"
  fi
  install -m 0644 "$source_file" "$installed_file"
done

source "$ROOT/scripts/activate_four_h20_env.sh"
"$WORKER_PYTHON" -m py_compile \
  "$VLLM_INSTALLED/v1/core/sched/scheduler.py" \
  "$VLLM_INSTALLED/v1/metrics/stats.py" \
  "$VLLM_INSTALLED/v1/metrics/loggers.py"
"$WORKER_PYTHON" -m pip check
"$WORKER_PYTHON" -c 'import lmcache, mooncake.store, vllm; from lmcache.v1.cache_engine import LMCacheEngine; from lmcache.integration.vllm.workload_aware import workload_aware_search_range; assert workload_aware_search_range({"lmcache.workload_aware.selected_path":"lmcache_l1"}) == ["LocalCPUBackend"]; print("V2_1_RUNTIME_OK")'
printf 'lmcache %s %s\nmooncake %s %s\n' "$lmcache_sha" "$lmcache_wheel" "$mooncake_sha" "$mooncake_wheel"
