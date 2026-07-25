#!/usr/bin/env bash
set -euo pipefail

VLLM_SOURCE="${VLLM_SOURCE:-/root/vllm/vllm}"
LMCACHE_SOURCE="${LMCACHE_SOURCE:-/root/LMCache/lmcache}"
WORKER_PYTHON="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
VLLM_INSTALLED="${VLLM_INSTALLED:-/root/miniconda3/lib/python3.12/site-packages/vllm}"
LMCACHE_INSTALLED="${LMCACHE_INSTALLED:-/root/.venvs/kv-worker/lib/python3.12/site-packages/lmcache}"
BACKUP_ROOT="${V2_2_BACKUP_ROOT:-/root/.venvs/kv-worker/lib/python3.12/site-packages/.v2-2-backup}"
MANIFEST_DIR="${V2_2_OVERLAY_DIR:-/root/wheels/workload-aware-kv-cache/v2-2/python-overlay}"

declare -a vllm_files=(
  "v1/core/sched/scheduler.py"
  "distributed/kv_transfer/kv_connector/v1/base.py"
  "distributed/kv_transfer/kv_connector/v1/lmcache_connector.py"
)
declare -a lmcache_files=(
  "integration/vllm/workload_aware.py"
  "integration/vllm/vllm_v1_adapter.py"
  "integration/vllm/lmcache_connector_v1.py"
  "v1/cache_engine.py"
  "v1/api_server/__main__.py"
  "v1/cache_controller/message.py"
  "v1/cache_controller/controllers/kv_controller.py"
)

install_one() {
  local component="$1" source_root="$2" installed_root="$3" relative="$4"
  local source="$source_root/$relative" installed="$installed_root/$relative"
  local backup="$BACKUP_ROOT/$component/$relative"
  test -f "$source"
  test -f "$installed"
  if [[ ! -f "$backup" ]]; then
    mkdir -p "$(dirname "$backup")"
    cp -a "$installed" "$backup"
  fi
  install -m 0644 "$source" "$installed"
}

for relative in "${vllm_files[@]}"; do
  install_one vllm "$VLLM_SOURCE" "$VLLM_INSTALLED" "$relative"
done
for relative in "${lmcache_files[@]}"; do
  install_one lmcache "$LMCACHE_SOURCE" "$LMCACHE_INSTALLED" "$relative"
done

mkdir -p "$MANIFEST_DIR"
manifest="$MANIFEST_DIR/python-overlay-v2-2.sha256"
: >"$manifest"
for relative in "${vllm_files[@]}"; do
  sha256sum "$VLLM_INSTALLED/$relative" >>"$manifest"
done
for relative in "${lmcache_files[@]}"; do
  sha256sum "$LMCACHE_INSTALLED/$relative" >>"$manifest"
done

"$WORKER_PYTHON" -m py_compile \
  "${vllm_files[@]/#/$VLLM_INSTALLED/}" \
  "${lmcache_files[@]/#/$LMCACHE_INSTALLED/}"
"$WORKER_PYTHON" - <<'PY'
import inspect

from lmcache.integration.vllm.workload_aware import WorkloadAwareResultTracker
from lmcache.v1.api_server.__main__ import create_app
from lmcache.v1.cache_controller.message import LookupRetMsg
from vllm.v1.core.sched.scheduler import Scheduler

assert hasattr(Scheduler, "prefix_cache_generation")
assert hasattr(WorkloadAwareResultTracker, "set_backend_generation")
assert "layout_info_v3" in LookupRetMsg.__struct_fields__
assert "layout_info_v3" in inspect.getsource(create_app)
print("V2_2_PYTHON_OVERLAY_OK")
PY
echo "$manifest"
