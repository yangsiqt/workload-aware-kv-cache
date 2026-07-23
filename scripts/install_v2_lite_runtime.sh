#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/root/workload-aware-kv-cache}"
VLLM_SOURCE="${VLLM_SOURCE:-/root/vllm}"
WORKER_PYTHON="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
WORKER_SITE="/root/.venvs/kv-worker/lib/python3.12/site-packages"
VLLM_INSTALLED="$($WORKER_PYTHON -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).parent)')"
BACKUP_ROOT="$WORKER_SITE/.v2-lite-backup/vllm"
UV_BIN="${UV_BIN:-/root/.local/bin/uv}"

test -x "$WORKER_PYTHON"
test -x "$UV_BIN"
test -d "$VLLM_INSTALLED"
[[ "$($WORKER_PYTHON -c 'import vllm; print(vllm.__version__)')" == "0.25.0" ]]

"$ROOT/scripts/build_patched_lmcache_wheel.sh"
patched_wheel="$(find /root/wheels/workload-aware-kv-cache/patched -maxdepth 1 -type f -name 'lmcache-0.5.1-*.whl' -print -quit)"
test -f "$patched_wheel"
"$UV_BIN" pip install --python "$WORKER_PYTHON" --reinstall --no-deps "$patched_wheel"

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

"$WORKER_PYTHON" -m py_compile \
  "$VLLM_INSTALLED/v1/core/sched/scheduler.py" \
  "$VLLM_INSTALLED/v1/metrics/stats.py" \
  "$VLLM_INSTALLED/v1/metrics/loggers.py"

sha256sum "$patched_wheel"
echo "V2 Lite runtime installed into /root/.venvs/kv-worker"
