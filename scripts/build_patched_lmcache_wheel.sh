#!/usr/bin/env bash
set -euo pipefail

BASE_WHEEL="${BASE_LMCACHE_WHEEL:-/root/wheels/workload-aware-kv-cache/lmcache-0.5.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl}"
LMCACHE_SOURCE="${LMCACHE_SOURCE:-/root/LMCache}"
OUTPUT_DIR="${PATCHED_WHEEL_DIR:-/root/wheels/workload-aware-kv-cache/patched}"
PYTHON_BIN="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
BUILD_DIR="$(mktemp -d /tmp/lmcache-wheel.XXXXXX)"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$LMCACHE_SOURCE" show -s --format=%ct HEAD)}"

cleanup() {
  rm -rf -- "$BUILD_DIR"
}
trap cleanup EXIT

test -f "$BASE_WHEEL"
test -f "$LMCACHE_SOURCE/lmcache/integration/vllm/workload_aware.py"
test -f "$LMCACHE_SOURCE/lmcache/integration/vllm/vllm_v1_adapter.py"
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" -m wheel unpack "$BASE_WHEEL" --dest "$BUILD_DIR"
mapfile -t unpacked < <(find "$BUILD_DIR" -mindepth 1 -maxdepth 1 -type d)
if [[ "${#unpacked[@]}" -ne 1 ]]; then
  echo "Expected exactly one unpacked wheel directory" >&2
  exit 1
fi

package_root="${unpacked[0]}"
install -m 0644 \
  "$LMCACHE_SOURCE/lmcache/integration/vllm/workload_aware.py" \
  "$package_root/lmcache/integration/vllm/workload_aware.py"
install -m 0644 \
  "$LMCACHE_SOURCE/lmcache/integration/vllm/vllm_v1_adapter.py" \
  "$package_root/lmcache/integration/vllm/vllm_v1_adapter.py"
find "$package_root" -exec touch -h -d "@$SOURCE_DATE_EPOCH" {} +

rm -f "$OUTPUT_DIR"/lmcache-0.5.1-*.whl
"$PYTHON_BIN" -m wheel pack "$package_root" --dest-dir "$OUTPUT_DIR"
patched_wheels=("$OUTPUT_DIR"/lmcache-0.5.1-*.whl)
if [[ "${#patched_wheels[@]}" -ne 1 || ! -f "${patched_wheels[0]}" ]]; then
  echo "Patched wheel was not created" >&2
  exit 1
fi
sha256sum "${patched_wheels[0]}" | tee "$OUTPUT_DIR/lmcache-0.5.1-workload-aware.sha256"
