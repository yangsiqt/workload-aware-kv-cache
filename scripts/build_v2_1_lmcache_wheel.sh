#!/usr/bin/env bash
set -euo pipefail

LMCACHE_SOURCE="${LMCACHE_SOURCE:-/root/LMCache}"
OUTPUT_DIR="${V2_1_LMCACHE_WHEEL_DIR:-/root/wheels/workload-aware-kv-cache/v2-1/lmcache}"
PYTHON_BIN="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
BUILD_ROOT="${V2_1_LMCACHE_BUILD_ROOT:-/root/autodl-tmp/lmcache-v2-1-source-wheel}"
BUILD_JOBS="${LMCACHE_BUILD_JOBS:-8}"

test -x "$CUDA_HOME/bin/nvcc"
mkdir -p "$OUTPUT_DIR" "$BUILD_ROOT/raw" "$BUILD_ROOT/repaired"
rm -f "$BUILD_ROOT/raw"/*.whl "$BUILD_ROOT/repaired"/*.whl

# Build Python and native CUDA layers from the same V2.1 source revision.  A
# frozen 0.5.1 wheel cannot safely be patched because its c_ops enum/API may no
# longer match the current Python layer.
env \
  PATH="$CUDA_HOME/bin:$PATH" \
  CUDA_HOME="$CUDA_HOME" \
  BUILD_WITH_CUDA=1 \
  ENABLE_CXX11_ABI=1 \
  TORCH_CUDA_ARCH_LIST="${LMCACHE_CUDA_ARCH_LIST:-7.5;9.0}" \
  MAX_JOBS="$BUILD_JOBS" \
  PIP_NO_INDEX="${PIP_NO_INDEX:-1}" \
  SETUPTOOLS_SCM_PRETEND_VERSION="${LMCACHE_WHEEL_VERSION:-0.5.1.post1}" \
  "$PYTHON_BIN" -m build --wheel --no-isolation \
    --outdir "$BUILD_ROOT/raw" "$LMCACHE_SOURCE"

"$PYTHON_BIN" -m auditwheel repair \
  --plat manylinux_2_35_x86_64 \
  --exclude libcuda.so.1 \
  --exclude libcudart.so.13 \
  --exclude libc10.so \
  --exclude libc10_cuda.so \
  --exclude libtorch.so \
  --exclude libtorch_cpu.so \
  --exclude libtorch_python.so \
  --exclude libtorch_cuda.so \
  -w "$BUILD_ROOT/repaired" "$BUILD_ROOT/raw"/*.whl

rm -f "$OUTPUT_DIR"/lmcache-*.whl
install -m 0600 "$BUILD_ROOT/repaired"/*.whl "$OUTPUT_DIR/"
mapfile -t wheels < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'lmcache-*.whl')
[[ "${#wheels[@]}" -eq 1 ]]
sha256sum "${wheels[0]}" | tee "$OUTPUT_DIR/lmcache-v2-1.sha256"
