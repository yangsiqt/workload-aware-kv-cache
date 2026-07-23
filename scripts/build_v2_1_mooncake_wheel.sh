#!/usr/bin/env bash
set -euo pipefail

MOONCAKE_SOURCE="${MOONCAKE_SOURCE:-/root/Mooncake}"
BUILD_DIR="${V2_1_MOONCAKE_BUILD_DIR:-/root/autodl-tmp/mooncake-v2-1-cuda13-build}"
DEPS_PREFIX="${V2_1_MOONCAKE_DEPS_PREFIX:-/root/autodl-tmp/mooncake-v2-1-deps}"
OUTPUT_DIR="${V2_1_MOONCAKE_WHEEL_DIR:-/root/wheels/workload-aware-kv-cache/v2-1/mooncake}"
PYTHON_BIN="${KV_WORKER_PYTHON:-/root/.venvs/kv-worker/bin/python}"
JOBS="${MOONCAKE_BUILD_JOBS:-6}"
CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-/usr/local/cuda-13.0/bin/nvcc}"
BUILD_LINK="$MOONCAKE_SOURCE/build-v2-1"

cleanup() {
  [[ -L "$BUILD_LINK" ]] && unlink "$BUILD_LINK"
  for relative in \
    mooncake-wheel/mooncake/async_store.py \
    mooncake-wheel/mooncake/fabric_allocator_utils.py \
    mooncake-wheel/mooncake/mooncake_client
  do
    if ! git -C "$MOONCAKE_SOURCE" ls-files --error-unmatch "$relative" \
      >/dev/null 2>&1; then
      rm -f -- "$MOONCAKE_SOURCE/$relative"
    fi
  done
}
trap cleanup EXIT

test -x "$PYTHON_BIN"
test -x "$CUDA_COMPILER"
test -f "$DEPS_PREFIX/lib/cmake/yalantinglibs/yalantinglibsConfig.cmake"
mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"

cmake -S "$MOONCAKE_SOURCE" -B "$BUILD_DIR" \
  -DCMAKE_PREFIX_PATH="$DEPS_PREFIX" \
  -DCMAKE_CXX_FLAGS="-isystem $DEPS_PREFIX/include" \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DPYTHON_EXECUTABLE="$PYTHON_BIN" \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DWITH_STORE=ON -DWITH_STORE_RUST=OFF -DWITH_TE=ON \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=ON -DBUILD_BENCHMARK=OFF \
  -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel "$JOBS"

if [[ -L "$BUILD_LINK" ]]; then
  unlink "$BUILD_LINK"
elif [[ -e "$BUILD_LINK" ]]; then
  echo "$BUILD_LINK exists and is not a symlink" >&2
  exit 1
fi
ln -s "$BUILD_DIR" "$BUILD_LINK"

rm -f "$OUTPUT_DIR"/*.whl
(
  cd "$MOONCAKE_SOURCE"
  PATH="$(dirname "$PYTHON_BIN"):$PATH" \
  PYTHON_VERSION=3.12 CU13_BUILD=1 NO_BUILD_ISOLATION=1 BUILD_DIR=build-v2-1 \
    bash scripts/build_wheel.sh 3.12 "$OUTPUT_DIR"
)
mapfile -t wheels < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.whl')
[[ "${#wheels[@]}" -eq 1 ]]
sha256sum "${wheels[0]}" | tee "$OUTPUT_DIR/mooncake-v2-1.sha256"
