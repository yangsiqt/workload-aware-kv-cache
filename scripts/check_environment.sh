#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="${LOG_DIR:-/root/log/workload-aware-kv-cache/environment}"
MODEL_DIR="${MODEL_DIR:-${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507}}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/check-$(date -u +%Y%m%dT%H%M%SZ).log}"

{
  date -u --iso-8601=seconds
  uname -a
  df -h / /root/autodl-tmp /root/autodl-fs
  nvidia-smi || true
  nvidia-smi topo -m || true
  nvcc --version || true
  ffmpeg -version | head -n 2 || true
  git lfs version || true
  python - <<'PY'
import platform
import torch
import vllm

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("vllm", vllm.__version__)
PY
  if vllm serve --help >/dev/null 2>&1; then
    echo "vllm_cli_ok=true"
  else
    echo "vllm_cli_ok=false"
  fi
  echo "model_dir=$MODEL_DIR"
  if [[ -d "$MODEL_DIR" ]]; then
    find "$MODEL_DIR" -maxdepth 1 -type f -printf '%f %s\n' | sort
    echo "model_file_count=$(find "$MODEL_DIR" -maxdepth 1 -type f | wc -l)"
    echo "model_total_bytes=$(find "$MODEL_DIR" -maxdepth 1 -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f\n", sum}')"
  else
    echo "model_missing=true"
  fi
} 2>&1 | tee "$LOG_FILE"

echo "$LOG_FILE"
