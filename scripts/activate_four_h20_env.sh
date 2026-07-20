#!/usr/bin/env bash
# Source this file before starting LMCache or Mooncake on the CUDA 13 image.
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
