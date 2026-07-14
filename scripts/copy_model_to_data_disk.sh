#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-/root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507}"
DEST="${2:-/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507}"
EXPECTED_FILES="${EXPECTED_FILES:-28}"
EXPECTED_BYTES="${EXPECTED_BYTES:-61084263662}"

mkdir -p "$(dirname "$DEST")"
rsync -ah --info=progress2 "$SOURCE/" "$DEST/"
actual_files=$(find "$DEST" -maxdepth 1 -type f | wc -l)
actual_bytes=$(find "$DEST" -maxdepth 1 -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')
printf 'files=%s expected=%s\nbytes=%s expected=%s\n' "$actual_files" "$EXPECTED_FILES" "$actual_bytes" "$EXPECTED_BYTES"
[[ "$actual_files" -eq "$EXPECTED_FILES" && "$actual_bytes" -eq "$EXPECTED_BYTES" ]]
