#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "usage: $0 LOG_FILE -- COMMAND [ARG ...]" >&2
  exit 2
fi

LOG_FILE="$1"
shift 2
MIN_FREE_GB="${MIN_FREE_GB:-10}"
LOCK_FILE="${RESOURCE_GUARD_LOCK:-/tmp/workload-aware-kv-cache.guard.lock}"
mkdir -p "$(dirname "$LOG_FILE")"

exec 9>"$LOCK_FILE"
flock 9

available_kb="$(df --output=avail / | tail -1 | tr -d ' ')"
minimum_kb="$((MIN_FREE_GB * 1024 * 1024))"
if (( available_kb < minimum_kb )); then
  printf 'resource_guard: system disk has less than %s GiB free\n' "$MIN_FREE_GB" \
    | tee -a "$LOG_FILE" >&2
  exit 75
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

start_epoch="$(date +%s)"
{
  printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
  printf 'memory_current_before=%s\n' "$(cat /sys/fs/cgroup/memory.current)"
  printf 'memory_events_before=%q\n' "$(tr '\n' ' ' </sys/fs/cgroup/memory.events)"
  df -B1 /
} >>"$LOG_FILE"

set +e
"$@" 2>&1 | tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e

{
  printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'elapsed_seconds=%s\n' "$(( $(date +%s) - start_epoch ))"
  printf 'exit_code=%s\n' "$rc"
  printf 'memory_current_after=%s\n' "$(cat /sys/fs/cgroup/memory.current)"
  printf 'memory_peak=%s\n' "$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
  printf 'memory_events_after=%q\n' "$(tr '\n' ' ' </sys/fs/cgroup/memory.events)"
  df -B1 /
} >>"$LOG_FILE"

exit "$rc"
