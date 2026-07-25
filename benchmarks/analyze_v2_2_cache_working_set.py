from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, write_json


def simulate_session_lru(
    workload_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    *,
    capacity_gib: float,
    bytes_per_token: int,
) -> dict[str, Any]:
    by_request = {str(row["request_id"]): row for row in workload_rows}
    capacity_tokens = int(capacity_gib * 2**30 / bytes_per_token)
    cache: OrderedDict[str, int] = OrderedDict()
    used_tokens = 0
    hits = 0
    reuse_requests = 0
    seen_sessions: set[str] = set()
    for item in trace_rows:
        row = by_request[str(item["request_id"])]
        session_id = str(row["session_id"])
        size = int(row["shared_prefix_tokens"])
        if session_id in seen_sessions:
            reuse_requests += 1
        seen_sessions.add(session_id)
        if session_id in cache:
            used_tokens -= cache.pop(session_id)
            hits += 1
        cache[session_id] = size
        used_tokens += size
        while used_tokens > capacity_tokens and cache:
            _key, evicted = cache.popitem(last=False)
            used_tokens -= evicted
    requests = len(trace_rows)
    return {
        "schema_version": "2.2",
        "result_scope": "SIMULATED_CAPACITY_SCREENING_NOT_PERFORMANCE",
        "capacity_gib": capacity_gib,
        "capacity_tokens": capacity_tokens,
        "bytes_per_token": bytes_per_token,
        "requests": requests,
        "reuse_requests": reuse_requests,
        "simulated_hits": hits,
        "simulated_hit_rate_all_requests": hits / requests if requests else 0.0,
        "simulated_hit_rate_reuse_requests": (
            hits / reuse_requests if reuse_requests else 0.0
        ),
    }


def analyze(
    workload: Path,
    trace: Path,
    *,
    capacity_gib: float = 64.0,
    bytes_per_token: int = 98_304,
) -> dict[str, Any]:
    return simulate_session_lru(
        list(read_jsonl(workload)),
        list(read_jsonl(trace)),
        capacity_gib=capacity_gib,
        bytes_per_token=bytes_per_token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate V2.2 cache working set")
    parser.add_argument("workload", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capacity-gib", type=float, default=64.0)
    parser.add_argument("--bytes-per-token", type=int, default=98_304)
    parser.add_argument("--min-hit-rate", type=float, default=0.10)
    parser.add_argument("--max-hit-rate", type=float, default=0.60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.workload,
        args.trace,
        capacity_gib=args.capacity_gib,
        bytes_per_token=args.bytes_per_token,
    )
    rate = report["simulated_hit_rate_all_requests"]
    report["screening_range"] = [args.min_hit_rate, args.max_hit_rate]
    report["passed"] = args.min_hit_rate <= rate <= args.max_hit_rate
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
