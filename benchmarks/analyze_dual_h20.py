from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.io_utils import read_jsonl, write_json


def _metric_summary(path: Path) -> dict[str, object]:
    rows = [row for row in read_jsonl(path) if row.get("error") is None]
    by_backend: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_backend[str(row["backend_id"])].append(row)
    backends = {}
    total_queries = 0.0
    total_hits = 0.0
    for backend_id, values in sorted(by_backend.items()):
        first, last = values[0], values[-1]
        queries = max(
            0.0, float(last.get("prefix_queries", 0)) - float(first.get("prefix_queries", 0))
        )
        hits = max(
            0.0, float(last.get("prefix_hits", 0)) - float(first.get("prefix_hits", 0))
        )
        total_queries += queries
        total_hits += hits
        backends[backend_id] = {
            "samples": len(values),
            "prefix_queries_delta": queries,
            "prefix_hits_delta": hits,
            "prefix_hit_rate": hits / queries if queries else None,
            "max_running": max(float(row.get("running", 0)) for row in values),
            "max_waiting": max(float(row.get("waiting", 0)) for row in values),
            "max_kv_usage": max(float(row.get("kv_usage", 0)) for row in values),
        }
    return {
        "backends": backends,
        "prefix_queries_delta": total_queries,
        "prefix_hits_delta": total_hits,
        "prefix_hit_rate": total_hits / total_queries if total_queries else None,
    }


def analyze(requests_path: Path, metrics_path: Path) -> dict[str, object]:
    requests = list(read_jsonl(requests_path))
    successful = [row for row in requests if row.get("success")]
    backend_counts = Counter(
        str(row["backend_id"]) for row in successful if row.get("backend_id")
    )
    transitions = 0
    migrations = 0
    by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in successful:
        by_session[str(row["session_id"])].append(row)
    for values in by_session.values():
        ordered = sorted(values, key=lambda row: int(row["turn_id"]))
        for left, right in zip(ordered, ordered[1:]):
            if left.get("backend_id") and right.get("backend_id"):
                transitions += 1
                migrations += left["backend_id"] != right["backend_id"]
    total_routed = sum(backend_counts.values())
    load_skew = None
    if total_routed and len(backend_counts) == 2:
        counts = list(backend_counts.values())
        load_skew = abs(counts[0] - counts[1]) / total_routed
    metrics = _metric_summary(metrics_path)
    queries = float(metrics["prefix_queries_delta"])
    hits = float(metrics["prefix_hits_delta"])
    return {
        "requests": len(requests),
        "successful_requests": len(successful),
        "backend_request_counts": dict(sorted(backend_counts.items())),
        "load_skew": load_skew,
        "session_transitions": transitions,
        "session_migrations": migrations,
        "session_migration_rate": migrations / transitions if transitions else None,
        "estimated_uncached_tokens": max(0.0, queries - hits),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.requests, args.metrics)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
