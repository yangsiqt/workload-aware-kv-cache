from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, sha256_file, write_json


CANDIDATE_RPS = (2.0, 2.5, 3.0, 3.5, 4.0)


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _metric_series(
    rows: list[dict[str, Any]], origin: float, arrival_end: float
) -> list[tuple[float, float]]:
    buckets: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("source", "vllm") != "vllm" or row.get("error"):
            continue
        timestamp = _timestamp(str(row["timestamp"]))
        if timestamp < origin or timestamp > arrival_end:
            continue
        bucket = int((timestamp - origin) * 4)
        buckets[bucket][str(row["backend_id"])] = float(row.get("waiting", 0))
    return [
        (bucket / 4.0, sum(values.values()))
        for bucket, values in sorted(buckets.items())
        if values
    ]


def _slope(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return math.inf
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0:
        return math.inf
    return (
        sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in points)
        / denominator
    )


def choose_formal_rps(
    capacity_rps: float,
    *,
    stable_at_four: bool,
    two_rps_confirmed: bool = False,
) -> dict[str, Any]:
    if stable_at_four:
        return {
            "status": "selected",
            "capacity_rps": capacity_rps,
            "target_rps": 4.0,
            "formal_rps": 4.0,
            "reason": "four_rps_stable",
        }
    if capacity_rps < 2.0:
        return {
            "status": "blocked",
            "capacity_rps": capacity_rps,
            "target_rps": capacity_rps * 0.9,
            "formal_rps": None,
            "reason": "capacity_below_two_rps",
        }
    target = capacity_rps * 0.9
    eligible = [value for value in CANDIDATE_RPS if value <= target + 1e-9]
    if eligible:
        selected = max(eligible)
        return {
            "status": "selected",
            "capacity_rps": capacity_rps,
            "target_rps": target,
            "formal_rps": selected,
            "reason": "ninety_percent_round_down",
        }
    if two_rps_confirmed:
        return {
            "status": "selected",
            "capacity_rps": capacity_rps,
            "target_rps": target,
            "formal_rps": 2.0,
            "reason": "two_rps_confirmation_passed",
        }
    return {
        "status": "needs_2rps_confirmation",
        "capacity_rps": capacity_rps,
        "target_rps": target,
        "formal_rps": None,
        "reason": "ninety_percent_target_below_two_rps",
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    requests = list(read_jsonl(run_dir / "requests.jsonl"))
    metrics = list(read_jsonl(run_dir / "backend_metrics.jsonl"))
    if not requests:
        raise ValueError("capacity run has no requests")
    validation_path = run_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    successful = [row for row in requests if row.get("success")]
    success_rate = len(successful) / len(requests)
    first_started = min(float(row["started_at_s"]) for row in requests)
    last_completed = max(float(row["completed_at_s"]) for row in requests)
    elapsed = max(last_completed - first_started, 1e-9)
    capacity_rps = len(successful) / elapsed

    offered = [float(row.get("offered_at_s") or 0.0) for row in requests]
    origin = min(
        float(row["started_at_s"]) - float(row.get("offered_at_s") or 0.0)
        for row in requests
    )
    arrival_duration = max(offered)
    arrival_end = origin + arrival_duration
    drain_s = max(0.0, last_completed - arrival_end)
    waiting = _metric_series(metrics, origin, arrival_end)
    second_half = [point for point in waiting if point[0] >= arrival_duration / 2]
    waiting_slope = _slope(second_half)
    q3 = [
        value
        for offset, value in waiting
        if arrival_duration * 0.5 <= offset < arrival_duration * 0.75
    ]
    q4 = [
        value
        for offset, value in waiting
        if arrival_duration * 0.75 <= offset <= arrival_duration
    ]
    q3_mean = statistics.fmean(q3) if q3 else math.inf
    q4_mean = statistics.fmean(q4) if q4 else math.inf

    vllm_rows = [row for row in metrics if row.get("source", "vllm") == "vllm"]
    metric_error_rate = (
        sum(bool(row.get("error")) for row in metrics) / len(metrics)
        if metrics
        else 1.0
    )
    preemptions = 0.0
    for backend in {str(row.get("backend_id")) for row in vllm_rows}:
        values = [
            float(row.get("preemptions_total", 0))
            for row in vllm_rows
            if str(row.get("backend_id")) == backend and not row.get("error")
        ]
        if values:
            preemptions += max(values) - min(values)
    backend_requests = Counter(
        str(row.get("backend_id")) for row in successful if row.get("backend_id")
    )
    max_running = max(
        (float(row.get("running", 0)) for row in vllm_rows if not row.get("error")),
        default=0.0,
    )
    min_free_blocks = min(
        (
            float(row.get("kv_cache_free_blocks", 0))
            for row in vllm_rows
            if not row.get("error") and float(row.get("kv_cache_total_blocks", 0)) > 0
        ),
        default=0.0,
    )
    stable = all(
        (
            len(requests) == 120,
            success_rate == 1.0,
            bool(validation.get("passed")),
            capacity_rps >= 3.8,
            waiting_slope <= 0.05,
            q4_mean <= q3_mean + 1.0,
            drain_s <= 10.0,
            preemptions == 0.0,
            metric_error_rate <= 0.01,
        )
    )
    return {
        "requests": len(requests),
        "successful_requests": len(successful),
        "success_rate": success_rate,
        "capacity_rps": capacity_rps,
        "arrival_duration_s": arrival_duration,
        "drain_s": drain_s,
        "waiting_second_half_slope": waiting_slope,
        "waiting_q3_mean": q3_mean,
        "waiting_q4_mean": q4_mean,
        "preemptions_delta": preemptions,
        "metric_error_rate": metric_error_rate,
        "max_backend_running": max_running,
        "min_kv_cache_free_blocks": min_free_blocks,
        "backend_request_distribution": dict(sorted(backend_requests.items())),
        "validation_passed": bool(validation.get("passed")),
        "stable_at_four_rps": stable,
        "run_dir": str(run_dir.resolve()),
        "requests_sha256": sha256_file(run_dir / "requests.jsonl"),
    }


def confirmation_passed(report: dict[str, Any]) -> bool:
    return all(
        (
            report["requests"] == 60,
            report["success_rate"] == 1.0,
            report["validation_passed"],
            report["waiting_second_half_slope"] <= 0.05,
            report["waiting_q4_mean"] <= report["waiting_q3_mean"] + 1.0,
            report["drain_s"] <= 10.0,
            report["preemptions_delta"] == 0.0,
            report["metric_error_rate"] <= 0.01,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--confirmation-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capacity = analyze_run(args.run_dir)
    confirmation = None
    confirmed = False
    if args.confirmation_run:
        confirmation = analyze_run(args.confirmation_run)
        confirmed = confirmation_passed(confirmation)
    selection = choose_formal_rps(
        capacity["capacity_rps"],
        stable_at_four=capacity["stable_at_four_rps"],
        two_rps_confirmed=confirmed,
    )
    report = {
        "schema_version": "2.1",
        "policy": {
            "candidates": list(CANDIDATE_RPS),
            "capacity_fraction": 0.9,
            "rounding": "down_to_0.5_rps",
        },
        "capacity": capacity,
        "confirmation": confirmation,
        "selection": selection,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if selection["status"] == "blocked":
        raise SystemExit(2)
    if selection["status"] == "needs_2rps_confirmation" and not args.confirmation_run:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
