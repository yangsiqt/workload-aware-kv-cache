from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, write_json


def _csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def validate_run(
    run_dir: Path,
    workload: Path,
    *,
    min_success_rate: float = 0.99,
    required_selected_kv_paths: set[str] | None = None,
    required_actual_kv_paths: set[str] | None = None,
    required_execution_modes: set[str] | None = None,
    require_v2_1_worker_lifecycle: bool = False,
    require_v2_2_worker_lifecycle: bool = False,
    require_v2_3_worker_lifecycle: bool = False,
) -> dict[str, Any]:
    workload_ids = {str(row["request_id"]) for row in read_jsonl(workload)}
    requests = list(read_jsonl(run_dir / "requests.jsonl"))
    request_ids = [str(row["request_id"]) for row in requests]
    successful = [row for row in requests if row.get("success")]
    success_rate = len(successful) / len(workload_ids) if workload_ids else 0.0
    joined = list(read_jsonl(run_dir / "joined_trace.jsonl"))
    joined_ids = [str(row["request_id"]) for row in joined]

    selected_paths = {
        str(row["selected_kv_path"])
        for row in successful
        if row.get("selected_kv_path")
    }
    execution_modes = {
        str(row["selected_execution_mode"])
        for row in successful
        if row.get("selected_execution_mode")
    }
    actual_rows = [
        row
        for path in sorted(run_dir.glob("connector_actual_trace_gpu*.jsonl"))
        for row in read_jsonl(path)
        if (
            row.get("event_type") == "actual_retrieve"
            or (
                row.get("event_type") == "kv_execution_feedback"
                and row.get("phase") in {"worker_retrieve", "load_completed"}
            )
        )
        and str(row.get("request_id", "")) in workload_ids
    ]
    actual_paths = {
        str(row["actual_kv_path"]) for row in actual_rows if row.get("actual_kv_path")
    }

    failures: list[str] = []
    lifecycle_attempts = 0
    scheduler_seen_attempts = 0
    terminal_attempts = 0
    path_mismatches: list[str] = []
    for row in joined:
        if not row.get("client", {}).get("success"):
            continue
        attempts = row.get("attempts") or []
        if not attempts:
            continue
        final = attempts[-1]
        if not final.get("completion", {}).get("success"):
            continue
        lifecycle_attempts += 1
        worker_events = final.get("worker_events") or []
        if sum(event.get("phase") == "scheduler_seen" for event in worker_events) == 1:
            scheduler_seen_attempts += 1
        terminals = [event for event in worker_events if event.get("terminal") is True]
        if (
            sum(event.get("phase") == "request_finished" for event in worker_events)
            == 1
            and len(terminals) == 1
        ):
            terminal_attempts += 1
        selected = str(
            (final.get("decision", {}).get("kv_path") or {}).get("selected_path", "")
        )
        for event in worker_events:
            if event.get("phase") != "load_completed":
                continue
            actual = str(event.get("actual_kv_path", ""))
            if event.get("path_mismatch") or (
                selected in {"lmcache_l1", "mooncake_l2"} and selected != actual
            ):
                path_mismatches.append(
                    f"{final.get('decision', {}).get('decision_id')}:"
                    f"{selected}/{actual}"
                )
    if len(request_ids) != len(set(request_ids)):
        failures.append("duplicate request IDs in client results")
    if set(request_ids) != workload_ids:
        failures.append("client result IDs do not exactly match workload")
    if success_rate < min_success_rate:
        failures.append(
            f"success rate {success_rate:.4f} is below {min_success_rate:.4f}"
        )
    if len(joined_ids) != len(set(joined_ids)) or set(joined_ids) != workload_ids:
        failures.append("joined Trace IDs do not exactly match workload")
    require_worker_lifecycle = (
        require_v2_1_worker_lifecycle
        or require_v2_2_worker_lifecycle
        or require_v2_3_worker_lifecycle
    )
    if require_worker_lifecycle:
        if scheduler_seen_attempts != lifecycle_attempts:
            failures.append(
                "successful final attempts do not each have one scheduler_seen event"
            )
        if terminal_attempts != lifecycle_attempts:
            failures.append(
                "successful final attempts do not each have one Worker terminal event"
            )
        if path_mismatches:
            failures.append(f"selected/actual KV path mismatch: {path_mismatches[:5]}")
    for label, required, observed in (
        ("selected KV", required_selected_kv_paths or set(), selected_paths),
        ("actual KV", required_actual_kv_paths or set(), actual_paths),
        ("execution mode", required_execution_modes or set(), execution_modes),
    ):
        missing = required - observed
        if missing:
            failures.append(f"missing required {label} evidence: {sorted(missing)}")

    report = {
        "schema_version": (
            "2.3"
            if require_v2_3_worker_lifecycle
            else "2.2"
            if require_v2_2_worker_lifecycle
            else "2.1"
            if require_v2_1_worker_lifecycle
            else "1.0"
        ),
        "run_dir": str(run_dir.resolve()),
        "workload": str(workload.resolve()),
        "expected_requests": len(workload_ids),
        "observed_requests": len(requests),
        "successful_requests": len(successful),
        "success_rate": success_rate,
        "joined_trace_rows": len(joined),
        "selected_kv_paths": sorted(selected_paths),
        "actual_kv_paths": sorted(actual_paths),
        "actual_retrieve_rows": len(actual_rows),
        "execution_modes": sorted(execution_modes),
        "worker_lifecycle_attempts": lifecycle_attempts,
        "scheduler_seen_attempts": scheduler_seen_attempts,
        "worker_terminal_attempts": terminal_attempts,
        "path_mismatches": path_mismatches,
        "failures": failures,
        "passed": not failures,
    }
    write_json(run_dir / "validation.json", report)
    if failures:
        raise ValueError("; ".join(failures))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("workload", type=Path)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--require-selected-kv-paths", default="")
    parser.add_argument("--require-actual-kv-paths", default="")
    parser.add_argument("--require-execution-modes", default="")
    parser.add_argument("--require-v2-1-worker-lifecycle", action="store_true")
    parser.add_argument("--require-v2-2-worker-lifecycle", action="store_true")
    parser.add_argument("--require-v2-3-worker-lifecycle", action="store_true")
    args = parser.parse_args()
    report = validate_run(
        args.run_dir,
        args.workload,
        min_success_rate=args.min_success_rate,
        required_selected_kv_paths=_csv_set(args.require_selected_kv_paths),
        required_actual_kv_paths=_csv_set(args.require_actual_kv_paths),
        required_execution_modes=_csv_set(args.require_execution_modes),
        require_v2_1_worker_lifecycle=args.require_v2_1_worker_lifecycle,
        require_v2_2_worker_lifecycle=args.require_v2_2_worker_lifecycle,
        require_v2_3_worker_lifecycle=args.require_v2_3_worker_lifecycle,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
