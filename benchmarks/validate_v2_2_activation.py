from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, write_json

EXTERNAL_PATHS = {"lmcache_l1", "mooncake_l2"}


def activation_metrics(joined_path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(joined_path))
    overrides = 0
    path_changes = 0
    external_overrides = 0
    external_hits = 0
    mismatches: list[str] = []
    generation_rows = 0
    for row in rows:
        attempts = row.get("attempts") or []
        if not attempts:
            continue
        final = attempts[-1]
        decision = final.get("decision") or {}
        context = decision.get("v2_context") or {}
        adaptive = context.get("adaptive") or {}
        fixed = context.get("fixed") or {}
        adaptive_path = str(adaptive.get("kv_path", ""))
        fixed_path = str(fixed.get("kv_path", ""))
        overridden = str(context.get("guard_reason", "")) == "v2_override_fixed"
        overrides += overridden
        changed = overridden and (
            adaptive.get("backend_url") != fixed.get("backend_url")
            or adaptive_path != fixed_path
        )
        path_changes += changed and adaptive_path != fixed_path

        events = final.get("worker_events") or []
        generation_rows += any(
            event.get("phase") == "scheduler_seen"
            and bool(event.get("backend_generation"))
            for event in events
        )
        if overridden and adaptive_path in EXTERNAL_PATHS:
            external_overrides += 1
            actual = [
                event for event in events if event.get("phase") == "load_completed"
            ]
            if any(event.get("actual_kv_path") == adaptive_path for event in actual):
                external_hits += 1
            if any(
                event.get("path_mismatch")
                or str(event.get("actual_kv_path", "")) != adaptive_path
                for event in actual
            ):
                mismatches.append(str(decision.get("decision_id", "")))

    return {
        "schema_version": "2.2",
        "joined_rows": len(rows),
        "adaptive_overrides": overrides,
        "kv_path_changes": path_changes,
        "external_overrides": external_overrides,
        "external_actual_hits": external_hits,
        "external_actual_hit_rate": (
            external_hits / external_overrides if external_overrides else None
        ),
        "scheduler_generation_rows": generation_rows,
        "path_mismatches": mismatches,
    }


def validate_activation(
    joined_path: Path,
    *,
    expected_rows: int,
    min_overrides: int,
    min_path_changes: int,
    min_external_overrides: int,
    min_external_hit_rate: float,
) -> dict[str, Any]:
    report = activation_metrics(joined_path)
    failures = []
    if report["joined_rows"] != expected_rows:
        failures.append("joined row count mismatch")
    if report["adaptive_overrides"] < min_overrides:
        failures.append("adaptive override count below gate")
    if report["kv_path_changes"] < min_path_changes:
        failures.append("KV path change count below gate")
    if report["external_overrides"] < min_external_overrides:
        failures.append("external override count below gate")
    rate = report["external_actual_hit_rate"]
    if report["external_overrides"] and (rate is None or rate < min_external_hit_rate):
        failures.append("external override actual-hit rate below gate")
    if report["path_mismatches"]:
        failures.append("selected/actual path mismatch")
    report["thresholds"] = {
        "expected_rows": expected_rows,
        "min_overrides": min_overrides,
        "min_path_changes": min_path_changes,
        "min_external_overrides": min_external_overrides,
        "min_external_hit_rate": min_external_hit_rate,
    }
    report["failures"] = failures
    report["passed"] = not failures
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V2.2 path activation")
    parser.add_argument("joined_trace", type=Path)
    parser.add_argument("--expected-rows", type=int, default=1200)
    parser.add_argument("--min-overrides", type=int, default=60)
    parser.add_argument("--min-path-changes", type=int, default=24)
    parser.add_argument("--min-external-overrides", type=int, default=1)
    parser.add_argument("--min-external-hit-rate", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_activation(
        args.joined_trace,
        expected_rows=args.expected_rows,
        min_overrides=args.min_overrides,
        min_path_changes=args.min_path_changes,
        min_external_overrides=args.min_external_overrides,
        min_external_hit_rate=args.min_external_hit_rate,
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
