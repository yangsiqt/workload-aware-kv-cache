from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, write_json

EXTERNAL_PATHS = {"lmcache_l1", "mooncake_l2"}


def _actual_path(events: list[dict[str, Any]]) -> str:
    # request_finished may intentionally retain the generic
    # ``lmcache_external`` marker. Prefer the tier-specific load result.
    for phase in ("load_completed", "request_finished", "lookup_completed"):
        for event in reversed(events):
            actual_path = str(event.get("actual_kv_path", ""))
            if (
                event.get("phase") == phase
                and actual_path
                and actual_path not in {"pending", "lmcache_external"}
            ):
                return actual_path
    scheduler = next(
        (event for event in reversed(events) if event.get("phase") == "scheduler_seen"),
        None,
    )
    if scheduler is None:
        return ""
    return (
        "local_hbm"
        if int(scheduler.get("vllm_cached_tokens", 0) or 0) > 0
        else "recompute"
    )


def activation_metrics(joined_path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(joined_path))
    overrides = 0
    path_changes = 0
    external_path_changes = 0
    external_path_changes_confirmed = 0
    adaptive_external_overrides = 0
    adaptive_external_hits = 0
    mismatches: list[str] = []
    generation_rows = 0
    refresh_attempts = 0
    refresh_failures = 0
    hbm_event_rows_raw = 0
    hbm_event_rows = 0
    hbm_event_eligible_rows = 0
    for row in rows:
        attempts = row.get("attempts") or []
        if not attempts:
            continue
        final = attempts[-1]
        decision = final.get("decision") or {}
        refresh = decision.get("cache_tier_refresh") or {}
        refresh_attempts += bool(refresh.get("attempted"))
        refresh_failures += bool(refresh.get("timed_out") or refresh.get("error"))
        has_hbm_event = any(
            candidate.get("cache_source") == "vllm_kv_event"
            for candidate in decision.get("candidates") or []
        )
        hbm_event_rows_raw += has_hbm_event
        raw_turn_id = (row.get("client") or {}).get("turn_id")
        try:
            hbm_event_eligible = raw_turn_id is None or int(raw_turn_id) > 0
        except (TypeError, ValueError):
            hbm_event_eligible = True
        if hbm_event_eligible:
            hbm_event_eligible_rows += 1
            hbm_event_rows += has_hbm_event
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

        events = final.get("worker_events") or []
        actual_path = _actual_path(events)
        selected_matches_actual = bool(actual_path) and actual_path == adaptive_path
        path_changes += changed and adaptive_path != fixed_path and selected_matches_actual
        expected_external_miss_fallback = adaptive_path in EXTERNAL_PATHS and any(
            event.get("fallback_reason") == "external_miss" for event in events
        )
        if (
            overridden
            and not selected_matches_actual
            and not expected_external_miss_fallback
        ):
            mismatches.append(str(decision.get("decision_id", "")))
        generation_rows += any(
            event.get("phase") == "scheduler_seen"
            and bool(event.get("backend_generation"))
            for event in events
        )
        external_involved = changed and bool(
            {adaptive_path, fixed_path} & EXTERNAL_PATHS
        )
        if external_involved:
            external_path_changes += 1
            external_path_changes_confirmed += selected_matches_actual
        if overridden and adaptive_path in EXTERNAL_PATHS:
            adaptive_external_overrides += 1
            if actual_path == adaptive_path:
                adaptive_external_hits += 1

    return {
        "schema_version": "2.2",
        "joined_rows": len(rows),
        "adaptive_overrides": overrides,
        "kv_path_changes": path_changes,
        # Keep legacy names for report/CLI compatibility. V2.2 treats either
        # entering an external tier or avoiding an external restore as an
        # external-involved path change.
        "external_overrides": external_path_changes,
        "external_actual_hits": external_path_changes_confirmed,
        "external_actual_hit_rate": (
            external_path_changes_confirmed / external_path_changes
            if external_path_changes
            else None
        ),
        "external_path_changes": external_path_changes,
        "external_path_changes_confirmed": external_path_changes_confirmed,
        "external_path_change_confirmation_rate": (
            external_path_changes_confirmed / external_path_changes
            if external_path_changes
            else None
        ),
        "adaptive_external_overrides": adaptive_external_overrides,
        "adaptive_external_actual_hits": adaptive_external_hits,
        "adaptive_external_actual_hit_rate": (
            adaptive_external_hits / adaptive_external_overrides
            if adaptive_external_overrides
            else None
        ),
        "scheduler_generation_rows": generation_rows,
        "cache_tier_refresh_attempts": refresh_attempts,
        "cache_tier_refresh_failures": refresh_failures,
        "cache_tier_refresh_failure_rate": (
            refresh_failures / refresh_attempts if refresh_attempts else None
        ),
        "hbm_event_rows": hbm_event_rows,
        "hbm_event_rows_raw": hbm_event_rows_raw,
        "hbm_event_eligible_rows": hbm_event_eligible_rows,
        "hbm_event_coverage": (
            hbm_event_rows / hbm_event_eligible_rows
            if hbm_event_eligible_rows
            else 0.0
        ),
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
    min_refresh_coverage: float = 0.0,
    max_refresh_failure_rate: float = 1.0,
    min_hbm_event_coverage: float = 0.0,
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
        failures.append("external-involved path change count below gate")
    rate = report["external_actual_hit_rate"]
    if report["external_overrides"] and (rate is None or rate < min_external_hit_rate):
        failures.append("external-involved path change confirmation rate below gate")
    if report["path_mismatches"]:
        failures.append("selected/actual path mismatch")
    refresh_coverage = report["cache_tier_refresh_attempts"] / max(
        report["joined_rows"], 1
    )
    if refresh_coverage < min_refresh_coverage:
        failures.append("Controller refresh coverage below gate")
    refresh_failure_rate = report["cache_tier_refresh_failure_rate"]
    if (
        refresh_failure_rate is not None
        and refresh_failure_rate > max_refresh_failure_rate
    ):
        failures.append("Controller refresh failure rate above gate")
    if report["hbm_event_coverage"] < min_hbm_event_coverage:
        failures.append("vLLM KV event coverage below gate")
    report["thresholds"] = {
        "expected_rows": expected_rows,
        "min_overrides": min_overrides,
        "min_path_changes": min_path_changes,
        "min_external_overrides": min_external_overrides,
        "min_external_hit_rate": min_external_hit_rate,
        "min_refresh_coverage": min_refresh_coverage,
        "max_refresh_failure_rate": max_refresh_failure_rate,
        "min_hbm_event_coverage": min_hbm_event_coverage,
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
    parser.add_argument("--min-external-overrides", type=int, default=24)
    parser.add_argument("--min-external-hit-rate", type=float, default=0.95)
    parser.add_argument("--min-refresh-coverage", type=float, default=0.99)
    parser.add_argument("--max-refresh-failure-rate", type=float, default=0.01)
    parser.add_argument("--min-hbm-event-coverage", type=float, default=0.90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_activation(
        args.joined_trace,
        expected_rows=args.expected_rows,
        min_overrides=args.min_overrides,
        min_path_changes=args.min_path_changes,
        min_external_overrides=args.min_external_overrides,
        min_external_hit_rate=args.min_external_hit_rate,
        min_refresh_coverage=args.min_refresh_coverage,
        max_refresh_failure_rate=args.max_refresh_failure_rate,
        min_hbm_event_coverage=args.min_hbm_event_coverage,
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
