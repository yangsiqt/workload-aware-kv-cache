from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, write_json


def validate_decode_telemetry(
    joined_path: Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    rows = list(read_jsonl(joined_path))
    contexts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    scheduler_enqueued_attempts = 0
    ordered_scheduler_handoffs = 0
    fixed_counterfactuals = 0

    for row in rows:
        attempts = row.get("attempts") or []
        if not attempts:
            continue
        final = attempts[-1]
        decision = final.get("decision") or {}
        context = decision.get("v2_context") or {}
        if context.get("version") == "2.3":
            contexts.append(context)
            fixed = context.get("fixed") or {}
            if all(
                key in fixed
                for key in (
                    "counterfactual_total_ms",
                    "counterfactual_external_kv_ms",
                    "counterfactual_cache_confidence",
                )
            ):
                fixed_counterfactuals += 1
        candidates.extend(decision.get("candidates") or [])

        worker_events = final.get("worker_events") or []
        enqueued = [
            event for event in worker_events if event.get("phase") == "scheduler_enqueued"
        ]
        seen = [
            event for event in worker_events if event.get("phase") == "scheduler_seen"
        ]
        if len(enqueued) == 1:
            scheduler_enqueued_attempts += 1
        if (
            len(enqueued) == 1
            and len(seen) == 1
            and float(enqueued[0].get("recorded_at", 0.0))
            <= float(seen[0].get("recorded_at", 0.0))
        ):
            ordered_scheduler_handoffs += 1

    positive_backlog = sum(
        float(candidate.get("decode_backlog_tokens", 0)) > 0
        for candidate in candidates
    )
    positive_reserved = sum(
        float(candidate.get("reserved_decode_tokens", 0)) > 0
        for candidate in candidates
    )
    positive_remaining = sum(
        float(candidate.get("remaining_decode_tokens", 0)) > 0
        for candidate in candidates
    )
    backend_ewma = sum(
        candidate.get("decode_cost_source") == "backend_ewma"
        and int(candidate.get("decode_throughput_samples", 0)) >= 3
        for candidate in candidates
    )
    backend_urls = {
        str(candidate.get("backend_url"))
        for candidate in candidates
        if candidate.get("backend_url")
    }
    failures: list[str] = []
    if len(rows) != expected_rows:
        failures.append(f"expected {expected_rows} joined rows, observed {len(rows)}")
    if len(contexts) != expected_rows:
        failures.append("successful decisions do not all contain V2.3 context")
    if fixed_counterfactuals != expected_rows:
        failures.append("fixed counterfactual fields are incomplete")
    if len(backend_urls) != 4:
        failures.append("V2.3 candidates do not cover four backends")
    if positive_backlog == 0:
        failures.append("no positive decode backlog was observed")
    if positive_reserved == 0:
        failures.append("no positive reserved decode tokens were observed")
    if positive_remaining == 0:
        failures.append("no positive scheduler remaining decode tokens were observed")
    if backend_ewma == 0:
        failures.append("Backend Decode throughput EWMA never became active")
    if scheduler_enqueued_attempts != expected_rows:
        failures.append("successful attempts do not each have one scheduler_enqueued event")
    if ordered_scheduler_handoffs != expected_rows:
        failures.append("scheduler_enqueued/scheduler_seen lifecycle order is invalid")

    return {
        "schema_version": "2.3",
        "joined_trace": str(joined_path.resolve()),
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "v2_3_contexts": len(contexts),
        "fixed_counterfactuals": fixed_counterfactuals,
        "backend_urls": sorted(backend_urls),
        "positive_decode_backlog_candidates": positive_backlog,
        "positive_reserved_decode_candidates": positive_reserved,
        "positive_remaining_decode_candidates": positive_remaining,
        "backend_decode_ewma_candidates": backend_ewma,
        "scheduler_enqueued_attempts": scheduler_enqueued_attempts,
        "ordered_scheduler_handoffs": ordered_scheduler_handoffs,
        "failures": failures,
        "passed": not failures,
        "performance_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V2.3 Decode telemetry")
    parser.add_argument("joined_trace", type=Path)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_decode_telemetry(
        args.joined_trace,
        expected_rows=args.expected_rows,
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
