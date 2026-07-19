from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.io_utils import read_jsonl
from benchmarks.trace_schema import RouteTraceEvent


def validate(path: Path) -> dict[str, object]:
    events = [RouteTraceEvent.model_validate(row) for row in read_jsonl(path)]
    counts = Counter(event.event for event in events)
    decisions = [event for event in events if event.event == "decision"]
    completions = [event for event in events if event.event == "completion"]
    decision_counts = Counter(event.decision_id for event in decisions)
    completion_counts = Counter(event.decision_id for event in completions)
    decision_ids = set(decision_counts)
    completion_ids = set(completion_counts)
    missing = sorted(decision_ids - completion_ids)
    orphan = sorted(completion_ids - decision_ids)
    duplicate_decisions = sorted(
        key for key, count in decision_counts.items() if count != 1
    )
    duplicate_completions = sorted(
        key for key, count in completion_counts.items() if count != 1
    )
    decision_positions = {
        event.decision_id: index
        for index, event in enumerate(events)
        if event.event == "decision"
    }
    invalid_order = sorted(
        event.decision_id
        for index, event in enumerate(events)
        if event.event == "completion"
        and decision_positions.get(event.decision_id, index + 1) >= index
    )
    decisions_by_id = {event.decision_id: event for event in decisions}
    mismatched_events = sorted(
        event.decision_id
        for event in completions
        if event.decision_id in decisions_by_id
        and (
            event.request_id != decisions_by_id[event.decision_id].request_id
            or event.attempt_id != decisions_by_id[event.decision_id].attempt_id
            or event.backend_url != decisions_by_id[event.decision_id].backend_url
        )
    )
    return {
        "valid": not any(
            (
                missing,
                orphan,
                duplicate_decisions,
                duplicate_completions,
                invalid_order,
                mismatched_events,
            )
        ),
        "events": len(events),
        "decisions": counts["decision"],
        "completions": counts["completion"],
        "missing_completions": missing,
        "orphan_completions": orphan,
        "duplicate_decisions": duplicate_decisions,
        "duplicate_completions": duplicate_completions,
        "invalid_order": invalid_order,
        "mismatched_events": mismatched_events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Router JSONL trace")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    report = validate(args.trace)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
