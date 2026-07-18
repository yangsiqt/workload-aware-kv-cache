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
    decisions = {event.request_id for event in events if event.event == "decision"}
    completions = {event.request_id for event in events if event.event == "completion"}
    missing = sorted(decisions - completions)
    return {
        "valid": not missing and counts["decision"] == counts["completion"],
        "events": len(events),
        "decisions": counts["decision"],
        "completions": counts["completion"],
        "missing_completions": missing,
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
