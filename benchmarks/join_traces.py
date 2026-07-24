from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, write_jsonl
from benchmarks.schemas import RequestResult
from benchmarks.trace_schema import JoinedTrace, RouteAttempt, RouteTraceEvent
from benchmarks.validate_trace import validate


def _worker_key(row: dict[str, Any]) -> tuple[str, int, str]:
    request_id = str(row.get("request_id", ""))
    backend_id = str(row.get("backend_id", ""))
    try:
        attempt_id = int(row.get("attempt_id", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Worker attempt_id for {request_id!r}") from exc
    if not request_id or not backend_id:
        raise ValueError("Worker 2.1 events require request_id and backend_id")
    return request_id, attempt_id, backend_id


def _load_worker_events(paths: list[Path]) -> dict[tuple[str, int, str], list[dict]]:
    events: dict[tuple[str, int, str], list[dict]] = {}
    seen: set[tuple[str, int, str, str, str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("schema_version") != "2.1":
                continue
            if row.get("event_type") != "kv_execution_feedback":
                continue
            key = _worker_key(row)
            phase = str(row.get("phase", ""))
            decision_id = str(row.get("decision_id", ""))
            load_attempt_id = (
                str(row.get("load_attempt_id", "legacy"))
                if phase in {"load_started", "load_completed"}
                else "single"
            )
            identity = (*key, decision_id, phase, load_attempt_id)
            if identity in seen:
                raise ValueError(
                    f"duplicate Worker phase {phase!r} for {key[0]}:{key[1]}"
                )
            seen.add(identity)
            events.setdefault(key, []).append(row)
    for rows in events.values():
        rows.sort(key=lambda row: float(row.get("recorded_at", 0.0)))
    return events


def _validate_successful_final_attempt(attempt: RouteAttempt) -> None:
    events = attempt.worker_events
    phases = Counter(str(row.get("phase", "")) for row in events)
    label = attempt.decision.decision_id
    if phases["scheduler_seen"] != 1:
        raise ValueError(f"{label} requires exactly one scheduler_seen event")
    terminals = [row for row in events if row.get("terminal") is True]
    if phases["request_finished"] != 1 or len(terminals) != 1:
        raise ValueError(f"{label} requires exactly one Worker terminal event")

    selected = str((attempt.decision.kv_path or {}).get("selected_path", ""))
    load_completed = [row for row in events if row.get("phase") == "load_completed"]
    if selected in {"lmcache_l1", "mooncake_l2"}:
        if not load_completed:
            terminal = terminals[0]
            actual = str(terminal.get("actual_kv_path", ""))
            fallback_reason = str(terminal.get("fallback_reason", ""))
            if actual not in {"local_hbm", "recompute"} or not fallback_reason:
                raise ValueError(
                    f"{label} requires a load_completed event or explicit "
                    "Local HBM/Recompute fallback"
                )
        for event in load_completed:
            actual = str(event.get("actual_kv_path", ""))
            if event.get("path_mismatch") or actual != selected:
                raise ValueError(
                    f"{label} selected/actual KV path mismatch: {selected}/{actual}"
                )
    elif load_completed:
        actual = str(load_completed[-1].get("actual_kv_path", ""))
        if actual in {"lmcache_l1", "mooncake_l2"}:
            raise ValueError(f"{label} unexpectedly used external KV path {actual}")


def join(
    client_path: Path,
    route_path: Path,
    output_path: Path,
    worker_paths: list[Path] | None = None,
) -> int:
    client_rows = [
        RequestResult.model_validate(value) for value in read_jsonl(client_path)
    ]
    if len(client_rows) != len({row.request_id for row in client_rows}):
        raise ValueError("client request IDs must be unique")
    clients = {row.request_id: row for row in client_rows}
    report = validate(route_path)
    if not report["valid"]:
        raise ValueError(f"invalid route trace: {report}")
    decisions: dict[str, RouteTraceEvent] = {}
    completions: dict[str, RouteTraceEvent] = {}
    for value in read_jsonl(route_path):
        event = RouteTraceEvent.model_validate(value)
        target = decisions if event.event == "decision" else completions
        target[event.decision_id] = event
    worker_events = _load_worker_events(worker_paths or [])
    by_request: dict[str, list[RouteAttempt]] = {}
    for decision_id, decision in decisions.items():
        completion = completions[decision_id]
        key = (decision.request_id, decision.attempt_id, decision.backend_url)
        events = worker_events.get(key, [])
        if events:
            for row in events:
                if str(row.get("decision_id", "")) != decision_id:
                    raise ValueError(
                        f"Worker decision_id mismatch for {decision_id}: "
                        f"{row.get('decision_id')!r}"
                    )
        by_request.setdefault(decision.request_id, []).append(
            RouteAttempt(
                attempt_id=decision.attempt_id,
                decision=decision,
                completion=completion,
                worker_events=events,
            )
        )
    missing = sorted(set(clients) - set(by_request))
    if missing:
        raise ValueError(f"missing completion traces for {len(missing)} requests")
    extra = sorted(set(by_request) - set(clients))
    if extra:
        raise ValueError(f"route trace contains {len(extra)} unknown requests")
    rows = []
    for request_id, client in sorted(clients.items()):
        attempts = sorted(by_request[request_id], key=lambda item: item.attempt_id)
        if worker_paths and client.success and attempts[-1].completion.success:
            _validate_successful_final_attempt(attempts[-1])
        rows.append(
            JoinedTrace(
                request_id=request_id,
                client=client.model_dump(mode="json"),
                route=attempts[-1].completion,
                attempts=attempts,
            )
        )
    return write_jsonl(output_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join client and Router request traces"
    )
    parser.add_argument("client", type=Path)
    parser.add_argument("router", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--worker-trace",
        action="append",
        type=Path,
        default=[],
        help="LMCache 2.1 scheduler or Worker trace (repeatable)",
    )
    args = parser.parse_args()
    print(join(args.client, args.router, args.output, args.worker_trace))


if __name__ == "__main__":
    main()
