from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.io_utils import read_jsonl, write_jsonl
from benchmarks.schemas import RequestResult
from benchmarks.trace_schema import JoinedTrace, RouteAttempt, RouteTraceEvent
from benchmarks.validate_trace import validate


def join(client_path: Path, route_path: Path, output_path: Path) -> int:
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
    by_request: dict[str, list[RouteAttempt]] = {}
    for decision_id, decision in decisions.items():
        completion = completions[decision_id]
        by_request.setdefault(decision.request_id, []).append(
            RouteAttempt(
                attempt_id=decision.attempt_id,
                decision=decision,
                completion=completion,
            )
        )
    missing = sorted(set(clients) - set(by_request))
    if missing:
        raise ValueError(f"missing completion traces for {len(missing)} requests")
    extra = sorted(set(by_request) - set(clients))
    if extra:
        raise ValueError(f"route trace contains {len(extra)} unknown requests")
    rows = [
        JoinedTrace(
            request_id=request_id,
            client=client.model_dump(mode="json"),
            route=sorted(by_request[request_id], key=lambda item: item.attempt_id)[
                -1
            ].completion,
            attempts=sorted(by_request[request_id], key=lambda item: item.attempt_id),
        )
        for request_id, client in sorted(clients.items())
    ]
    return write_jsonl(output_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join client and Router request traces"
    )
    parser.add_argument("client", type=Path)
    parser.add_argument("router", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(join(args.client, args.router, args.output))


if __name__ == "__main__":
    main()
