from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.io_utils import read_jsonl, write_jsonl
from benchmarks.schemas import RequestResult
from benchmarks.trace_schema import JoinedTrace, RouteTraceEvent


def join(client_path: Path, route_path: Path, output_path: Path) -> int:
    clients = {
        row.request_id: row
        for row in (
            RequestResult.model_validate(value) for value in read_jsonl(client_path)
        )
    }
    completions = {}
    for value in read_jsonl(route_path):
        event = RouteTraceEvent.model_validate(value)
        if event.event == "completion":
            completions[event.request_id] = event
    missing = sorted(set(clients) - set(completions))
    if missing:
        raise ValueError(f"missing completion traces for {len(missing)} requests")
    rows = [
        JoinedTrace(
            request_id=request_id,
            client=client.model_dump(mode="json"),
            route=completions[request_id],
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
