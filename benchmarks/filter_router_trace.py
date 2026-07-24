from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.io_utils import read_jsonl, write_jsonl


def filter_trace(client_path: Path, raw_trace_path: Path, output_path: Path) -> int:
    request_ids = {str(row["request_id"]) for row in read_jsonl(client_path)}
    rows = [
        row
        for row in read_jsonl(raw_trace_path)
        if str(row.get("request_id", "")) in request_ids
    ]
    return write_jsonl(output_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove non-measurement Router control probes from a trace"
    )
    parser.add_argument("client", type=Path)
    parser.add_argument("raw_trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(filter_trace(args.client, args.raw_trace, args.output))


if __name__ == "__main__":
    main()
