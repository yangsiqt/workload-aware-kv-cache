from pathlib import Path

from benchmarks.filter_router_trace import filter_trace
from benchmarks.io_utils import read_jsonl, write_jsonl


def test_filter_trace_keeps_only_measured_request_events(tmp_path: Path) -> None:
    client = tmp_path / "requests.jsonl"
    raw = tmp_path / "raw.jsonl"
    output = tmp_path / "filtered.jsonl"
    write_jsonl(client, [{"request_id": "measured"}])
    write_jsonl(
        raw,
        [
            {"request_id": "refresh", "event": "decision"},
            {"request_id": "refresh", "event": "completion"},
            {"request_id": "measured", "event": "decision"},
            {"request_id": "measured", "event": "completion"},
        ],
    )
    assert filter_trace(client, raw, output) == 2
    assert [row["request_id"] for row in read_jsonl(output)] == [
        "measured",
        "measured",
    ]
