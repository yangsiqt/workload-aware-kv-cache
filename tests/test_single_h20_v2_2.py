from __future__ import annotations

from pathlib import Path

from benchmarks.io_utils import read_jsonl
from benchmarks.single_h20_v2_2 import (
    _event_sequences_are_contiguous,
    build_profile,
    normalize_layout_v3,
)


def test_normalize_layout_v3_preserves_locations_and_latest_revision() -> None:
    payload = {
        "layout_info_v3": [
            ["backend", 0, "LocalCPUBackend", 8192, 2],
            ["backend", 0, "RemoteBackend", 16384, 3],
            ["backend", 0, "LocalCPUBackend", 8192, 4],
        ]
    }
    assert normalize_layout_v3(payload) == {
        "LocalCPUBackend": {"tokens": 8192, "revision": 4},
        "RemoteBackend": {"tokens": 16384, "revision": 3},
    }


def test_event_sequence_gate_accepts_batches_and_rejects_gap() -> None:
    assert _event_sequences_are_contiguous(
        [{"sequence": 4}, {"sequence": 4}, {"sequence": 5}]
    )
    assert not _event_sequences_are_contiguous(
        [{"sequence": 4}, {"sequence": 6}]
    )


def test_profile_has_unique_targets_and_fillers(tmp_path: Path) -> None:
    data_root = Path("/root/workload-aware-kv-cache-data")
    manifest = build_profile(data_root, tmp_path)
    rows = list(read_jsonl(Path(manifest["profile_path"])))
    assert manifest["targets"] == [8192, 16384, 32768]
    assert manifest["filler_count"] == 14
    assert len(rows) == 17
    assert len({row["session_id"] for row in rows}) == 17
