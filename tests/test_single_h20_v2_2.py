from __future__ import annotations

from pathlib import Path

from benchmarks.io_utils import read_jsonl
from benchmarks.single_h20_v2_2 import (
    _event_sequences_are_contiguous,
    _validate_lifecycle,
    build_profile,
    normalize_layout_v3,
)
from benchmarks.trace_schema import CandidateTrace


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


def test_trace_schema_accepts_authoritative_vllm_kv_event_source() -> None:
    score = CandidateTrace(
        backend_url="http://127.0.0.1:8000",
        running=0,
        waiting=0,
        cached_tokens=8192,
        cache_source="vllm_kv_event",
        cache_confidence=1.0,
        queue_ms=0.0,
        prefill_ms=0.0,
        slo_penalty_ms=0.0,
        total_ms=0.0,
        stale=False,
    )
    assert score.cache_source == "vllm_kv_event"


def test_lifecycle_validation_uses_causal_cross_file_order() -> None:
    identity = {
        "schema_version": "2.2",
        "request_id": "request",
        "attempt_id": "0",
        "decision_id": "request:0",
        "backend_id": "backend",
    }
    rows = [
        {**identity, "phase": "scheduler_seen", "recorded_at": 1.0},
        {**identity, "phase": "load_started", "recorded_at": 2.0},
        {**identity, "phase": "load_completed", "recorded_at": 3.0},
        {
            **identity,
            "phase": "request_finished",
            "recorded_at": 4.0,
            "terminal": True,
        },
    ]
    report = _validate_lifecycle(rows, {"request"})
    assert report["phase_order_valid"] is True
