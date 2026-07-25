from __future__ import annotations

from pathlib import Path

from benchmarks.generate_v2_2_arrival_traces import generate_trace
from benchmarks.analyze_v2_2_cache_working_set import simulate_session_lru
from benchmarks.io_utils import write_jsonl
from benchmarks.validate_v2_2_activation import validate_activation


def test_wave_trace_separates_turns_and_preserves_session_order() -> None:
    rows = [
        {
            "request_id": f"{session}-t{turn}",
            "session_id": session,
            "turn_id": turn,
        }
        for session in ("a", "b", "c")
        for turn in (1, 2, 3)
    ]
    trace = generate_trace(rows, seed=53)
    ids = [item.request_id for item in trace]
    assert all(value.endswith("t1") for value in ids[:3])
    assert all(value.endswith("t2") for value in ids[3:6])
    assert all(value.endswith("t3") for value in ids[6:])
    assert [item.offset_s for item in trace] == sorted(item.offset_s for item in trace)
    assert abs(trace[-1].offset_s - 9 / 2.5) < 1e-9


def _joined_row(index: int, *, changed: bool, hit: bool = True) -> dict:
    adaptive_path = "lmcache_l1" if changed else "local_hbm"
    fixed_path = "recompute" if changed else "local_hbm"
    events = [
        {
            "phase": "scheduler_seen",
            "backend_generation": "boot:0",
            "vllm_cached_tokens": 0,
        }
    ]
    if changed:
        events.append(
            {
                "phase": "load_completed",
                "actual_kv_path": adaptive_path if hit else "recompute",
                "path_mismatch": not hit,
            }
        )
    return {
        "request_id": f"r{index}",
        "attempts": [
            {
                "decision": {
                    "decision_id": f"r{index}:0",
                    "v2_context": {
                        "guard_reason": (
                            "v2_override_fixed" if changed else "insufficient_gain"
                        ),
                        "adaptive": {"backend_url": "a", "kv_path": adaptive_path},
                        "fixed": {"backend_url": "b", "kv_path": fixed_path},
                    },
                },
                "worker_events": events,
            }
        ],
    }


def test_activation_gate_counts_real_path_changes(tmp_path: Path) -> None:
    path = tmp_path / "joined.jsonl"
    write_jsonl(path, [_joined_row(i, changed=i < 6) for i in range(10)])
    report = validate_activation(
        path,
        expected_rows=10,
        min_overrides=6,
        min_path_changes=6,
        min_external_overrides=1,
        min_external_hit_rate=0.95,
    )
    assert report["passed"]
    assert report["external_actual_hit_rate"] == 1.0


def test_activation_gate_rejects_worker_path_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "joined.jsonl"
    write_jsonl(path, [_joined_row(0, changed=True, hit=False)])
    report = validate_activation(
        path,
        expected_rows=1,
        min_overrides=1,
        min_path_changes=1,
        min_external_overrides=1,
        min_external_hit_rate=0.95,
    )
    assert not report["passed"]
    assert report["path_mismatches"] == ["r0:0"]


def test_activation_gate_rejects_local_hbm_prediction_that_recomputed(
    tmp_path: Path,
) -> None:
    row = _joined_row(0, changed=False)
    context = row["attempts"][0]["decision"]["v2_context"]
    context["guard_reason"] = "v2_override_fixed"
    context["adaptive"] = {"backend_url": "a", "kv_path": "local_hbm"}
    context["fixed"] = {"backend_url": "b", "kv_path": "recompute"}
    path = tmp_path / "joined.jsonl"
    write_jsonl(path, [row])
    report = validate_activation(
        path,
        expected_rows=1,
        min_overrides=1,
        min_path_changes=1,
        min_external_overrides=0,
        min_external_hit_rate=0.95,
    )
    assert not report["passed"]
    assert report["path_mismatches"] == ["r0:0"]


def test_capacity_screening_is_marked_simulated() -> None:
    rows = [
        {
            "request_id": f"{session}-t{turn}",
            "session_id": session,
            "turn_id": turn,
            "shared_prefix_tokens": 4,
        }
        for turn in range(2)
        for session in ("a", "b")
    ]
    trace = [{"request_id": row["request_id"]} for row in rows]
    report = simulate_session_lru(
        rows,
        trace,
        capacity_gib=1,
        bytes_per_token=1,
    )
    assert report["result_scope"] == "SIMULATED_CAPACITY_SCREENING_NOT_PERFORMANCE"
    assert report["simulated_hits"] == 2
