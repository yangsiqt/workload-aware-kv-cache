import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmarks.generate_arrival_traces import generate
from benchmarks.io_utils import read_jsonl, write_jsonl
from benchmarks.join_traces import join
from benchmarks.run_benchmark import validate_arrival_trace, validate_workload_ids
from benchmarks.schemas import (
    ArrivalTraceItem,
    ChatMessage,
    RequestResult,
    SourceInfo,
    WorkloadItem,
)
from benchmarks.validate_trace import validate


def workload_item(request_id: str, session_id: str, turn_id: int) -> WorkloadItem:
    return WorkloadItem(
        dataset_name="SWE-bench Verified",
        dataset_revision="revision",
        dataset_instance_id=session_id,
        request_id=request_id,
        session_id=session_id,
        turn_id=turn_id,
        priority=1,
        request_type="code_agent_multiturn",
        prefix_hash=f"prefix-{session_id}",
        messages=[ChatMessage(role="user", content="test")],
        prompt_tokens=10,
        shared_prefix_tokens=0,
        expected_output_tokens=2,
        source=SourceInfo(dataset="test", license="test", snapshot_id="snapshot"),
    )


def request_result(request_id: str) -> RequestResult:
    return RequestResult(
        run_id="run",
        request_id=request_id,
        session_id="session",
        turn_id=0,
        dataset_name="test",
        request_type="test",
        prefix_hash="prefix",
        priority=1,
        route_policy="P-R3",
        offered_at_s=0,
        started_at_s=1,
        completed_at_s=2,
        e2e_ms=1000,
        input_tokens=10,
        output_tokens=2,
        success=True,
    )


def route_event(request_id: str, event: str, success=None, attempt_id: int = 0) -> dict:
    return {
        "schema_version": "1.0",
        "event": event,
        "request_id": request_id,
        "attempt_id": attempt_id,
        "decision_id": f"{request_id}:{attempt_id}",
        "policy": "P-R3",
        "backend_url": "http://backend-a",
        "reason": "queue_prefill_slo_cost",
        "metadata": {"request_id": request_id},
        "candidates": [
            {
                "backend_url": "http://backend-a",
                "running": 0,
                "waiting": 0,
                "cached_tokens": 0,
                "cache_source": "none",
                "cache_confidence": 0,
                "queue_ms": 0,
                "prefill_ms": 10,
                "slo_penalty_ms": 0,
                "total_ms": 10,
                "stale": False,
            }
        ],
        "decided_at": 1.0,
        "success": success,
        "error": "",
    }


def test_arrival_traces_are_reproducible_and_cover_workload(tmp_path: Path) -> None:
    workload = tmp_path / "workload.jsonl"
    write_jsonl(
        workload,
        [
            workload_item("a0", "a", 0),
            workload_item("a1", "a", 1),
            workload_item("b0", "b", 0),
        ],
    )
    first = generate(workload, tmp_path / "first", 2.0, [42])[0]
    second = generate(workload, tmp_path / "second", 2.0, [42])[0]
    assert first.read_bytes() == second.read_bytes()
    rows = list(read_jsonl(first))
    assert {row["request_id"] for row in rows} == {"a0", "a1", "b0"}
    assert [row["offset_s"] for row in rows] == sorted(row["offset_s"] for row in rows)


def test_arrival_validation_rejects_duplicate_ids_and_unsorted_offsets() -> None:
    items = [workload_item("a", "a", 0), workload_item("b", "b", 0)]
    validate_workload_ids(items)
    with pytest.raises(ValueError, match="must be unique"):
        validate_workload_ids([items[0], items[0]])
    with pytest.raises(ValueError, match="must be unique"):
        validate_arrival_trace(
            items,
            [
                ArrivalTraceItem(request_id="a", offset_s=0),
                ArrivalTraceItem(request_id="a", offset_s=1),
            ],
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_arrival_trace(
            items,
            [
                ArrivalTraceItem(request_id="a", offset_s=1),
                ArrivalTraceItem(request_id="b", offset_s=0),
            ],
        )


def test_trace_validation_and_join(tmp_path: Path) -> None:
    client = tmp_path / "client.jsonl"
    router = tmp_path / "router.jsonl"
    output = tmp_path / "joined.jsonl"
    write_jsonl(client, [request_result("request")])
    write_jsonl(
        router,
        [
            route_event("request", "decision"),
            route_event("request", "completion", True),
        ],
    )

    assert validate(router)["valid"]
    assert join(client, router, output) == 1
    joined = json.loads(output.read_text())
    assert joined["request_id"] == "request"
    assert len(joined["attempts"]) == 1


def test_trace_join_rejects_missing_completion(tmp_path: Path) -> None:
    client = tmp_path / "client.jsonl"
    router = tmp_path / "router.jsonl"
    write_jsonl(client, [request_result("request")])
    write_jsonl(router, [route_event("request", "decision")])
    with pytest.raises(ValueError, match="invalid route trace"):
        join(client, router, tmp_path / "joined.jsonl")


def test_trace_validation_rejects_duplicates_and_orphans(tmp_path: Path) -> None:
    router = tmp_path / "router.jsonl"
    write_jsonl(
        router,
        [
            route_event("request", "decision"),
            route_event("request", "decision"),
            route_event("request", "completion", True),
            route_event("orphan", "completion", False),
        ],
    )
    report = validate(router)
    assert not report["valid"]
    assert report["duplicate_decisions"] == ["request:0"]
    assert report["orphan_completions"] == ["orphan:0"]


def test_trace_validation_rejects_mismatched_attempt_pair(tmp_path: Path) -> None:
    router = tmp_path / "router.jsonl"
    decision = route_event("request", "decision")
    completion = route_event("other", "completion", True)
    completion["decision_id"] = decision["decision_id"]
    write_jsonl(router, [decision, completion])

    report = validate(router)
    assert not report["valid"]
    assert report["mismatched_events"] == ["request:0"]


def test_trace_join_preserves_failover_attempts(tmp_path: Path) -> None:
    client = tmp_path / "client.jsonl"
    router = tmp_path / "router.jsonl"
    output = tmp_path / "joined.jsonl"
    write_jsonl(client, [request_result("request")])
    write_jsonl(
        router,
        [
            route_event("request", "decision", attempt_id=0),
            route_event("request", "completion", False, attempt_id=0),
            route_event("request", "decision", attempt_id=1),
            route_event("request", "completion", True, attempt_id=1),
        ],
    )

    assert join(client, router, output) == 1
    joined = json.loads(output.read_text())
    assert [item["attempt_id"] for item in joined["attempts"]] == [0, 1]
    assert joined["route"]["decision_id"] == "request:1"


def test_environment_log_file_can_be_overridden(tmp_path: Path) -> None:
    log = tmp_path / "environment.log"
    result = subprocess.run(
        ["bash", "scripts/check_environment.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "LOG_FILE": str(log)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert log.exists()
    assert result.stdout.rstrip().endswith(str(log))
