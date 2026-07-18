import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmarks.generate_arrival_traces import generate
from benchmarks.io_utils import read_jsonl, write_jsonl
from benchmarks.join_traces import join
from benchmarks.schemas import ChatMessage, RequestResult, SourceInfo, WorkloadItem
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


def route_event(request_id: str, event: str, success=None) -> dict:
    return {
        "schema_version": "1.0",
        "event": event,
        "request_id": request_id,
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
    assert json.loads(output.read_text())["request_id"] == "request"


def test_trace_join_rejects_missing_completion(tmp_path: Path) -> None:
    client = tmp_path / "client.jsonl"
    router = tmp_path / "router.jsonl"
    write_jsonl(client, [request_result("request")])
    write_jsonl(router, [route_event("request", "decision")])
    with pytest.raises(ValueError, match="missing completion"):
        join(client, router, tmp_path / "joined.jsonl")


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
