from datetime import UTC, datetime
from pathlib import Path

from benchmarks.io_utils import write_json, write_jsonl
from benchmarks.select_v2_1_formal_rps import analyze_run, choose_formal_rps


def test_capacity_selection_uses_four_when_stable() -> None:
    result = choose_formal_rps(3.85, stable_at_four=True)
    assert result["formal_rps"] == 4.0
    assert result["reason"] == "four_rps_stable"


def test_capacity_selection_rounds_ninety_percent_down() -> None:
    assert choose_formal_rps(3.9, stable_at_four=False)["formal_rps"] == 3.5
    assert choose_formal_rps(3.49, stable_at_four=False)["formal_rps"] == 3.0
    assert choose_formal_rps(3.0, stable_at_four=False)["formal_rps"] == 2.5


def test_capacity_selection_requires_confirmation_near_two() -> None:
    pending = choose_formal_rps(2.1, stable_at_four=False)
    assert pending["status"] == "needs_2rps_confirmation"
    confirmed = choose_formal_rps(2.1, stable_at_four=False, two_rps_confirmed=True)
    assert confirmed["formal_rps"] == 2.0
    assert choose_formal_rps(1.99, stable_at_four=False)["status"] == "blocked"


def test_analyze_capacity_accepts_clean_four_rps_run(tmp_path: Path) -> None:
    rows = []
    for index in range(120):
        offered = index / 4
        rows.append(
            {
                "request_id": f"r{index}",
                "backend_id": f"gpu{index % 4}",
                "offered_at_s": offered,
                "started_at_s": 1000 + offered,
                "completed_at_s": 1000 + offered + 0.2,
                "success": True,
            }
        )
    write_jsonl(tmp_path / "requests.jsonl", rows)
    metrics = []
    for tick in range(120):
        timestamp = datetime.fromtimestamp(1000 + tick / 4, UTC).isoformat()
        for backend in range(4):
            metrics.append(
                {
                    "timestamp": timestamp,
                    "source": "vllm",
                    "backend_id": f"gpu{backend}",
                    "waiting": 0,
                    "running": 2,
                    "preemptions_total": 0,
                    "kv_cache_total_blocks": 100,
                    "kv_cache_free_blocks": 50,
                    "error": None,
                }
            )
    write_jsonl(tmp_path / "backend_metrics.jsonl", metrics)
    write_json(tmp_path / "validation.json", {"passed": True})
    report = analyze_run(tmp_path)
    assert report["stable_at_four_rps"] is True
    assert report["capacity_rps"] >= 3.8
    assert report["drain_s"] < 1
