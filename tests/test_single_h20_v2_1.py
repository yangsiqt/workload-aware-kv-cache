from __future__ import annotations

import json
from pathlib import Path

from benchmarks.io_utils import read_jsonl
from benchmarks.single_h20_v2_1 import SCOPE, build_profile, summarize_costs


def _row(request_id: str, length: int, session: str) -> dict:
    return {
        "schema_version": "1.0",
        "dataset_name": "test",
        "dataset_revision": "test",
        "dataset_instance_id": session,
        "transform_version": "1.0",
        "request_id": request_id,
        "session_id": session,
        "turn_id": 0,
        "priority": 1,
        "request_type": "code_agent",
        "prefix_hash": request_id,
        "messages": [{"role": "user", "content": "x"}],
        "prompt_tokens": length + 16,
        "shared_prefix_tokens": length,
        "expected_output_tokens": 32,
        "source": {"dataset": "test", "license": "test", "snapshot_id": "test"},
    }


def test_build_profile_has_exact_required_matrix(tmp_path: Path) -> None:
    data = tmp_path / "data"
    profiles = data / "processed/four_h20/profiles"
    profiles.mkdir(parents=True)
    smoke = [_row(f"smoke-{length}", length, f"s-{length}") for length in (8192, 16384, 32768)]
    with (profiles / "four_h20_smoke.jsonl").open("w") as handle:
        for row in smoke:
            handle.write(json.dumps(row) + "\n")
    formal = data / "processed/four_h20/swebench.jsonl"
    formal.parent.mkdir(parents=True, exist_ok=True)
    with formal.open("w") as handle:
        for index in range(15):
            handle.write(json.dumps(_row(f"c-{index}", 16384, f"c-{index}")) + "\n")

    report = build_profile(data, tmp_path / "output")
    rows = list(read_jsonl(report["profile_path"]))
    assert report["scope"] == SCOPE
    assert len(rows) == 42
    assert len({row["request_id"] for row in rows}) == 42
    assert report["phase_counts"]["recompute"] == 9
    assert report["phase_counts"]["lmcache_l1"] == 9
    assert report["phase_counts"]["mooncake_l2"] == 9
    assert report["phase_counts"]["concurrency_16k_c12"] == 12


def test_summarize_costs_uses_three_samples_per_length() -> None:
    clients = []
    workers = []
    for path in ("recompute", "lmcache_l1", "mooncake_l2"):
        for round_id in range(3):
            for length in (8192, 16384, 32768):
                request_id = f"h20-v21-{path}-r{round_id}-{length // 1024}k"
                clients.append(
                    {
                        "request_id": request_id,
                        "input_tokens": length,
                        "ttft_ms": length / 8,
                        "e2e_ms": length / 8 + 10,
                    }
                )
                if path != "recompute":
                    workers.append(
                        {
                            "request_id": request_id,
                            "phase": "load_completed",
                            "actual_kv_path": path,
                            "retrieved_tokens": length,
                            "load_ms": length / 100,
                            "to_gpu_ms": length / 200,
                            "transfer_bytes": length * 96 * 1024,
                        }
                    )
    report = summarize_costs(clients, workers)
    assert report["recompute_fit"]["samples"] == 9
    assert report["recompute_fit"]["prefill_tokens_per_s"] == 8000.0
    assert report["lmcache_l1_overall_tokens_per_s"]["samples"] == 9
    assert report["mooncake_l2_overall_tokens_per_s"]["samples"] == 9
