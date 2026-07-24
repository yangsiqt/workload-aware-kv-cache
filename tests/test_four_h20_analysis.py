from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.analyze_four_h20 import analyze
from benchmarks.analyze_results import summarize
from benchmarks.freeze_four_h20_config import select_kv, select_pd
from benchmarks.fit_four_h20_costs import fit_kv, fit_pd
from benchmarks.io_utils import write_json, write_jsonl
from benchmarks.validate_four_h20_run import validate_run


def _request(index: int, *, path: str = "local_hbm", mode: str = "monolithic"):
    return {
        "schema_version": "1.0",
        "run_id": "run",
        "request_id": f"r-{index}",
        "session_id": f"s-{index}",
        "turn_id": 0,
        "dataset_name": "test",
        "request_type": "test",
        "prefix_hash": f"p-{index}",
        "priority": 1,
        "route_policy": "agent_slo_aware",
        "backend_id": "http://backend-0",
        "route_reason": "cost",
        "selected_kv_path": path,
        "selected_execution_mode": mode,
        "offered_at_s": float(index),
        "started_at_s": float(index),
        "completed_at_s": float(index + 1),
        "ttft_ms": float(100 + index),
        "e2e_ms": float(500 + index),
        "tpot_ms": 10.0,
        "inter_chunk_latencies_ms": [10.0, 11.0],
        "input_tokens": 8192,
        "output_tokens": 16,
        "status_code": 200,
        "success": True,
    }


def _run(tmp_path: Path, name: str, path: str = "local_hbm") -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    write_jsonl(run_dir / "requests.jsonl", (_request(i, path=path) for i in range(20)))
    summarize(run_dir / "requests.jsonl", run_dir)
    write_json(
        run_dir / "run_manifest.json",
        {
            "workload_sha256": "workload",
            "arrival_trace_sha256": "trace",
            "router_config_sha256": "config",
        },
    )
    return run_dir


def test_summary_reports_p95_slo_and_path_distribution(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "summary", "mooncake_l2")
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["ttft_ms_p95"]) > float(row["ttft_ms_p90"])
    assert float(row["slo_violation_rate"]) == 0.0
    assert json.loads(row["selected_kv_path_distribution"]) == {"mooncake_l2": 20}


def test_four_h20_report_keeps_fixed_trace_provenance(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "adaptive")
    rows = analyze([f"adaptive={run_dir}"], tmp_path / "report", "Adaptive KV")
    assert rows[0]["workload_sha256"] == "workload"
    assert rows[0]["arrival_trace_sha256"] == "trace"
    assert (tmp_path / "report" / "four_h20_report.md").exists()


def test_freeze_kv_and_pd_configs_from_measured_runs(tmp_path: Path) -> None:
    first = _run(tmp_path, "fixed-256")
    second = _run(tmp_path, "fixed-1024")
    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    config_a.write_text("kv_fixed_min_retrieve_tokens: 256\n", encoding="utf-8")
    config_b.write_text("kv_fixed_min_retrieve_tokens: 1024\n", encoding="utf-8")
    frozen = tmp_path / "frozen-kv.yaml"
    report = select_kv(
        [f"a={config_a}={first}", f"b={config_b}={second}"],
        frozen,
        tmp_path / "kv.json",
    )
    assert report["selected"]["label"] in {"a", "b"}
    assert frozen.exists()

    pd_rows = list(_request(i, mode="pd") for i in range(20))
    for row in pd_rows:
        row["e2e_ms"] = 400.0
    write_jsonl(second / "requests.jsonl", pd_rows)
    template = tmp_path / "pd.yaml"
    template.write_text(
        "pd_enabled: true\npd_policy: fixed\npd_fixed_min_prompt_tokens: 16384\n",
        encoding="utf-8",
    )
    pd_report = select_pd(
        first,
        second,
        template,
        tmp_path / "frozen-pd.yaml",
        tmp_path / "pd.json",
    )
    assert pd_report["selected_threshold"] == 8192


def test_analyze_rejects_mismatched_before_after_trace(tmp_path: Path) -> None:
    before = _run(tmp_path, "before")
    after = _run(tmp_path, "after")
    manifest = json.loads((after / "run_manifest.json").read_text())
    manifest["arrival_trace_sha256"] = "different"
    write_json(after / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="arrival Trace"):
        analyze(
            [f"before={before}", f"after={after}"],
            tmp_path / "report",
            "Adaptive KV",
        )


def test_validate_run_gates_success_trace_and_actual_path(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "validated", "lmcache_l1")
    workload = tmp_path / "workload.jsonl"
    workload_rows = [{"request_id": f"r-{index}"} for index in range(20)]
    write_jsonl(workload, workload_rows)
    write_jsonl(
        run_dir / "joined_trace.jsonl",
        ({"request_id": f"r-{index}"} for index in range(20)),
    )
    write_jsonl(
        run_dir / "connector_actual_trace_gpu0.jsonl",
        (
            {
                "event_type": "kv_execution_feedback",
                "phase": "worker_retrieve",
                "request_id": f"r-{index}",
                "actual_kv_path": "lmcache_l1",
            }
            for index in range(20)
        ),
    )
    report = validate_run(
        run_dir,
        workload,
        min_success_rate=1.0,
        required_selected_kv_paths={"lmcache_l1"},
        required_actual_kv_paths={"lmcache_l1"},
    )
    assert report["passed"] is True


def test_validate_run_gates_v2_1_worker_lifecycle(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "v2-1-validated", "lmcache_l1")
    workload = tmp_path / "workload-v2-1.jsonl"
    write_jsonl(workload, ({"request_id": f"r-{index}"} for index in range(20)))
    write_jsonl(
        run_dir / "joined_trace.jsonl",
        (
            {
                "request_id": f"r-{index}",
                "client": {"success": True},
                "attempts": [
                    {
                        "decision": {
                            "decision_id": f"r-{index}:0",
                            "kv_path": {"selected_path": "lmcache_l1"},
                        },
                        "completion": {"success": True},
                        "worker_events": [
                            {"phase": "scheduler_seen", "terminal": False},
                            {
                                "phase": "load_completed",
                                "terminal": False,
                                "actual_kv_path": "lmcache_l1",
                                "path_mismatch": False,
                            },
                            {"phase": "request_finished", "terminal": True},
                        ],
                    }
                ],
            }
            for index in range(20)
        ),
    )
    report = validate_run(
        run_dir,
        workload,
        min_success_rate=1.0,
        require_v2_1_worker_lifecycle=True,
    )
    assert report["passed"] is True
    assert report["scheduler_seen_attempts"] == 20
    assert report["worker_terminal_attempts"] == 20


def test_fit_kv_freezes_worker_measured_costs(tmp_path: Path) -> None:
    recompute = _run(tmp_path, "recompute")
    rows = []
    for index in range(20):
        row = _request(index, path="recompute")
        row["input_tokens"] = 8000 + index * 1000
        row["ttft_ms"] = 5.0 + row["input_tokens"] / 8.0
        rows.append(row)
    write_jsonl(recompute / "requests.jsonl", rows)
    l1 = _run(tmp_path, "l1", "lmcache_l1")
    l2 = _run(tmp_path, "l2", "mooncake_l2")
    for run_dir, path_name, load_ms in (
        (l1, "lmcache_l1", 10.0),
        (l2, "mooncake_l2", 20.0),
    ):
        write_jsonl(
            run_dir / "connector_actual_trace_gpu0.jsonl",
            (
                {
                    "event_type": "kv_execution_feedback",
                    "phase": "load_completed",
                    "request_id": f"r-{index}",
                    "actual_kv_path": path_name,
                    "retrieved_tokens": 1000,
                    "load_ms": load_ms,
                }
                for index in range(20)
            ),
        )
    template = tmp_path / "adaptive.yaml"
    template.write_text("kv_policy: adaptive\n", encoding="utf-8")
    output = tmp_path / "fitted.yaml"
    report = fit_kv(
        recompute,
        l1,
        l2,
        template,
        output,
        tmp_path / "fit.json",
    )
    assert report["measured"]["prefill_tokens_per_s"] == pytest.approx(8000.0)
    assert report["measured"]["kv_l1_tokens_per_s"] == pytest.approx(100000.0)
    assert report["measured"]["kv_l2_tokens_per_s"] == pytest.approx(50000.0)


def test_fit_pd_freezes_measured_transfer_cost(tmp_path: Path) -> None:
    monolithic = _run(tmp_path, "mono")
    disaggregated = _run(tmp_path, "pd")
    mono_rows = []
    pd_rows = []
    for index in range(20):
        tokens = 8000 + index * 1000
        mono = _request(index, mode="monolithic")
        mono["input_tokens"] = tokens
        mono["ttft_ms"] = 5.0 + tokens / 8.0
        mono_rows.append(mono)
        pd_row = _request(index, mode="pd")
        pd_row["input_tokens"] = tokens
        pd_row["ttft_ms"] = mono["ttft_ms"] + 2.0 + tokens / 20.0
        pd_rows.append(pd_row)
    write_jsonl(monolithic / "requests.jsonl", mono_rows)
    write_jsonl(disaggregated / "requests.jsonl", pd_rows)
    template = tmp_path / "pd-adaptive.yaml"
    template.write_text("pd_enabled: true\npd_policy: adaptive\n", encoding="utf-8")
    report = fit_pd(
        monolithic,
        disaggregated,
        template,
        tmp_path / "pd-fitted.yaml",
        tmp_path / "pd-fit.json",
    )
    assert report["measured"]["prefill_tokens_per_s"] == pytest.approx(8000.0)
    assert report["measured"]["pd_transfer_tokens_per_s"] == pytest.approx(20000.0)
