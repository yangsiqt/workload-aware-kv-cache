from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.analyze_four_h20 import analyze
from benchmarks.analyze_results import summarize
from benchmarks.freeze_four_h20_config import select_kv, select_pd
from benchmarks.io_utils import write_json, write_jsonl


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
