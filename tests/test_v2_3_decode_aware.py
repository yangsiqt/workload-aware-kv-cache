from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmarks import analyze_v2_3_pair, check_v2_3_readiness
from benchmarks.io_utils import write_jsonl
from benchmarks.validate_v2_3_decode_telemetry import validate_decode_telemetry


ROOT = Path("/root/workload-aware-kv-cache")


def _run_metrics(*, adaptive: bool) -> dict:
    multiplier = 0.8 if adaptive else 1.0
    return {
        "requests": 1200,
        "successful_requests": 1200,
        "workload_sha256": "workload",
        "arrival_trace_sha256": "trace",
        "request_per_s": 2.5,
        "slo_goodput_request_per_s": 2.0 if adaptive else 1.5,
        **{
            f"{family}_ms_p{percentile}": 1000.0 * multiplier
            for family in ("ttft", "e2e")
            for percentile in (50, 90, 95, 99)
        },
    }


def test_v2_3_pair_gate_rejects_e2e_p99_regression(monkeypatch) -> None:
    fixed = _run_metrics(adaptive=False)
    adaptive = _run_metrics(adaptive=True)
    monkeypatch.setattr(
        analyze_v2_3_pair,
        "load_run",
        lambda label, _path: adaptive if "v2-3" in label else fixed,
    )
    monkeypatch.setattr(
        analyze_v2_3_pair,
        "validate_activation",
        lambda *_args, **_kwargs: {"passed": True, "failures": []},
    )

    report = analyze_v2_3_pair.analyze_pair(Path("fixed"), Path("adaptive"))
    assert report["passed"]
    assert report["metrics"]["e2e_p99_improvement"] == 0.2

    adaptive["e2e_ms_p99"] = 1010.0
    report = analyze_v2_3_pair.analyze_pair(Path("fixed"), Path("adaptive"))
    assert not report["passed"]
    assert "E2E p99 regressed" in report["failures"]

    adaptive["e2e_ms_p99"] = 1000.0
    report = analyze_v2_3_pair.analyze_pair(Path("fixed"), Path("adaptive"))
    assert report["passed"]


def test_v2_3_replicated_gate_requires_independent_trace(monkeypatch) -> None:
    def load(label: str, path: Path) -> dict:
        result = dict(_run_metrics(adaptive="v2-3" in label))
        result["arrival_trace_sha256"] = "replicate" if "rep" in path.name else "main"
        return result

    monkeypatch.setattr(analyze_v2_3_pair, "load_run", load)
    monkeypatch.setattr(
        analyze_v2_3_pair,
        "validate_activation",
        lambda *_args, **_kwargs: {"passed": True, "failures": []},
    )
    report = analyze_v2_3_pair.analyze_replicated_pairs(
        Path("fixed-main"),
        Path("adaptive-main"),
        Path("fixed-rep"),
        Path("adaptive-rep"),
    )
    assert report["passed"]
    assert report["metrics"]["mean_e2e_p99_improvement"] == 0.2


def test_v2_3_replicated_gate_requires_five_percent_mean(monkeypatch) -> None:
    def load(label: str, path: Path) -> dict:
        adaptive = "v2-3" in label
        result = dict(_run_metrics(adaptive=adaptive))
        result["arrival_trace_sha256"] = "replicate" if "rep" in path.name else "main"
        if adaptive:
            result["e2e_ms_p99"] = 980.0 if "rep" not in path.name else 930.0
        return result

    monkeypatch.setattr(analyze_v2_3_pair, "load_run", load)
    monkeypatch.setattr(
        analyze_v2_3_pair,
        "validate_activation",
        lambda *_args, **_kwargs: {"passed": True, "failures": []},
    )
    report = analyze_v2_3_pair.analyze_replicated_pairs(
        Path("fixed-main"),
        Path("adaptive-main"),
        Path("fixed-rep"),
        Path("adaptive-rep"),
    )
    assert not report["passed"]
    assert report["metrics"]["mean_e2e_p99_improvement"] == pytest.approx(0.045)
    assert (
        "mean E2E p99 improvement across traces is below 5%"
        in report["failures"]
    )


def test_v2_3_readiness_replaces_v2_2_runtime_overlay(monkeypatch) -> None:
    called = {}

    def base_readiness(
        require_gpu: bool,
        expected_gpu_count: int,
        require_runtime_overlay: bool,
    ) -> dict:
        called.update(
            require_gpu=require_gpu,
            expected_gpu_count=expected_gpu_count,
            require_runtime_overlay=require_runtime_overlay,
        )
        return {"checks": []}

    monkeypatch.setattr(check_v2_3_readiness, "check_v2_2", base_readiness)
    report = check_v2_3_readiness.check_readiness()
    assert report["passed"]
    assert called == {
        "require_gpu": False,
        "expected_gpu_count": 4,
        "require_runtime_overlay": False,
    }


def test_v2_3_decode_telemetry_gate(tmp_path: Path) -> None:
    rows = []
    for index in range(2):
        events = [
            {"phase": "scheduler_enqueued", "recorded_at": 1.0},
            {"phase": "scheduler_seen", "recorded_at": 1.1},
            {"phase": "request_finished", "recorded_at": 2.0, "terminal": True},
        ]
        rows.append(
            {
                "request_id": f"r-{index}",
                "attempts": [
                    {
                        "decision": {
                            "candidates": [
                                {
                                    "backend_url": f"http://backend-{backend}",
                                    "decode_backlog_tokens": 8,
                                    "reserved_decode_tokens": 4,
                                    "remaining_decode_tokens": 4,
                                    "decode_cost_source": "backend_ewma",
                                    "decode_throughput_samples": 3,
                                }
                                for backend in range(4)
                            ],
                            "v2_context": {
                                "version": "2.3",
                                "fixed": {
                                    "counterfactual_total_ms": 10.0,
                                    "counterfactual_external_kv_ms": 2.0,
                                    "counterfactual_cache_confidence": 1.0,
                                },
                            },
                        },
                        "worker_events": events,
                    }
                ],
            }
        )
    path = tmp_path / "joined.jsonl"
    write_jsonl(path, rows)
    report = validate_decode_telemetry(path, expected_rows=2)
    assert report["passed"]
    assert report["backend_decode_ewma_candidates"] == 8

    rows[0]["attempts"][0]["worker_events"][0]["recorded_at"] = 1.2
    write_jsonl(path, rows)
    report = validate_decode_telemetry(path, expected_rows=2)
    assert not report["passed"]
    assert "scheduler_enqueued/scheduler_seen lifecycle order is invalid" in report["failures"]


def test_v2_3_single_h20_smoke_configuration() -> None:
    router = yaml.safe_load(
        (ROOT / "configs/v2_3_single_h20/agent-slo-adaptive.yaml").read_text()
    )
    lmcache = yaml.safe_load(
        (ROOT / "configs/v2_3_single_h20/lmcache.yaml").read_text()
    )

    assert router["kv_policy"] == "adaptive_v2_3"
    assert router["v2_decode_tokens_per_s"] == 60.0
    assert router["cache_instance_backend_map"] == {
        "v2-3-single-h20": "http://127.0.0.1:8000"
    }
    assert lmcache["max_local_cpu_size"] == 8.0
    assert lmcache["extra_config"]["global_segment_size"] == 25_769_803_776
