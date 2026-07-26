from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks import analyze_v2_3_pair


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


def test_v2_3_pair_gate_requires_e2e_p99_improvement(monkeypatch) -> None:
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

    adaptive["e2e_ms_p99"] = 1100.0
    report = analyze_v2_3_pair.analyze_pair(Path("fixed"), Path("adaptive"))
    assert not report["passed"]
    assert "E2E p99 improvement below 5%" in report["failures"]


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
