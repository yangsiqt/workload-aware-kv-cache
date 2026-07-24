from pathlib import Path

from benchmarks.build_dual_h20_profiles import build
from benchmarks.analyze_dual_h20 import analyze
from benchmarks.io_utils import read_jsonl
from benchmarks.sample_backend_metrics import MOONCAKE_METRICS, parse_metrics


def test_dual_profiles_are_fixed(tmp_path: Path) -> None:
    manifest = build(Path("/root/workload-aware-kv-cache-data/processed"), tmp_path)
    assert manifest["artifacts"]["controlled-16k32k.jsonl"]["rows"] == 8
    assert manifest["artifacts"]["calibration-60.jsonl"]["rows"] == 60
    assert manifest["artifacts"]["hotspot-16k-c16.jsonl"]["rows"] == 64
    assert manifest["artifacts"]["failure-12.jsonl"]["rows"] == 12
    hotspot = list(read_jsonl(tmp_path / "hotspot-16k-c16.jsonl"))
    assert len({row["session_id"] for row in hotspot}) == 16
    assert len({row["prefix_hash"] for row in hotspot}) == 1


def test_parse_vllm_025_metrics() -> None:
    values = parse_metrics(
        """
vllm:num_requests_running 2
vllm:num_requests_waiting 3
vllm:prefix_cache_queries_total 1000
vllm:prefix_cache_hits_total 750
vllm:kv_cache_usage_perc 0.25
"""
    )
    assert values == {
        "active_decode_sequences": 0.0,
        "kv_cache_free_blocks": 0.0,
        "kv_cache_total_blocks": 0.0,
        "running": 2.0,
        "running_prefill_tokens": 0.0,
        "scheduled_decode_tokens": 0.0,
        "scheduled_prefill_tokens": 0.0,
        "skipped_waiting_prefill_tokens": 0.0,
        "waiting": 3.0,
        "waiting_prefill_tokens": 0.0,
        "prefix_queries": 1000.0,
        "prefix_hits": 750.0,
        "kv_usage": 0.25,
        "preemptions_total": 0.0,
    }


def test_parse_mooncake_v2_1_metrics() -> None:
    values = parse_metrics(
        """
mooncake_transfer_read_bytes 1024
mooncake_transfer_read_operation_count 2
mooncake_transfer_inflight_read_operations 1
mooncake_transfer_inflight_read_bytes 512
mooncake_transfer_read_failures 3
mooncake_transfer_read_misses 4
""",
        MOONCAKE_METRICS,
    )
    assert values == {
        "read_bytes_total": 1024.0,
        "read_operations_total": 2.0,
        "inflight_read_operations": 1.0,
        "inflight_read_bytes": 512.0,
        "read_failures_total": 3.0,
        "read_misses_total": 4.0,
    }


def test_parse_typed_labeled_mooncake_counters() -> None:
    values = parse_metrics(
        """
# HELP mooncake_transfer_read_bytes Total bytes read
# TYPE mooncake_transfer_read_bytes counter
mooncake_transfer_read_bytes{client_mode="real"} 1024
# HELP mooncake_transfer_read_operation_count Total read operations
# TYPE mooncake_transfer_read_operation_count counter
mooncake_transfer_read_operation_count{op_name="get_into"} 2
mooncake_transfer_read_operation_count{op_name="batch_get_into"} 3
# HELP mooncake_transfer_read_failures Total failures
# TYPE mooncake_transfer_read_failures counter
mooncake_transfer_read_failures 1
# HELP mooncake_transfer_read_misses Total misses
# TYPE mooncake_transfer_read_misses counter
mooncake_transfer_read_misses 4
""",
        MOONCAKE_METRICS,
    )
    assert values["read_bytes_total"] == 1024
    assert values["read_operations_total"] == 5
    assert values["read_failures_total"] == 1
    assert values["read_misses_total"] == 4


def test_dual_analysis_counts_migrations_and_metric_deltas(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        "\n".join(
            [
                '{"session_id":"s","turn_id":0,"backend_id":"a","success":true}',
                '{"session_id":"s","turn_id":1,"backend_id":"b","success":true}',
            ]
        )
        + "\n"
    )
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                '{"backend_id":"a","error":null,"prefix_queries":10,"prefix_hits":2,"running":0,"waiting":0,"kv_usage":0}',
                '{"backend_id":"b","error":null,"prefix_queries":20,"prefix_hits":4,"running":0,"waiting":0,"kv_usage":0}',
                '{"backend_id":"a","error":null,"prefix_queries":110,"prefix_hits":82,"running":2,"waiting":1,"kv_usage":0.5}',
                '{"backend_id":"b","error":null,"prefix_queries":220,"prefix_hits":154,"running":3,"waiting":2,"kv_usage":0.6}',
            ]
        )
        + "\n"
    )
    report = analyze(requests, metrics)
    assert report["session_migration_rate"] == 1.0
    assert report["load_skew"] == 0.0
    assert report["metrics"]["prefix_queries_delta"] == 300.0
    assert report["metrics"]["prefix_hits_delta"] == 230.0
