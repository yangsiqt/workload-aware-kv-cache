from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from benchmarks.io_utils import read_jsonl, sha256_file, write_json
from benchmarks.analyze_v2_2_cache_working_set import analyze as analyze_capacity

ROOT = Path("/root/workload-aware-kv-cache")
TRACE_ROOT = Path("/root/workload-aware-kv-cache-data/traces/four_h20/v2_2")
WORKLOAD = Path("/root/workload-aware-kv-cache-data/processed/four_h20/swebench.jsonl")


def check_readiness(
    require_gpu: bool = False, expected_gpu_count: int = 4
) -> dict[str, Any]:
    checks = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected = {
        "calibration": ("v2-2-calibration-cohort30-bursty-2.5rps.jsonl", 180),
        "formal": ("v2-2-formal-cohort30-bursty-2.5rps.jsonl", 1200),
        "replicate": ("v2-2-replicate-cohort30-bursty-2.5rps.jsonl", 1200),
    }
    workload_rows = list(read_jsonl(WORKLOAD))
    add("workload", len(workload_rows) == 1200, sha256_file(WORKLOAD))
    for name, (filename, count) in expected.items():
        path = TRACE_ROOT / filename
        rows = list(read_jsonl(path)) if path.exists() else []
        offsets = [float(row["offset_s"]) for row in rows]
        add(
            f"trace_{name}",
            len(rows) == count and offsets == sorted(offsets),
            {"rows": len(rows), "sha256": sha256_file(path) if path.exists() else ""},
        )

    calibration_workload = TRACE_ROOT / "v2-2-calibration-workload-cohort30.jsonl"
    capacity_inputs = {
        "calibration": (
            calibration_workload,
            TRACE_ROOT / expected["calibration"][0],
            0.60,
            0.75,
        ),
        "formal": (WORKLOAD, TRACE_ROOT / expected["formal"][0], 0.75, 0.90),
    }
    for name, (profile, trace, minimum, maximum) in capacity_inputs.items():
        report = analyze_capacity(profile, trace, capacity_gib=96.0)
        rate = report["simulated_hit_rate_all_requests"]
        add(
            f"simulated_cache_working_set_{name}",
            minimum <= rate <= maximum,
            report,
        )

    trace_manifest = TRACE_ROOT / "manifest-cohort30.json"
    manifest = (
        json.loads(trace_manifest.read_text(encoding="utf-8"))
        if trace_manifest.exists()
        else {}
    )
    add(
        "cohort30_manifest",
        manifest.get("cohort_size") == 30
        and manifest.get("workload_sha256") == sha256_file(WORKLOAD),
        str(trace_manifest),
    )

    adaptive = ROOT / "configs/four_h20/agent-slo-kv-adaptive-v2-2.yaml"
    config = yaml.safe_load(adaptive.read_text(encoding="utf-8"))
    fixed = yaml.safe_load(
        (ROOT / "configs/four_h20/agent-slo-kv-fixed-4096-v2-2.yaml").read_text(
            encoding="utf-8"
        )
    )
    add(
        "adaptive_config",
        config.get("kv_policy") == "adaptive_v2_2"
        and config.get("v2_min_gain_ms") == 500.0
        and config.get("v2_min_gain_ratio") == 0.15,
        str(adaptive),
    )
    add(
        "shared_cache_visibility_plane",
        config.get("cache_tier_route_wait_timeout_s") == 0.3
        and fixed.get("cache_tier_route_wait_timeout_s") == 0.3
        and config.get("vllm_kv_event_endpoints")
        == fixed.get("vllm_kv_event_endpoints")
        and len(config.get("vllm_kv_event_endpoints", {})) == 4,
        {
            "route_wait_s": config.get("cache_tier_route_wait_timeout_s"),
            "event_endpoints": config.get("vllm_kv_event_endpoints"),
        },
    )
    lmcache_configs = [
        yaml.safe_load(
            (ROOT / f"configs/four_h20/lmcache-backend-{index}.yaml").read_text(
                encoding="utf-8"
            )
        )
        for index in range(4)
    ]
    add(
        "cohort30_cache_capacity",
        all(config.get("max_local_cpu_size") == 8.0 for config in lmcache_configs)
        and all(
            config.get("extra_config", {}).get("global_segment_size")
            == 25_769_803_776
            for config in lmcache_configs
        ),
        {"l1_gib_per_backend": 8, "l2_gib_per_backend": 24},
    )
    required_files = [
        ROOT / "benchmarks/generate_v2_2_arrival_traces.py",
        ROOT / "benchmarks/validate_v2_2_activation.py",
        ROOT / "benchmarks/analyze_v2_2_pair.py",
        ROOT / "benchmarks/select_v2_2_thresholds.py",
        ROOT / "benchmarks/analyze_v2_2_cache_working_set.py",
        ROOT / "benchmarks/record_v2_2_stage.py",
        ROOT / "scripts/run_four_h20_kv_v2_2.sh",
        ROOT / "scripts/install_v2_2_python_overlay.sh",
    ]
    if expected_gpu_count == 1:
        single_config_dir = ROOT / "configs/v2_2_single_h20"
        single_adaptive = yaml.safe_load(
            (single_config_dir / "agent-slo-adaptive.yaml").read_text(
                encoding="utf-8"
            )
        )
        add(
            "single_h20_config",
            single_adaptive.get("kv_policy") == "adaptive_v2_2"
            and single_adaptive.get("cache_tier_route_wait_timeout_s") == 0.2
            and single_adaptive.get("vllm_kv_event_endpoints")
            == {"http://127.0.0.1:8000": "tcp://127.0.0.1:9400"},
            str(single_config_dir),
        )
        required_files.extend(
            [
                ROOT / "benchmarks/single_h20_v2_2.py",
                ROOT / "scripts/run_v2_2_single_h20.sh",
                ROOT / "scripts/v2_2_single_h20_stack.sh",
            ]
        )
    add("workflow_files", all(path.exists() for path in required_files), None)
    overlay_manifest = Path(
        "/root/wheels/workload-aware-kv-cache/v2-2/python-overlay/"
        "python-overlay-v2-2.sha256"
    )
    overlay_valid = overlay_manifest.exists()
    overlay_rows = overlay_manifest.read_text().splitlines() if overlay_valid else []
    for row in overlay_rows:
        expected_sha, separator, raw_path = row.partition("  ")
        path = Path(raw_path)
        if not separator or not path.exists():
            overlay_valid = False
            break
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            overlay_valid = False
            break
    add(
        "v2_2_runtime_overlay",
        overlay_valid and len(overlay_rows) == 10,
        str(overlay_manifest),
    )

    if require_gpu:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ]
        try:
            gpu_rows = subprocess.check_output(command, text=True).splitlines()
        except (OSError, subprocess.CalledProcessError):
            gpu_rows = []
        add(
            "single_h20" if expected_gpu_count == 1 else "four_h20",
            len(gpu_rows) == expected_gpu_count
            and all("H20" in row for row in gpu_rows),
            gpu_rows,
        )
    scope = "NO_GPU_STATIC_READY"
    if require_gpu:
        scope = "SINGLE_H20_READY" if expected_gpu_count == 1 else "FOUR_H20_READY"
    return {
        "schema_version": "2.2",
        "scope": scope,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "performance_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--single-h20", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_readiness(
        args.require_gpu or args.single_h20,
        expected_gpu_count=1 if args.single_h20 else 4,
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
