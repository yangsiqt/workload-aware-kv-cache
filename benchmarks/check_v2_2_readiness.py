from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from benchmarks.io_utils import read_jsonl, sha256_file, write_json

ROOT = Path("/root/workload-aware-kv-cache")
TRACE_ROOT = Path("/root/workload-aware-kv-cache-data/traces/four_h20/v2_2")
WORKLOAD = Path("/root/workload-aware-kv-cache-data/processed/four_h20/swebench.jsonl")


def check_readiness(require_gpu: bool = False) -> dict[str, Any]:
    checks = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected = {
        "calibration": ("v2-2-calibration-wave-bursty-2.5rps.jsonl", 180),
        "formal": ("v2-2-formal-wave-bursty-2.5rps.jsonl", 1200),
        "replicate": ("v2-2-replicate-wave-bursty-2.5rps.jsonl", 1200),
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

    adaptive = ROOT / "configs/four_h20/agent-slo-kv-adaptive-v2-2.yaml"
    config = yaml.safe_load(adaptive.read_text(encoding="utf-8"))
    add(
        "adaptive_config",
        config.get("kv_policy") == "adaptive_v2_2"
        and config.get("v2_min_gain_ms") == 500.0
        and config.get("v2_min_gain_ratio") == 0.15,
        str(adaptive),
    )
    required_files = [
        ROOT / "benchmarks/generate_v2_2_arrival_traces.py",
        ROOT / "benchmarks/validate_v2_2_activation.py",
        ROOT / "benchmarks/analyze_v2_2_pair.py",
        ROOT / "benchmarks/select_v2_2_thresholds.py",
        ROOT / "scripts/run_four_h20_kv_v2_2.sh",
        ROOT / "scripts/install_v2_2_python_overlay.sh",
    ]
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
        overlay_valid and len(overlay_rows) == 9,
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
            "four_h20",
            len(gpu_rows) == 4 and all("H20" in row for row in gpu_rows),
            gpu_rows,
        )
    return {
        "schema_version": "2.2",
        "scope": "NO_GPU_STATIC_READY" if not require_gpu else "FOUR_H20_READY",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "performance_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_readiness(args.require_gpu)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
