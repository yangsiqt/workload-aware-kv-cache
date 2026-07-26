from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from benchmarks.check_v2_2_readiness import check_readiness as check_v2_2
from benchmarks.io_utils import write_json

ROOT = Path("/root/workload-aware-kv-cache")


def _overlay_valid(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    rows = path.read_text(encoding="utf-8").splitlines()
    if len(rows) != expected_rows:
        return False
    for row in rows:
        expected_sha, separator, raw_path = row.partition("  ")
        candidate = Path(raw_path)
        if not separator or not candidate.exists():
            return False
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_sha:
            return False
    return True


def check_readiness(
    require_gpu: bool = False, expected_gpu_count: int = 4
) -> dict[str, Any]:
    base = check_v2_2(
        require_gpu,
        expected_gpu_count,
        require_runtime_overlay=False,
    )
    checks = list(base["checks"])

    adaptive_path = ROOT / "configs/four_h20/agent-slo-kv-adaptive-v2-3.yaml"
    fixed_path = ROOT / "configs/four_h20/agent-slo-kv-fixed-4096-v2-3.yaml"
    adaptive = yaml.safe_load(adaptive_path.read_text(encoding="utf-8"))
    fixed = yaml.safe_load(fixed_path.read_text(encoding="utf-8"))
    decode_keys = (
        "v2_decode_tokens_per_s",
        "v2_decode_throughput_ewma_alpha",
        "v2_decode_throughput_min_samples",
        "v2_decode_throughput_stale_after_s",
    )
    checks.append(
        {
            "name": "v2_3_decode_config",
            "passed": adaptive.get("kv_policy") == "adaptive_v2_3"
            and fixed.get("kv_policy") == "fixed"
            and all(adaptive.get(key) == fixed.get(key) for key in decode_keys)
            and adaptive.get("v2_decode_tokens_per_s") == 60.0,
            "detail": {key: adaptive.get(key) for key in decode_keys},
        }
    )
    source_checks = {
        ROOT.parent / "vllm/vllm/v1/core/sched/scheduler.py": (
            "remaining_decode_tokens"
        ),
        ROOT.parent / "production-stack/src/vllm_router/routers/routing_logic.py": (
            "fixed_counterfactual_candidates"
        ),
        ROOT.parent / "production-stack/src/vllm_router/routers/agent_slo.py": (
            "reserved_decode_tokens"
        ),
    }
    checks.append(
        {
            "name": "v2_3_source_interfaces",
            "passed": all(
                path.exists() and marker in path.read_text(encoding="utf-8")
                for path, marker in source_checks.items()
            ),
            "detail": [str(path) for path in source_checks],
        }
    )
    workflow = [
        ROOT / "scripts/run_four_h20_kv_v2_3.sh",
        ROOT / "scripts/install_v2_3_python_overlay.sh",
        ROOT / "benchmarks/analyze_v2_3_pair.py",
    ]
    checks.append(
        {
            "name": "v2_3_workflow_files",
            "passed": all(path.exists() for path in workflow),
            "detail": [str(path) for path in workflow],
        }
    )
    if require_gpu:
        manifest = Path(
            "/root/wheels/workload-aware-kv-cache/v2-3/python-overlay/"
            "python-overlay-v2-3.sha256"
        )
        checks.append(
            {
                "name": "v2_3_runtime_overlay",
                "passed": _overlay_valid(manifest, 12),
                "detail": str(manifest),
            }
        )
    return {
        "schema_version": "2.3",
        "scope": "FOUR_H20_READY" if require_gpu else "NO_GPU_DEVELOPMENT_READY",
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
