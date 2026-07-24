from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, sha256_file, write_json
from benchmarks.schemas import ArrivalTraceItem, WorkloadItem


ROOT = Path("/root/workload-aware-kv-cache")
DATA = Path("/root/workload-aware-kv-cache-data/processed/four_h20")
TRACES = Path("/root/workload-aware-kv-cache-data/traces/four_h20")
V2_1_WHEELS = Path("/root/wheels/workload-aware-kv-cache/v2-1")
MULTITIER_REPORT = Path(
    os.environ.get(
        "V2_1_MULTITIER_REPORT",
        "/root/workload-aware-kv-cache-data/runs/single_h20_v2_1_multitier/"
        "20260724T1628Z-h20-v21-multitier-lru-r1/report.json",
    )
)


def _wheel_from_manifest(path: Path) -> tuple[str, Path]:
    expected_sha, wheel_path = (
        path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    )
    return expected_sha, Path(wheel_path)


def _git_state(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def inspect() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    workload_path = DATA / "swebench.jsonl"
    sessions: dict[str, list[WorkloadItem]] = defaultdict(list)
    request_ids = set()
    for raw in read_jsonl(workload_path):
        item = WorkloadItem.model_validate(raw)
        sessions[item.session_id].append(item)
        request_ids.add(item.request_id)
    tiers = defaultdict(int)
    turns_valid = True
    for values in sessions.values():
        tiers[values[0].shared_prefix_tokens] += 1
        turns_valid &= sorted(item.turn_id for item in values) == list(range(6))
    check("formal_workload_rows", len(request_ids) == 1200, len(request_ids))
    check("formal_sessions", len(sessions) == 200, len(sessions))
    check(
        "formal_prefix_tiers",
        dict(tiers) == {8192: 67, 16384: 67, 32768: 66},
        dict(tiers),
    )
    check("six_serial_turns", turns_valid, "turn_id 0..5 per session")
    check(
        "prompt_limit",
        all(
            item.prompt_tokens <= 40960
            for values in sessions.values()
            for item in values
        ),
        "max_model_len=40960",
    )
    model_path = Path("/root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507")
    model_files = (
        [path for path in model_path.iterdir() if path.is_file()]
        if model_path.is_dir()
        else []
    )
    model_bytes = sum(path.stat().st_size for path in model_files)
    check(
        "model_artifacts",
        len(model_files) == 28 and model_bytes == 61_084_263_662,
        {"path": str(model_path), "files": len(model_files), "bytes": model_bytes},
    )

    profile_manifest = json.loads(
        (DATA / "profiles" / "manifest.json").read_text(encoding="utf-8")
    )
    required_profiles = {
        "four_h20_smoke.jsonl": 12,
        "kv_cost_controlled.jsonl": 24,
        "kv_threshold_screening.jsonl": 60,
        "pd_crossover.jsonl": 48,
        "failure.jsonl": 8,
        "v2_1_k01_strict.jsonl": 4,
        "v2_1_k01_lru.jsonl": 5,
        "v2_1_k01_lru_target.jsonl": 1,
        "v2_1_k01_lru_fillers.jsonl": 4,
        "v2_1_k02_cost.jsonl": 12,
        "v2_1_k03_capacity.jsonl": 120,
        "v2_1_k03_capacity_confirm.jsonl": 60,
    }
    observed_profiles = {
        name: int(value["rows"])
        for name, value in profile_manifest["artifacts"].items()
    }
    check(
        "frozen_profiles",
        all(
            observed_profiles.get(name) == rows
            for name, rows in required_profiles.items()
        ),
        observed_profiles,
    )
    profile_hashes_valid = all(
        sha256_file(Path(value["path"])) == value["sha256"]
        for value in profile_manifest["artifacts"].values()
    )
    check("profile_sha256", profile_hashes_valid, "all profile hashes match")

    trace_details = {}
    traces_valid = True
    for rps in (2, 4, 6):
        path = TRACES / f"{rps}rps" / f"swe-final-poisson-{rps}rps-r1.jsonl"
        rows = [ArrivalTraceItem.model_validate(row) for row in read_jsonl(path)]
        ids = [row.request_id for row in rows]
        offsets = [row.offset_s for row in rows]
        valid = (
            len(rows) == 1200
            and len(ids) == len(set(ids))
            and set(ids) == request_ids
            and offsets == sorted(offsets)
        )
        traces_valid &= valid
        trace_details[f"{rps}rps"] = {"sha256": sha256_file(path), "valid": valid}
    check("candidate_arrival_traces", traces_valid, trace_details)
    independent_details = {}
    independent_valid = True
    for rps in (2.0, 2.5, 3.0, 3.5, 4.0):
        label = f"{rps:g}"
        path = TRACES / f"{label}rps" / f"swe-final-poisson-{label}rps-r2.jsonl"
        rows = [ArrivalTraceItem.model_validate(row) for row in read_jsonl(path)]
        ids = [row.request_id for row in rows]
        offsets = [row.offset_s for row in rows]
        valid = (
            len(rows) == 1200
            and len(ids) == len(set(ids))
            and set(ids) == request_ids
            and offsets == sorted(offsets)
        )
        independent_valid &= valid
        independent_details[f"{label}rps-r2"] = {
            "sha256": sha256_file(path),
            "valid": valid,
        }
    check("v2_1_independent_arrival_traces", independent_valid, independent_details)
    capacity_details = {}
    capacity_valid = True
    for label, profile_name, rows_expected in (
        ("4rps", "v2_1_k03_capacity.jsonl", 120),
        ("2rps", "v2_1_k03_capacity_confirm.jsonl", 60),
    ):
        path = TRACES / "v2_1_capacity" / label / f"swe-final-poisson-{label}-r1.jsonl"
        profile_ids = {
            str(row["request_id"])
            for row in read_jsonl(DATA / "profiles" / profile_name)
        }
        rows = [ArrivalTraceItem.model_validate(row) for row in read_jsonl(path)]
        ids = [row.request_id for row in rows]
        offsets = [row.offset_s for row in rows]
        valid = (
            len(rows) == rows_expected
            and len(ids) == len(set(ids))
            and set(ids) == profile_ids
            and offsets == sorted(offsets)
        )
        capacity_valid &= valid
        capacity_details[label] = {"sha256": sha256_file(path), "valid": valid}
    check("v2_1_capacity_traces", capacity_valid, capacity_details)

    runtime = json.loads(
        subprocess.check_output(
            [
                "/root/.venvs/kv-worker/bin/python",
                "-c",
                "import json, torch, vllm; print(json.dumps({'torch':torch.__version__,'vllm':vllm.__version__}))",
            ],
            text=True,
        ).splitlines()[-1]
    )
    check(
        "torch_vllm_abi",
        runtime == {"torch": "2.11.0+cu130", "vllm": "0.25.0"},
        runtime,
    )
    lmcache_sha, lmcache_wheel = _wheel_from_manifest(
        V2_1_WHEELS / "lmcache" / "lmcache-v2-1.sha256"
    )
    check(
        "v2_1_lmcache_wheel",
        lmcache_wheel.exists() and sha256_file(lmcache_wheel) == lmcache_sha,
        {"path": str(lmcache_wheel), "sha256": lmcache_sha},
    )
    mooncake_sha, mooncake_wheel = _wheel_from_manifest(
        V2_1_WHEELS / "mooncake" / "mooncake-v2-1.sha256"
    )
    check(
        "v2_1_mooncake_cuda13_wheel",
        mooncake_wheel.exists() and sha256_file(mooncake_wheel) == mooncake_sha,
        {"path": str(mooncake_wheel), "sha256": mooncake_sha},
    )
    c_ops_path = subprocess.check_output(
        [
            "/root/.venvs/kv-worker/bin/python",
            "-c",
            "import lmcache.c_ops; print(lmcache.c_ops.__file__)",
        ],
        text=True,
    ).splitlines()[-1]
    cubins = subprocess.check_output(
        ["/usr/local/cuda-13.0/bin/cuobjdump", "--list-elf", c_ops_path], text=True
    )
    check(
        "lmcache_cuda_architectures",
        ".sm_75.cubin" in cubins and ".sm_90.cubin" in cubins,
        "wheel contains sm_75 for 2080 Ti and sm_90 for H20",
    )

    scripts = [
        ROOT / "scripts" / name
        for name in (
            "four_h20_stack.sh",
            "run_four_h20_stage.sh",
            "run_four_h20_kv_window.sh",
            "run_four_h20_pd_window.sh",
            "run_four_h20_kv_v2_1.sh",
            "run_v2_1_2080ti_smoke.sh",
            "build_v2_1_lmcache_wheel.sh",
            "build_v2_1_mooncake_wheel.sh",
            "install_v2_1_runtime.sh",
        )
    ]
    check(
        "orchestration_entrypoints",
        all(path.exists() and os.access(path, os.X_OK) for path in scripts),
        [str(path) for path in scripts],
    )
    support_modules = [
        ROOT / "benchmarks" / "fit_four_h20_costs.py",
        ROOT / "benchmarks" / "validate_four_h20_run.py",
        ROOT / "benchmarks" / "smoke_v2_1_2080ti.py",
        ROOT / "benchmarks" / "single_h20_v2_1_multitier.py",
        ROOT / "benchmarks" / "freeze_v2_1_kv_costs.py",
        ROOT / "benchmarks" / "select_v2_1_formal_rps.py",
        ROOT / "benchmarks" / "record_v2_1_four_h20.py",
        ROOT / "benchmarks" / "refresh_v2_1_router_tier.py",
        ROOT / "benchmarks" / "validate_v2_1_k01.py",
        ROOT / "benchmarks" / "filter_router_trace.py",
    ]
    check(
        "measured_cost_and_runtime_gates",
        all(path.is_file() for path in support_modules),
        [str(path) for path in support_modules],
    )
    stack_source = (ROOT / "scripts" / "four_h20_stack.sh").read_text(encoding="utf-8")
    check(
        "pd_mooncake_tcp",
        stack_source.count('"mooncake_protocol":"tcp"') == 2,
        "producer and consumer explicitly use TCP",
    )
    smoke_report_path = Path(
        "/root/log/workload-aware-kv-cache/v2-1-2080ti/smoke-report.json"
    )
    smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    check(
        "v2_1_2080ti_functional_smoke",
        smoke_report.get("passed") is True
        and smoke_report.get("scope") == "FUNCTIONAL_SMOKE_NOT_PERFORMANCE"
        and smoke_report.get("strict_l1_actual_path") == "lmcache_l1"
        and smoke_report.get("strict_l2_actual_path") == "mooncake_l2",
        {
            "path": str(smoke_report_path),
            "scope": smoke_report.get("scope"),
            "run_id": smoke_report.get("run_id"),
        },
    )
    multitier_report = json.loads(MULTITIER_REPORT.read_text(encoding="utf-8"))
    expected_paths = {
        "h20-v21-multitier-adaptive-lmcache_l1-0": "lmcache_l1",
        "h20-v21-multitier-adaptive-lmcache_l1-1": "lmcache_l1",
        "h20-v21-multitier-adaptive-mooncake_l2-0": "mooncake_l2",
    }
    snapshots = multitier_report.get("controller_snapshots", {})
    before_lru = snapshots.get("before_lru", {}).get("by_location", {})
    after_lru = snapshots.get("after_lru", {}).get("by_location", {})
    check(
        "v2_1_single_h20_multitier_lru_smoke",
        multitier_report.get("passed") is True
        and multitier_report.get("status") == "PASS"
        and multitier_report.get("scope")
        == "SINGLE_H20_MULTI_TIER_ADAPTIVE_FUNCTIONAL_SMOKE_NOT_FOUR_H20_PERFORMANCE"
        and multitier_report.get("success_rate") == 1.0
        and multitier_report.get("adaptive_selected_paths") == expected_paths
        and multitier_report.get("adaptive_actual_paths") == expected_paths
        and before_lru.get("LocalCPUBackend", 0) >= 8192
        and before_lru.get("RemoteBackend", 0) >= 8192
        and after_lru.get("LocalCPUBackend", 0) == 0
        and after_lru.get("RemoteBackend", 0) >= 8192,
        {
            "path": str(MULTITIER_REPORT),
            "scope": multitier_report.get("scope"),
            "run_id": multitier_report.get("run_id"),
            "before_lru": before_lru,
            "after_lru": after_lru,
        },
    )

    repositories = {
        "project": _git_state(ROOT),
        "production_stack": _git_state(Path("/root/production-stack")),
        "vllm": _git_state(Path("/root/vllm")),
        "lmcache": _git_state(Path("/root/LMCache")),
        "mooncake": _git_state(Path("/root/Mooncake")),
    }
    expected_branches = {
        "project": "feature/four-h20-v2-1",
        "production_stack": "feature/v2-1-router",
        "vllm": "feature/v2-1-scheduler-telemetry",
        "lmcache": "feature/v2-1-feedback-path-control",
        "mooncake": "feature/v2-1-transfer-telemetry",
    }
    check(
        "feature_branches",
        all(
            repositories[name]["branch"] == branch
            for name, branch in expected_branches.items()
        ),
        repositories,
    )
    check(
        "clean_source_trees",
        all(not value["dirty"] for value in repositories.values()),
        repositories,
    )

    ready = all(value["passed"] for value in checks)
    return {
        "schema_version": "2.1",
        "status": "READY FOR 4×H20 VALIDATION" if ready else "NOT READY",
        "scope": (
            "V2.1 source, wheels, scripts, 2080 Ti smoke and single-H20 "
            "multi-tier LRU smoke are ready; four-H20 topology and performance "
            "validation remain pending"
        ),
        "checks": checks,
        "repositories": repositories,
        "formal_workload_sha256": sha256_file(workload_path),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {report['status']}",
        "",
        report["scope"],
        "",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for item in report["checks"]:
        detail = json.dumps(item["detail"], ensure_ascii=False, sort_keys=True)
        lines.append(
            f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | `{detail[:300]}` |"
        )
    lines.extend(
        [
            "",
            "This status means the data, code, dual-architecture wheels, dry-run "
            "orchestration, 2080 Ti strict-path evidence and single-H20 Adaptive "
            "multi-tier LRU evidence are prepared. Real four-backend topology and "
            "performance validation remain mandatory on 4×H20 and cannot be "
            "claimed from this host.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/performance-results/workload-aware-kv-cache/four-h20/readiness"
        ),
    )
    args = parser.parse_args()
    report = inspect()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "pre-four-h20-readiness.json", report)
    write_markdown(report, args.output_root / "pre-four-h20-readiness.md")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "READY FOR 4×H20 VALIDATION" else 1)


if __name__ == "__main__":
    main()
