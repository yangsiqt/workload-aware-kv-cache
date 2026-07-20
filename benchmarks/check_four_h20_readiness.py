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
    check("formal_prefix_tiers", dict(tiers) == {8192: 67, 16384: 67, 32768: 66}, dict(tiers))
    check("six_serial_turns", turns_valid, "turn_id 0..5 per session")
    check(
        "prompt_limit",
        all(item.prompt_tokens <= 40960 for values in sessions.values() for item in values),
        "max_model_len=40960",
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
    }
    observed_profiles = {
        name: int(value["rows"])
        for name, value in profile_manifest["artifacts"].items()
    }
    check(
        "five_profiles",
        all(observed_profiles.get(name) == rows for name, rows in required_profiles.items()),
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
    patched = Path(
        "/root/wheels/workload-aware-kv-cache/patched/"
        "lmcache-0.5.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    expected_patched_sha = "55a0acb0b7336cf5828924ed724f9711d16723c605993c4d015d20251ff7c1b3"
    check(
        "patched_lmcache_wheel",
        patched.exists() and sha256_file(patched) == expected_patched_sha,
        expected_patched_sha,
    )
    mooncake = Path(
        "/root/wheels/workload-aware-kv-cache/"
        "mooncake_transfer_engine_cuda13-0.3.11.post1-cp312-cp312-manylinux_2_35_x86_64.whl"
    )
    check(
        "mooncake_cuda13_wheel",
        mooncake.exists()
        and sha256_file(mooncake)
        == "1f0b62ef625bf017eb4f1717d7240bf72ba22cb9c0abd93fcd28ed53bbb492b9",
        "CUDA13 wheel hash",
    )

    scripts = [
        ROOT / "scripts" / name
        for name in (
            "four_h20_stack.sh",
            "run_four_h20_stage.sh",
            "run_four_h20_kv_window.sh",
            "run_four_h20_pd_window.sh",
            "build_patched_lmcache_wheel.sh",
            "install_patched_lmcache.sh",
            "restore_official_lmcache.sh",
        )
    ]
    check(
        "orchestration_entrypoints",
        all(path.exists() and os.access(path, os.X_OK) for path in scripts),
        [str(path) for path in scripts],
    )

    repositories = {
        "project": _git_state(ROOT),
        "production_stack": _git_state(Path("/root/production-stack")),
        "vllm": _git_state(Path("/root/vllm")),
        "lmcache": _git_state(Path("/root/LMCache")),
        "mooncake": _git_state(Path("/root/Mooncake")),
    }
    expected_branches = {
        "project": "feature/four-h20-adaptive-kv-pd",
        "production_stack": "feature/adaptive-kv-pd-router",
        "lmcache": "feature/workload-aware-kv-path",
    }
    check(
        "feature_branches",
        all(repositories[name]["branch"] == branch for name, branch in expected_branches.items()),
        repositories,
    )
    check(
        "clean_source_trees",
        all(not value["dirty"] for value in repositories.values()),
        repositories,
    )

    ready = all(value["passed"] for value in checks)
    return {
        "schema_version": "1.0",
        "status": "READY FOR 4xH20" if ready else "NOT READY",
        "scope": "pre-rental artifacts; K01/P01 remain mandatory runtime gates",
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
            "This status means the data, code, wheels, dry-run orchestration and analysis "
            "are prepared. Real LMCache/Mooncake/PD runtime validation is intentionally "
            "deferred to K01/P01 on 4 x H20 and cannot be claimed from the 2080 Ti host.",
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
    raise SystemExit(0 if report["status"] == "READY FOR 4xH20" else 1)


if __name__ == "__main__":
    main()
