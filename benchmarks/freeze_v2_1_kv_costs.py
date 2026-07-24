from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from benchmarks.build_four_h20_profiles import stable_backend
from benchmarks.fit_four_h20_costs import _prefill_fit, _tier_rate
from benchmarks.io_utils import read_jsonl, sha256_file, write_json


PATHS = {
    "recompute": "recompute",
    "l1": "lmcache_l1",
    "l2": "mooncake_l2",
}


def _backend_index(value: str) -> int:
    port = int(value.rstrip("/").rsplit(":", 1)[-1])
    if port not in range(8000, 8004):
        raise ValueError(f"unexpected Backend URL: {value}")
    return port - 8000


def _validate_client_coverage(
    run_dir: Path,
    profile: dict[str, dict[str, Any]],
    selected_path: str,
) -> dict[str, Any]:
    requests = list(read_jsonl(run_dir / "requests.jsonl"))
    if len(requests) != 12 or not all(row.get("success") for row in requests):
        raise ValueError(f"{run_dir} must contain 12 successful requests")
    observed: set[tuple[int, int]] = set()
    for row in requests:
        request_id = str(row["request_id"])
        item = profile.get(request_id)
        if item is None:
            raise ValueError(f"unexpected request in calibration: {request_id}")
        if str(row.get("selected_kv_path")) != selected_path:
            raise ValueError(
                f"{request_id} selected {row.get('selected_kv_path')}, expected {selected_path}"
            )
        expected_backend = stable_backend(str(item["session_id"]))
        actual_backend = _backend_index(str(row.get("backend_id", "")))
        if actual_backend != expected_backend:
            raise ValueError(
                f"{request_id} routed to Backend {actual_backend}, expected {expected_backend}"
            )
        observed.add((int(item["shared_prefix_tokens"]), actual_backend))
    expected = {
        (length, backend) for length in (8192, 16384, 32768) for backend in range(4)
    }
    if observed != expected:
        raise ValueError(f"incomplete calibration coverage: {sorted(observed)}")
    return {"requests": len(requests), "coverage": sorted(observed)}


def _validate_actual_paths(run_dir: Path, expected_path: str) -> int:
    request_ids = {
        str(row["request_id"])
        for row in read_jsonl(run_dir / "requests.jsonl")
        if row.get("success")
    }
    actual: dict[str, set[str]] = {}
    for trace in sorted(run_dir.glob("connector_actual_trace_gpu*.jsonl")):
        for row in read_jsonl(trace):
            request_id = str(row.get("request_id", ""))
            if request_id not in request_ids or row.get("phase") != "load_completed":
                continue
            actual.setdefault(request_id, set()).add(str(row.get("actual_kv_path", "")))
    missing = request_ids - set(actual)
    mismatched = {
        request_id: paths
        for request_id, paths in actual.items()
        if paths != {expected_path}
    }
    if missing or mismatched:
        raise ValueError(
            f"actual path coverage failed for {expected_path}: "
            f"missing={sorted(missing)}, mismatched={mismatched}"
        )
    return len(actual)


def freeze(
    *,
    recompute_run: Path,
    l1_run: Path,
    l2_run: Path,
    profile_path: Path,
    fixed_template: Path,
    adaptive_template: Path,
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    profile = {str(row["request_id"]): row for row in read_jsonl(profile_path)}
    if len(profile) != 12:
        raise ValueError("V2.1 compact cost profile must contain 12 requests")
    coverage = {
        name: _validate_client_coverage(run_dir, profile, PATHS[name])
        for name, run_dir in (
            ("recompute", recompute_run),
            ("l1", l1_run),
            ("l2", l2_run),
        )
    }
    coverage["l1"]["actual_loads"] = _validate_actual_paths(l1_run, "lmcache_l1")
    coverage["l2"]["actual_loads"] = _validate_actual_paths(l2_run, "mooncake_l2")

    prefill_rate, prefill_intercept, prefill_samples = _prefill_fit(
        recompute_run, "recompute"
    )
    l1_rate, l1_samples = _tier_rate(l1_run, "lmcache_l1")
    l2_rate, l2_samples = _tier_rate(l2_run, "mooncake_l2")
    measured = {
        "prefill_tokens_per_s": prefill_rate,
        "prefill_intercept_ms": prefill_intercept,
        "kv_l1_tokens_per_s": l1_rate,
        "kv_l2_tokens_per_s": l2_rate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, template in (
        ("fixed-4096-frozen.yaml", fixed_template),
        ("adaptive-v2-1-frozen.yaml", adaptive_template),
    ):
        config = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
        config.update(measured)
        path = output_dir / name
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        outputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    report = {
        "schema_version": "2.1",
        "mode": "v2_1_compact_kv_cost_freeze",
        "profile": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
        "measured": measured,
        "samples": {
            "prefill": prefill_samples,
            "lmcache_l1": l1_samples,
            "mooncake_l2": l2_samples,
        },
        "coverage": coverage,
        "source_runs": [
            str(path.resolve()) for path in (recompute_run, l1_run, l2_run)
        ],
        "outputs": outputs,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-run", type=Path, required=True)
    parser.add_argument("--l1-run", type=Path, required=True)
    parser.add_argument("--l2-run", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--fixed-template", type=Path, required=True)
    parser.add_argument("--adaptive-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = freeze(
        recompute_run=args.recompute_run,
        l1_run=args.l1_run,
        l2_run=args.l2_run,
        profile_path=args.profile,
        fixed_template=args.fixed_template,
        adaptive_template=args.adaptive_template,
        output_dir=args.output_dir,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
