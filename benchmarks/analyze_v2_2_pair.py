from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.analyze_four_h20 import load_run
from benchmarks.io_utils import read_jsonl, write_json
from benchmarks.validate_v2_2_activation import validate_activation


def _relative_improvement(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (before - after) / before


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _long_prefix_p90(run_dir: Path) -> float | None:
    values = []
    for row in read_jsonl(run_dir / "joined_trace.jsonl"):
        route = row.get("route") or {}
        metadata = route.get("metadata") or {}
        client = row.get("client") or {}
        if (
            int(metadata.get("shared_prefix_tokens", 0)) >= 16_384
            and client.get("success")
            and client.get("e2e_ms") is not None
        ):
            values.append(float(client["e2e_ms"]))
    return _percentile(values, 0.90)


def analyze_pair(fixed_dir: Path, adaptive_dir: Path) -> dict[str, Any]:
    fixed = load_run("fixed-4096", fixed_dir)
    adaptive = load_run("adaptive-v2-2", adaptive_dir)
    activation = validate_activation(
        adaptive_dir / "joined_trace.jsonl",
        expected_rows=1200,
        min_overrides=60,
        min_path_changes=24,
        min_external_overrides=24,
        min_external_hit_rate=0.95,
        min_refresh_coverage=0.99,
        max_refresh_failure_rate=0.01,
        min_hbm_event_coverage=0.90,
    )
    fixed_long_p90 = _long_prefix_p90(fixed_dir)
    adaptive_long_p90 = _long_prefix_p90(adaptive_dir)
    metrics = {
        "ttft_p90_improvement": _relative_improvement(
            fixed.get("ttft_ms_p90"), adaptive.get("ttft_ms_p90")
        ),
        "e2e_p90_improvement": _relative_improvement(
            fixed.get("e2e_ms_p90"), adaptive.get("e2e_ms_p90")
        ),
        "slo_goodput_improvement": (
            None
            if not fixed.get("slo_goodput_request_per_s")
            or adaptive.get("slo_goodput_request_per_s") is None
            else (
                adaptive["slo_goodput_request_per_s"]
                - fixed["slo_goodput_request_per_s"]
            )
            / fixed["slo_goodput_request_per_s"]
        ),
        "throughput_change": (adaptive["request_per_s"] - fixed["request_per_s"])
        / fixed["request_per_s"],
        "long_prefix_e2e_p90_improvement": _relative_improvement(
            fixed_long_p90, adaptive_long_p90
        ),
    }
    failures = []
    if fixed["workload_sha256"] != adaptive["workload_sha256"]:
        failures.append("workload SHA mismatch")
    if fixed["arrival_trace_sha256"] != adaptive["arrival_trace_sha256"]:
        failures.append("arrival Trace SHA mismatch")
    if (
        min(
            fixed["successful_requests"] / fixed["requests"],
            adaptive["successful_requests"] / adaptive["requests"],
        )
        < 0.99
    ):
        failures.append("success rate below 99%")
    if not activation["passed"]:
        failures.extend(f"activation: {value}" for value in activation["failures"])
    if (
        max(
            metrics["ttft_p90_improvement"] or float("-inf"),
            metrics["e2e_p90_improvement"] or float("-inf"),
        )
        < 0.05
    ):
        failures.append("neither TTFT nor E2E p90 improved by 5%")
    if (metrics["slo_goodput_improvement"] or float("-inf")) < 0.03:
        failures.append("SLO Goodput improvement below 3%")
    if metrics["throughput_change"] < -0.01:
        failures.append("throughput regression exceeds 1%")
    if (metrics["long_prefix_e2e_p90_improvement"] or float("-inf")) < 0.10:
        failures.append("long-prefix subgroup E2E p90 improvement below 10%")
    return {
        "schema_version": "2.2",
        "fixed": fixed,
        "adaptive": adaptive,
        "activation": activation,
        "metrics": metrics,
        "failures": failures,
        "passed": not failures,
        "result_scope": "FOUR_H20_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze formal V2.2 pair")
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_pair(args.fixed, args.adaptive)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
