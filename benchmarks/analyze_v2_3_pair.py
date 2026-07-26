from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.analyze_four_h20 import load_run
from benchmarks.analyze_v2_2_pair import _relative_improvement
from benchmarks.io_utils import write_json
from benchmarks.validate_v2_2_activation import validate_activation


def analyze_pair(fixed_dir: Path, adaptive_dir: Path) -> dict[str, Any]:
    fixed = load_run("fixed-4096", fixed_dir)
    adaptive = load_run("adaptive-v2-3", adaptive_dir)
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
    latency_metrics = {
        f"{family}_p{percentile}_improvement": _relative_improvement(
            fixed.get(f"{family}_ms_p{percentile}"),
            adaptive.get(f"{family}_ms_p{percentile}"),
        )
        for family in ("ttft", "e2e")
        for percentile in (50, 90, 95, 99)
    }
    metrics = {
        **latency_metrics,
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
        "throughput_change": (
            adaptive["request_per_s"] - fixed["request_per_s"]
        )
        / fixed["request_per_s"],
    }
    failures: list[str] = []
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
    for family in ("ttft", "e2e"):
        for percentile in (50, 90, 95):
            key = f"{family}_p{percentile}_improvement"
            if metrics[key] is None or metrics[key] < 0:
                failures.append(f"{family.upper()} p{percentile} regressed")
    e2e_p99_improvement = metrics["e2e_p99_improvement"]
    if e2e_p99_improvement is None or e2e_p99_improvement < 0.0:
        failures.append("E2E p99 regressed")
    if (metrics["slo_goodput_improvement"] or float("-inf")) < 0.03:
        failures.append("SLO Goodput improvement below 3%")
    if metrics["throughput_change"] < -0.01:
        failures.append("throughput regression exceeds 1%")
    return {
        "schema_version": "2.3",
        "scenario": "COHORT30_HOTSET",
        "fixed": fixed,
        "adaptive": adaptive,
        "activation": activation,
        "metrics": metrics,
        "failures": failures,
        "passed": not failures,
        "result_scope": "FOUR_H20_COHORT30_HOTSET",
        "cross_trace_comparable": False,
        "requires_independent_trace_replication": True,
    }


def analyze_replicated_pairs(
    fixed_dir: Path,
    adaptive_dir: Path,
    replicate_fixed_dir: Path,
    replicate_adaptive_dir: Path,
) -> dict[str, Any]:
    primary = analyze_pair(fixed_dir, adaptive_dir)
    replicate = analyze_pair(replicate_fixed_dir, replicate_adaptive_dir)
    failures = [f"primary: {value}" for value in primary["failures"]]
    failures.extend(f"replicate: {value}" for value in replicate["failures"])
    primary_trace = primary["fixed"]["arrival_trace_sha256"]
    replicate_trace = replicate["fixed"]["arrival_trace_sha256"]
    if primary_trace == replicate_trace:
        failures.append("independent replicate reused the primary Arrival Trace")
    if primary["fixed"]["workload_sha256"] != replicate["fixed"]["workload_sha256"]:
        failures.append("replicate workload SHA mismatch")
    p99_values = [
        report["metrics"]["e2e_p99_improvement"]
        for report in (primary, replicate)
    ]
    numeric_p99 = [float(value) for value in p99_values if value is not None]
    mean_e2e_p99 = sum(numeric_p99) / 2 if len(numeric_p99) == 2 else None
    if mean_e2e_p99 is None or mean_e2e_p99 < 0.05:
        failures.append("mean E2E p99 improvement across traces is below 5%")
    return {
        "schema_version": "2.3",
        "scenario": "COHORT30_HOTSET",
        "primary": primary,
        "replicate": replicate,
        "metrics": {"mean_e2e_p99_improvement": mean_e2e_p99},
        "failures": failures,
        "passed": not failures,
        "result_scope": "FOUR_H20_COHORT30_HOTSET_TWO_TRACES",
        "cross_trace_comparable": False,
        "requires_independent_trace_replication": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze formal V2.3 pair")
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--replicate-fixed", type=Path)
    parser.add_argument("--replicate-adaptive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.replicate_fixed is None) != (args.replicate_adaptive is None):
        parser.error("replicate-fixed and replicate-adaptive must be provided together")
    report = (
        analyze_replicated_pairs(
            args.fixed,
            args.adaptive,
            args.replicate_fixed,
            args.replicate_adaptive,
        )
        if args.replicate_fixed is not None
        and args.replicate_adaptive is not None
        else analyze_pair(args.fixed, args.adaptive)
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
