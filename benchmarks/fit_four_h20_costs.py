from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from benchmarks.io_utils import read_jsonl, sha256_file, write_json


def _successful_requests(run_dir: Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(run_dir / "requests.jsonl") if row.get("success")]


def _linear_fit(points: list[tuple[float, float]], name: str) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError(f"{name} requires at least two measured points")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0:
        raise ValueError(f"{name} requires more than one token length")
    slope = (
        sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in points)
        / denominator
    )
    if slope <= 0:
        raise ValueError(f"{name} measured a non-positive cost slope")
    return max(0.0, mean_y - slope * mean_x), slope


def _prefill_fit(
    run_dir: Path, required_path: str | None = None
) -> tuple[float, float, int]:
    points = [
        (float(row["input_tokens"]), float(row["ttft_ms"]))
        for row in _successful_requests(run_dir)
        if row.get("ttft_ms") is not None
        and row.get("input_tokens") is not None
        and (required_path is None or row.get("selected_kv_path") == required_path)
    ]
    intercept_ms, slope_ms_per_token = _linear_fit(points, "prefill")
    return 1000.0 / slope_ms_per_token, intercept_ms, len(points)


def _actual_rows(run_dir: Path, path_name: str) -> list[dict[str, Any]]:
    return [
        row
        for path in sorted(run_dir.glob("connector_actual_trace_gpu*.jsonl"))
        for row in read_jsonl(path)
        if (
            row.get("event_type") == "actual_retrieve"
            or (
                row.get("event_type") == "kv_execution_feedback"
                and row.get("phase") in {"worker_retrieve", "load_completed"}
            )
        )
        and row.get("actual_kv_path") == path_name
        and float(row.get("load_ms", 0)) > 0
        and int(row.get("retrieved_tokens", 0)) > 0
    ]


def _tier_rate(run_dir: Path, path_name: str) -> tuple[float, int]:
    rows = _actual_rows(run_dir, path_name)
    if not rows:
        raise ValueError(f"no worker-observed {path_name} retrievals in {run_dir}")
    rates = [
        1000.0 * int(row["retrieved_tokens"]) / float(row["load_ms"]) for row in rows
    ]
    return statistics.median(rates), len(rows)


def fit_kv(
    recompute_run: Path,
    l1_run: Path,
    l2_run: Path,
    template: Path,
    output_config: Path,
    report_path: Path,
) -> dict[str, Any]:
    prefill_rate, prefill_intercept, prefill_samples = _prefill_fit(
        recompute_run, "recompute"
    )
    l1_rate, l1_samples = _tier_rate(l1_run, "lmcache_l1")
    l2_rate, l2_samples = _tier_rate(l2_run, "mooncake_l2")
    config = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    config.update(
        {
            "prefill_tokens_per_s": prefill_rate,
            "prefill_intercept_ms": prefill_intercept,
            "kv_l1_tokens_per_s": l1_rate,
            "kv_l2_tokens_per_s": l2_rate,
        }
    )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {
        "mode": "adaptive_kv_cost_fit",
        "measured": {
            "prefill_tokens_per_s": prefill_rate,
            "prefill_intercept_ms": prefill_intercept,
            "kv_l1_tokens_per_s": l1_rate,
            "kv_l2_tokens_per_s": l2_rate,
        },
        "samples": {
            "prefill": prefill_samples,
            "lmcache_l1": l1_samples,
            "mooncake_l2": l2_samples,
        },
        "source_runs": [
            str(path.resolve()) for path in (recompute_run, l1_run, l2_run)
        ],
        "frozen_config": str(output_config.resolve()),
        "frozen_config_sha256": sha256_file(output_config),
    }
    write_json(report_path, report)
    return report


def fit_pd(
    monolithic_run: Path,
    pd_run: Path,
    template: Path,
    output_config: Path,
    report_path: Path,
) -> dict[str, Any]:
    prefill_rate, prefill_intercept, prefill_samples = _prefill_fit(monolithic_run)
    monolithic = {
        str(row["request_id"]): row for row in _successful_requests(monolithic_run)
    }
    pd_rows = {str(row["request_id"]): row for row in _successful_requests(pd_run)}
    transfer_points = []
    for request_id in sorted(set(monolithic) & set(pd_rows)):
        mono = monolithic[request_id]
        disagg = pd_rows[request_id]
        if mono.get("ttft_ms") is None or disagg.get("ttft_ms") is None:
            continue
        delta_ms = float(disagg["ttft_ms"]) - float(mono["ttft_ms"])
        if delta_ms > 0:
            transfer_points.append((float(disagg["input_tokens"]), delta_ms))
    transfer_intercept, transfer_slope = _linear_fit(transfer_points, "PD transfer")
    output_token_ms_values = [
        float(row["tpot_ms"])
        for row in [*monolithic.values(), *pd_rows.values()]
        if row.get("tpot_ms") is not None and float(row["tpot_ms"]) > 0
    ]
    if not output_token_ms_values:
        raise ValueError("PD calibration has no measured TPOT samples")
    config = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    config.update(
        {
            "prefill_tokens_per_s": prefill_rate,
            "prefill_intercept_ms": prefill_intercept,
            "pd_transfer_intercept_ms": transfer_intercept,
            "pd_transfer_tokens_per_s": 1000.0 / transfer_slope,
            "output_token_ms": statistics.median(output_token_ms_values),
        }
    )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {
        "mode": "adaptive_pd_cost_fit",
        "measured": {
            key: config[key]
            for key in (
                "prefill_tokens_per_s",
                "prefill_intercept_ms",
                "pd_transfer_intercept_ms",
                "pd_transfer_tokens_per_s",
                "output_token_ms",
            )
        },
        "samples": {
            "prefill": prefill_samples,
            "pd_transfer": len(transfer_points),
            "tpot": len(output_token_ms_values),
        },
        "source_runs": [str(monolithic_run.resolve()), str(pd_run.resolve())],
        "frozen_config": str(output_config.resolve()),
        "frozen_config_sha256": sha256_file(output_config),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    kv = subparsers.add_parser("kv")
    kv.add_argument("--recompute-run", type=Path, required=True)
    kv.add_argument("--l1-run", type=Path, required=True)
    kv.add_argument("--l2-run", type=Path, required=True)
    pd = subparsers.add_parser("pd")
    pd.add_argument("--monolithic-run", type=Path, required=True)
    pd.add_argument("--pd-run", type=Path, required=True)
    for child in (kv, pd):
        child.add_argument("--template", type=Path, required=True)
        child.add_argument("--output-config", type=Path, required=True)
        child.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "kv":
        report = fit_kv(
            args.recompute_run,
            args.l1_run,
            args.l2_run,
            args.template,
            args.output_config,
            args.report,
        )
    else:
        report = fit_pd(
            args.monolithic_run,
            args.pd_run,
            args.template,
            args.output_config,
            args.report,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
