from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from benchmarks.io_utils import read_jsonl, write_json


PERCENTILES = ("p50", "p90", "p95", "p99")


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return None if value in {"", "None", "nan"} else float(value)


def _load_summary(run_dir: Path) -> dict[str, str]:
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def _trace_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "joined_trace_rows": 0,
            "retry_requests": 0,
            "fallback_requests": 0,
            "prediction_e2e_absolute_error_ms_p50": None,
            "prediction_e2e_absolute_error_ms_p90": None,
        }
    errors: list[float] = []
    retries = 0
    fallbacks = 0
    rows = list(read_jsonl(path))
    for row in rows:
        attempts = row.get("attempts", [])
        retries += len(attempts) > 1
        route = row.get("route", {})
        reason = str(route.get("reason", ""))
        fallbacks += "fallback" in reason or any(
            "fallback" in str(item.get("completion", {}).get("reason", ""))
            for item in attempts
        )
        client = row.get("client", {})
        actual = client.get("e2e_ms")
        backend = route.get("backend_url")
        candidate = next(
            (
                value
                for value in route.get("candidates", [])
                if value.get("backend_url") == backend
            ),
            None,
        )
        if actual is not None and candidate and candidate.get("total_ms") is not None:
            predicted = float(candidate["total_ms"]) - float(
                candidate.get("slo_penalty_ms", 0.0)
            )
            errors.append(abs(float(actual) - predicted))

    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return {
        "joined_trace_rows": len(rows),
        "retry_requests": retries,
        "fallback_requests": fallbacks,
        "prediction_e2e_absolute_error_ms_p50": percentile(errors, 0.50),
        "prediction_e2e_absolute_error_ms_p90": percentile(errors, 0.90),
    }


def load_run(label: str, run_dir: Path) -> dict[str, Any]:
    summary = _load_summary(run_dir)
    requests = list(read_jsonl(run_dir / "requests.jsonl"))
    paths = Counter(
        str(row["selected_kv_path"])
        for row in requests
        if row.get("success") and row.get("selected_kv_path")
    )
    modes = Counter(
        str(row["selected_execution_mode"])
        for row in requests
        if row.get("success") and row.get("selected_execution_mode")
    )
    request_ids = {str(row["request_id"]) for row in requests}
    connector_rows = [
        row
        for path in sorted(run_dir.glob("connector_trace_gpu*.jsonl"))
        for row in read_jsonl(path)
        if str(row.get("request_id", "")) in request_ids
    ]
    actual_rows = [
        row
        for path in sorted(run_dir.glob("connector_actual_trace_gpu*.jsonl"))
        for row in read_jsonl(path)
        if (
            row.get("event_type") == "actual_retrieve"
            or (
                row.get("event_type") == "kv_execution_feedback"
                and row.get("phase") == "worker_retrieve"
            )
        )
        and str(row.get("request_id", "")) in request_ids
    ]
    actual_paths = Counter(
        str(row["actual_kv_path"])
        for row in actual_rows
        if row.get("actual_kv_path")
    )
    connector_fallbacks = sum(bool(row.get("fallback_reason")) for row in connector_rows)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "label": label,
        "run_id": run_dir.name,
        "requests": int(summary["requests"]),
        "successful_requests": int(summary["successful_requests"]),
        "error_rate": float(summary["error_rate"]),
        "request_per_s": float(summary["request_per_s"]),
        "output_tok_per_s": float(summary["output_tok_per_s"]),
        "slo_goodput_request_per_s": _number(summary, "slo_goodput_request_per_s"),
        "slo_violation_rate": _number(summary, "slo_violation_rate"),
        "kv_path_distribution": dict(sorted(paths.items())),
        "execution_mode_distribution": dict(sorted(modes.items())),
        "connector_actual_kv_path_distribution": dict(sorted(actual_paths.items())),
        "connector_result_rows": len(connector_rows),
        "connector_actual_retrieve_rows": len(actual_rows),
        "connector_fallback_rows": connector_fallbacks,
        "workload_sha256": manifest["workload_sha256"],
        "arrival_trace_sha256": manifest.get("arrival_trace_sha256"),
        "router_config_sha256": manifest.get("router_config_sha256"),
        "single_fixed_trace": True,
        "confidence_interval": None,
    }
    for metric in (
        "ttft_ms",
        "e2e_ms",
        "tpot_ms",
        "itl_ms",
        "router_decision_ms",
    ):
        for percentile in PERCENTILES:
            result[f"{metric}_{percentile}"] = _number(
                summary, f"{metric}_{percentile}"
            )
    result.update(_trace_metrics(run_dir / "joined_trace.jsonl"))
    return result


def analyze(specs: list[str], output_dir: Path, title: str) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"invalid run specification: {spec}")
        rows.append(load_run(label, Path(raw_path)))
    if not rows:
        raise ValueError("at least one run is required")
    if any(row["successful_requests"] / row["requests"] < 0.99 for row in rows):
        raise ValueError("all formal runs must reach at least 99% success")
    workload_hashes = {row["workload_sha256"] for row in rows}
    trace_hashes = {row["arrival_trace_sha256"] for row in rows}
    if len(rows) > 1 and len(workload_hashes) != 1:
        raise ValueError("Before/After workload SHA256 values differ")
    if len(rows) > 1 and (None in trace_hashes or len(trace_hashes) != 1):
        raise ValueError("Before/After arrival Trace SHA256 values differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "four_h20_summary.json", {"title": title, "runs": rows})
    with (output_dir / "four_h20_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        flattened = [
            {
                **row,
                "kv_path_distribution": json.dumps(row["kv_path_distribution"]),
                "execution_mode_distribution": json.dumps(
                    row["execution_mode_distribution"]
                ),
                "connector_actual_kv_path_distribution": json.dumps(
                    row["connector_actual_kv_path_distribution"]
                ),
            }
            for row in rows
        ]
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    _write_markdown(rows, output_dir / "four_h20_report.md", title)
    _plot(rows, output_dir / "four_h20_percentiles.png", title)
    return rows


def _write_markdown(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "REAL GPU result template. Each formal run uses one fixed arrival trace; "
        "p99 has about 12 tail samples at 1200 successful requests and no confidence interval.",
        "",
        "| Strategy | Success | TTFT p50/p90/p95/p99 ms | E2E p50/p90/p95/p99 ms | req/s | SLO goodput | violation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ttft = "/".join(
            "NA" if row[f"ttft_ms_{q}"] is None else f"{row[f'ttft_ms_{q}']:.1f}"
            for q in PERCENTILES
        )
        e2e = "/".join(
            "NA" if row[f"e2e_ms_{q}"] is None else f"{row[f'e2e_ms_{q}']:.1f}"
            for q in PERCENTILES
        )
        goodput = row["slo_goodput_request_per_s"]
        violation = row["slo_violation_rate"]
        lines.append(
            f"| {row['label']} | {row['successful_requests']}/{row['requests']} | "
            f"{ttft} | {e2e} | {row['request_per_s']:.3f} | "
            f"{'NA' if goodput is None else f'{goodput:.3f}'} | "
            f"{'NA' if violation is None else f'{violation:.2%}'} |"
        )
    lines.extend(
        [
            "",
            "`kv_path_distribution` is the Router-selected path. Connector-observed "
            "actual transfer data must be reported separately when available; the two are not conflated.",
            "",
            "![Four-H20 percentiles](four_h20_percentiles.png)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    labels = [str(row["label"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.18
    colors = ("#2563eb", "#16a34a", "#f59e0b", "#dc2626")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for offset, (percentile, color) in enumerate(zip(PERCENTILES, colors)):
        shift = (offset - 1.5) * width
        axes[0].bar(
            [value + shift for value in x],
            [row[f"ttft_ms_{percentile}"] or 0 for row in rows],
            width,
            label=percentile,
            color=color,
        )
        axes[1].bar(
            [value + shift for value in x],
            [row[f"e2e_ms_{percentile}"] or 0 for row in rows],
            width,
            label=percentile,
            color=color,
        )
    for axis, name in zip(axes, ("TTFT", "E2E")):
        axis.set_title(name)
        axis.set_ylabel("milliseconds")
        axis.set_xticks(x, labels, rotation=10)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    figure.suptitle(f"{title} - REAL GPU - single fixed trace")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    analyze(args.run, args.output_dir, args.title)


if __name__ == "__main__":
    main()
