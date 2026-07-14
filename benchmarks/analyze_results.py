from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.io_utils import read_jsonl


LATENCY_COLUMNS = ["ttft_ms", "e2e_ms", "tpot_ms"]


def _percentile(series: pd.Series, q: float) -> float | None:
    clean = series.dropna()
    return float(clean.quantile(q)) if not clean.empty else None


def summarize(results_path: Path, output_dir: Path, simulated: bool = False) -> pd.DataFrame:
    rows = read_jsonl(results_path)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No request results in {results_path}")

    elapsed = max(float(frame["completed_at_s"].max() - frame["started_at_s"].min()), 1e-9)
    successful = frame[frame["success"] == True]  # noqa: E712
    summary: dict[str, object] = {
        "requests": len(frame),
        "successful_requests": len(successful),
        "error_rate": float(1.0 - len(successful) / len(frame)),
        "request_per_s": float(len(successful) / elapsed),
        "output_tok_per_s": float(successful["output_tokens"].sum() / elapsed),
        "cache_hit_rate": None,
        "simulated": simulated,
    }
    cache = frame["cache_hit"].dropna() if "cache_hit" in frame else pd.Series(dtype=bool)
    if not cache.empty:
        summary["cache_hit_rate"] = float(cache.astype(bool).mean())
    for column in LATENCY_COLUMNS:
        for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
            summary[f"{column}_{name}"] = _percentile(successful[column], quantile)
    itl = [value for values in successful["inter_chunk_latencies_ms"] for value in values]
    itl_series = pd.Series(itl, dtype=float)
    for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        summary[f"itl_ms_{name}"] = _percentile(itl_series, quantile)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    _plot(summary_frame.iloc[0], output_dir / "latency_summary.png", simulated)
    return summary_frame


def _plot(row: pd.Series, path: Path, simulated: bool) -> None:
    labels = ["TTFT p50", "TTFT p99", "E2E p50", "E2E p99"]
    values = [row.ttft_ms_p50, row.ttft_ms_p99, row.e2e_ms_p50, row.e2e_ms_p99]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color=["#2563eb", "#dc2626", "#16a34a", "#ea580c"])
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Serving latency summary")
    axis.grid(axis="y", alpha=0.25)
    if simulated:
        figure.text(0.5, 0.5, "SIMULATED", ha="center", va="center", fontsize=36,
                    color="black", alpha=0.13, rotation=25, weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--simulated", action="store_true")
    args = parser.parse_args()
    summarize(args.results, args.output_dir or args.results.parent, args.simulated)


if __name__ == "__main__":
    main()
