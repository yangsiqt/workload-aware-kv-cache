from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def compare(run_dirs: list[Path], output_dir: Path, simulated: bool = True) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for run_dir in run_dirs:
        row = pd.read_csv(run_dir / "summary.csv").iloc[0]
        row["run"] = run_dir.name
        rows.append(row)
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "comparison.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#2563eb", "#16a34a", "#dc2626", "#ea580c"][: len(frame)]
    frame.plot.bar(x="run", y=["ttft_ms_p50", "ttft_ms_p99"], ax=axes[0], color=colors[:2])
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("TTFT")
    frame.plot.bar(x="run", y="request_per_s", ax=axes[1], color=colors, legend=False)
    axes[1].set_ylabel("Requests/s")
    axes[1].set_title("Throughput")
    frame.plot.bar(x="run", y="cache_hit_rate", ax=axes[2], color=colors, legend=False)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Hit rate")
    axes[2].set_title("Backend prefix cache")
    for axis in axes:
        axis.set_xlabel("")
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=0.2)
    if simulated:
        figure.text(0.5, 0.5, "SIMULATED", ha="center", va="center", fontsize=42,
                    color="black", alpha=0.11, rotation=25, weight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(figure)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--real", action="store_true", help="Do not add the SIMULATED watermark")
    args = parser.parse_args()
    compare(args.run_dirs, args.output_dir, simulated=not args.real)


if __name__ == "__main__":
    main()
