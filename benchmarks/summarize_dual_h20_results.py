from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def load_run(spec: str) -> dict[str, object]:
    label, separator, raw_path = spec.partition("=")
    if not separator or not label or not raw_path:
        raise ValueError(f"invalid run specification: {spec}")
    path = Path(raw_path)
    with (path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = next(csv.DictReader(handle))
    dual = json.loads((path / "dual_metrics.json").read_text(encoding="utf-8"))
    metrics = dual["metrics"]
    return {
        "label": label,
        "run_id": path.name,
        "requests": int(summary["requests"]),
        "successful_requests": int(summary["successful_requests"]),
        "error_rate": float(summary["error_rate"]),
        "ttft_ms_p50": float(summary["ttft_ms_p50"]),
        "ttft_ms_p90": float(summary["ttft_ms_p90"]),
        "e2e_ms_p50": float(summary["e2e_ms_p50"]),
        "e2e_ms_p90": float(summary["e2e_ms_p90"]),
        "request_per_s": float(summary["request_per_s"]),
        "output_tok_per_s": float(summary["output_tok_per_s"]),
        "token_hit_rate": float(metrics["prefix_hit_rate"]),
        "session_migration_rate": float(dual["session_migration_rate"]),
        "estimated_uncached_tokens": int(dual["estimated_uncached_tokens"]),
        "backend_request_counts": json.dumps(
            dual["backend_request_counts"], sort_keys=True
        ),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "REAL GPU - 2 x NVIDIA H20 - single fixed trace/profile - p99 excluded",
        "",
        "| Policy | Success | TTFT p50/p90 (ms) | E2E p50/p90 (ms) | req/s | token hit | migration |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {successful_requests}/{requests} | "
            "{ttft_ms_p50:.1f}/{ttft_ms_p90:.1f} | "
            "{e2e_ms_p50:.1f}/{e2e_ms_p90:.1f} | "
            "{request_per_s:.3f} | {token_hit_rate:.2%} | "
            "{session_migration_rate:.2%} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"![{title}]({path.with_suffix('.png').name})",
            "",
            "`estimated_uncached_tokens` is derived from vLLM aggregate token "
            "counters; it is not an exact per-request recomputation count.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, object]], path: Path, title: str) -> None:
    labels = [str(row["label"]) for row in rows]
    colors = ["#64748b", "#16a34a", "#dc2626", "#2563eb"][: len(rows)]
    positions = list(range(len(rows)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    width = 0.36
    axes[0, 0].bar(
        [value - width / 2 for value in positions],
        [float(row["ttft_ms_p50"]) for row in rows],
        width,
        label="p50",
        color="#60a5fa",
    )
    axes[0, 0].bar(
        [value + width / 2 for value in positions],
        [float(row["ttft_ms_p90"]) for row in rows],
        width,
        label="p90",
        color="#f97316",
    )
    axes[0, 0].set_title("TTFT (lower is better)")
    axes[0, 0].set_ylabel("milliseconds")
    axes[0, 0].legend()

    axes[0, 1].bar(
        positions, [float(row["request_per_s"]) for row in rows], color=colors
    )
    axes[0, 1].set_title("Request throughput")
    axes[0, 1].set_ylabel("requests / second")

    axes[1, 0].bar(
        positions,
        [100 * float(row["token_hit_rate"]) for row in rows],
        color=colors,
    )
    axes[1, 0].set_title("vLLM prefix-cache token hit")
    axes[1, 0].set_ylabel("percent")
    axes[1, 0].set_ylim(0, 100)

    axes[1, 1].bar(
        positions,
        [100 * float(row["session_migration_rate"]) for row in rows],
        color=colors,
    )
    axes[1, 1].set_title("Session migration (lower is better)")
    axes[1, 1].set_ylabel("percent")
    axes[1, 1].set_ylim(0, 100)

    for axis in axes.flat:
        axis.set_xticks(positions, labels, rotation=10)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{title}\nREAL GPU - 2 x H20 - single fixed trace/profile")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = [load_run(spec) for spec in args.run]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / args.name
    write_csv(rows, stem.with_suffix(".csv"))
    write_markdown(rows, stem.with_suffix(".md"), args.title)
    plot(rows, stem.with_suffix(".png"), args.title)
    print(stem)


if __name__ == "__main__":
    main()
