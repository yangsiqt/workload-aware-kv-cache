from __future__ import annotations

import argparse
import random
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from benchmarks.schemas import ArrivalTraceItem


def generate(
    workload: Path, output_dir: Path, rate: float, seeds: list[int]
) -> list[Path]:
    if rate <= 0:
        raise ValueError("rate must be positive")
    items = [
        (str(row["session_id"]), int(row["turn_id"]), str(row["request_id"]))
        for row in read_jsonl(workload)
        if row.get("dataset_name") == "SWE-bench Verified"
    ]
    ordered = sorted(items)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for repeat, seed in enumerate(seeds, start=1):
        rng = random.Random(seed)
        offset = 0.0
        trace: list[ArrivalTraceItem] = []
        for _session_id, _turn_id, request_id in ordered:
            offset += rng.expovariate(rate)
            trace.append(ArrivalTraceItem(request_id=request_id, offset_s=offset))
        path = output_dir / f"swe-final-poisson-{rate:g}rps-r{repeat}.jsonl"
        write_jsonl(path, trace)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fixed Poisson arrival traces"
    )
    parser.add_argument("workload", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    paths = generate(args.workload, args.output_dir, args.request_rate, seeds)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "workload_path": str(args.workload.resolve()),
        "workload_sha256": sha256_file(args.workload),
        "request_rate": args.request_rate,
        "seeds": seeds,
        "traces": [
            {
                "path": str(path.resolve()),
                "rows": sum(1 for _ in read_jsonl(path)),
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
