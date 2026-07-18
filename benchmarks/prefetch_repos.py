from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.io_utils import load_yaml, read_jsonl, write_json
from benchmarks.repo_context import GitRepoCache
from benchmarks.sampling import stratified_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch immutable SWE-bench repository snapshots"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/workloads.yaml"))
    parser.add_argument(
        "--profile",
        default="pre_rental",
        help="Profile name from configs/workloads.yaml",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.profile not in config["profiles"]:
        parser.error(f"unknown profile {args.profile!r}")
    profile = config["profiles"][args.profile]
    data_root = Path(config["data_root"])
    rows = list(read_jsonl(data_root / "raw" / "swebench_verified.jsonl"))
    instance_ids = [str(value) for value in profile.get("swebench_instance_ids", [])]
    if instance_ids:
        by_id = {str(row["instance_id"]): row for row in rows}
        missing = [
            instance_id for instance_id in instance_ids if instance_id not in by_id
        ]
        if missing:
            raise ValueError(f"Unknown SWE-bench instance IDs: {missing}")
        selected = [by_id[instance_id] for instance_id in instance_ids]
    else:
        selected = stratified_sample(
            rows, int(profile["swebench_sessions"]), int(config["seed"])
        )
    snapshots = [(str(row["repo"]), str(row["base_commit"])) for row in selected]
    cache = GitRepoCache(data_root / "repo-cache")
    artifacts = cache.prefetch_archives(snapshots, args.workers)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "seed": config["seed"],
        "snapshots": artifacts,
    }
    path = data_root / "manifests" / f"repo-snapshots-{args.profile}.json"
    write_json(path, manifest)
    print(
        json.dumps(
            {
                "path": str(path),
                "snapshots": len(artifacts),
                "bytes": sum(row["bytes"] for row in artifacts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
