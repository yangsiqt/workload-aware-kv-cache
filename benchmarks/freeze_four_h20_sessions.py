from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.io_utils import load_yaml, read_jsonl, sha256_file, write_json
from benchmarks.profile_utils import profile_instance_ids
from benchmarks.sampling import stratified_sample


def freeze(config_path: Path, output_path: Path, count: int) -> dict[str, object]:
    config = load_yaml(config_path)
    base_profile = config["profiles"]["final"]
    existing = profile_instance_ids(base_profile, config_path)
    if count < len(existing):
        raise ValueError("count must include all existing final sessions")
    raw_path = Path(config["data_root"]) / "raw" / "swebench_verified.jsonl"
    rows = list(read_jsonl(raw_path))
    existing_set = set(existing)
    additional = stratified_sample(
        [row for row in rows if str(row["instance_id"]) not in existing_set],
        count - len(existing),
        int(config["seed"]),
    )
    instance_ids = existing + [str(row["instance_id"]) for row in additional]
    if len(instance_ids) != count or len(instance_ids) != len(set(instance_ids)):
        raise ValueError("frozen selection is incomplete or contains duplicates")
    by_id = {str(row["instance_id"]): row for row in rows}
    repository_counts = Counter(str(by_id[item]["repo"]) for item in instance_ids)
    manifest = {
        "schema_version": "1.0",
        "profile": "four_h20",
        "seed": int(config["seed"]),
        "source": {
            "path": str(raw_path),
            "sha256": sha256_file(raw_path),
            "base_profile": "final",
            "reused_sessions": len(existing),
        },
        "instance_ids": instance_ids,
        "repository_counts": dict(sorted(repository_counts.items())),
    }
    write_json(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the 200-session four-H20 SWE-bench selection"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/workloads.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("configs/four_h20_sessions.json")
    )
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(freeze(args.config, args.output, args.count), indent=2))


if __name__ == "__main__":
    main()
