from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

from benchmarks.io_utils import load_yaml, sha256_file, write_json, write_jsonl


def _dataset_revision(repo_id: str) -> str:
    return HfApi().dataset_info(repo_id).sha


def _save_hf_dataset(
    repo_id: str,
    config_name: str | None,
    split: str,
    output_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    revision = _dataset_revision(repo_id)
    dataset = load_dataset(
        repo_id,
        config_name,
        split=split,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    count = write_jsonl(output_path, dataset)
    return {
        "repo_id": repo_id,
        "config": config_name,
        "split": split,
        "revision": revision,
        "fingerprint": dataset._fingerprint,
        "rows": count,
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def _download_file(url: str, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "workload-aware-kv-cache/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
        etag = response.headers.get("ETag")
    temporary.replace(output_path)
    with output_path.open(encoding="utf-8") as handle:
        rows = len(json.load(handle))
    return {
        "url": url,
        "etag": etag,
        "rows": rows,
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def download_datasets(config_path: Path, selected: set[str]) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_root = Path(config["data_root"])
    raw_dir = data_root / "raw"
    cache_dir = data_root / "hf-cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}

    if "swebench" in selected:
        spec = config["datasets"]["swebench"]
        artifacts["swebench"] = _save_hf_dataset(
            spec["repo_id"],
            None,
            spec["split"],
            raw_dir / "swebench_verified.jsonl",
            cache_dir,
        )

    if "longbench" in selected:
        spec = config["datasets"]["longbench"]
        artifacts["longbench"] = []
        for name in spec["configs"]:
            artifacts["longbench"].append(
                _save_hf_dataset(
                    spec["repo_id"],
                    name,
                    spec["split"],
                    raw_dir / f"longbench_{name}.jsonl",
                    cache_dir,
                )
            )

    if "sharegpt" in selected:
        spec = config["datasets"]["sharegpt"]
        artifacts["sharegpt"] = _download_file(
            spec["url"], raw_dir / "sharegpt.json"
        )

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": artifacts,
    }
    write_json(data_root / "manifests" / "datasets.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and pin public workload datasets")
    parser.add_argument("--config", default="configs/workloads.yaml")
    parser.add_argument(
        "--datasets",
        default="swebench,longbench,sharegpt",
        help="Comma-separated: swebench,longbench,sharegpt",
    )
    args = parser.parse_args()
    selected = {name.strip() for name in args.datasets.split(",") if name.strip()}
    manifest = download_datasets(Path(args.config), selected)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

