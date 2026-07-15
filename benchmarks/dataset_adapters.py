from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi
import ijson

from benchmarks.io_utils import load_yaml, sha256_file, write_json, write_jsonl


def _dataset_revision(repo_id: str) -> str:
    return HfApi().dataset_info(repo_id).sha


def _save_hf_dataset(
    repo_id: str,
    config_name: str | None,
    split: str,
    output_path: Path,
    cache_dir: Path,
    revision: str | None = None,
) -> dict[str, Any]:
    revision = revision or _dataset_revision(repo_id)
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
    rows = _count_json_array(output_path)
    return {
        "url": url,
        "etag": etag,
        "rows": rows,
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def _count_json_array(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in ijson.items(handle, "item"))


def _modelscope_lfs_file(
    mirror: dict[str, str], cache_dir: Path, expected_sha256: str
) -> Path:
    repo = mirror["repo"]
    revision = mirror["revision"]
    relative_path = mirror["path"]
    checkout = cache_dir / "modelscope" / repo.replace("/", "--")
    env = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        env.pop(name, None)
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://www.modelscope.cn/datasets/{repo}.git", str(checkout)],
            check=True, env=env,
        )
    current = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    if current != revision:
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", revision],
            check=True, env=env,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", revision], check=True, env=env
        )
    source = checkout / relative_path
    if not source.exists() or sha256_file(source) != expected_sha256:
        subprocess.run(
            ["git", "-C", str(checkout), "lfs", "pull", "--include", relative_path],
            check=True, env=env,
        )
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(f"ModelScope mirror hash mismatch for {repo}/{relative_path}: {actual}")
    return source


def _materialize(source: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        temporary.hardlink_to(source)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(output_path)


def _save_longbench(
    repo_id: str,
    names: list[str],
    output_dir: Path,
    cache_dir: Path,
    revision: str | None = None,
    archive_source: Path | None = None,
) -> list[dict[str, Any]]:
    revision = revision or _dataset_revision(repo_id)
    archive = cache_dir / "longbench" / revision / "data.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive_source is not None:
        _materialize(archive_source, archive)
    if archive.exists() and not zipfile.is_zipfile(archive):
        archive.unlink()
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/data.zip"
        request = urllib.request.Request(url, headers={"User-Agent": "workload-aware-kv-cache/0.1"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
            expected_size = int(source.headers.get("Content-Length", "0"))
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if expected_size and temporary.stat().st_size != expected_size:
            raise IOError(
                f"LongBench archive truncated: {temporary.stat().st_size} != {expected_size}"
            )
        if not zipfile.is_zipfile(temporary):
            raise zipfile.BadZipFile(f"Invalid LongBench archive: {temporary}")
        temporary.replace(archive)
    artifacts: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        members = {Path(member).name: member for member in bundle.namelist()}
        for name in names:
            filename = f"{name}.jsonl"
            member = members.get(filename)
            if member is None:
                raise FileNotFoundError(f"{filename} not found in LongBench data.zip")
            output_path = output_dir / f"longbench_{name}.jsonl"
            with bundle.open(member) as source, output_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            with output_path.open(encoding="utf-8") as handle:
                rows = sum(1 for line in handle if line.strip())
            artifacts.append({
                "repo_id": repo_id,
                "config": name,
                "split": "test",
                "revision": revision,
                "archive_sha256": sha256_file(archive),
                "rows": rows,
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            })
    return artifacts


def download_datasets(config_path: Path, selected: set[str]) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_root = Path(config["data_root"])
    raw_dir = data_root / "raw"
    cache_dir = data_root / "hf-cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "manifests" / "datasets.json"
    if manifest_path.exists():
        artifacts = json.loads(manifest_path.read_text(encoding="utf-8")).get("artifacts", {})
    else:
        artifacts = {}

    if "swebench" in selected:
        spec = config["datasets"]["swebench"]
        artifacts["swebench"] = _save_hf_dataset(
            spec["repo_id"],
            None,
            spec["split"],
            raw_dir / "swebench_verified.jsonl",
            cache_dir,
            spec.get("revision"),
        )

    if "longbench" in selected:
        spec = config["datasets"]["longbench"]
        archive_source = None
        if spec.get("mirror"):
            archive_source = _modelscope_lfs_file(
                spec["mirror"], cache_dir, spec["archive_sha256"]
            )
        artifacts["longbench"] = _save_longbench(
            spec["repo_id"], spec["configs"], raw_dir, cache_dir,
            spec.get("revision"), archive_source,
        )
        for artifact in artifacts["longbench"]:
            artifact["mirror"] = spec.get("mirror")

    if "sharegpt" in selected:
        spec = config["datasets"]["sharegpt"]
        if spec.get("mirror"):
            source = _modelscope_lfs_file(spec["mirror"], cache_dir, spec["sha256"])
            output = raw_dir / "sharegpt.json"
            _materialize(source, output)
            rows = _count_json_array(output)
            artifacts["sharegpt"] = {
                "url": spec["url"], "mirror": spec["mirror"], "rows": rows,
                "path": str(output), "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
        else:
            artifacts["sharegpt"] = _download_file(spec["url"], raw_dir / "sharegpt.json")

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": artifacts,
    }
    write_json(manifest_path, manifest)
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
