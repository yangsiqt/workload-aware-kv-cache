import json
from pathlib import Path

import pytest

from benchmarks.profile_utils import profile_instance_ids


def test_profile_instance_ids_loads_frozen_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "sessions.json"
    manifest.write_text(json.dumps({"instance_ids": ["a", "b"]}), encoding="utf-8")
    assert profile_instance_ids(
        {"swebench_instance_manifest": "sessions.json"}, tmp_path / "workloads.yaml"
    ) == ["a", "b"]


def test_profile_instance_ids_rejects_conflicting_or_duplicate_sources(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "sessions.json"
    manifest.write_text(json.dumps({"instance_ids": ["a", "a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        profile_instance_ids(
            {"swebench_instance_manifest": "sessions.json"},
            tmp_path / "workloads.yaml",
        )
    with pytest.raises(ValueError, match="either"):
        profile_instance_ids(
            {"swebench_instance_ids": ["a"], "swebench_instance_manifest": "sessions.json"},
            tmp_path / "workloads.yaml",
        )
