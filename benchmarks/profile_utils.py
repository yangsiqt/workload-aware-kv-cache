from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def profile_instance_ids(profile: dict[str, Any], config_path: Path) -> list[str]:
    """Load an explicitly frozen SWE-bench selection for a workload profile."""
    inline = [str(value) for value in profile.get("swebench_instance_ids", [])]
    manifest_name = profile.get("swebench_instance_manifest")
    if inline and manifest_name:
        raise ValueError("use either swebench_instance_ids or swebench_instance_manifest")
    if not manifest_name:
        return inline
    manifest_path = (config_path.parent / str(manifest_name)).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    instance_ids = [str(value) for value in payload.get("instance_ids", [])]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"duplicate instance IDs in {manifest_path}")
    return instance_ids
