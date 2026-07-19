from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from benchmarks.schemas import WorkloadItem


def _rows(path: Path) -> list[WorkloadItem]:
    return [WorkloadItem.model_validate(row) for row in read_jsonl(path)]


def _select_sessions(
    rows: list[WorkloadItem], counts: dict[int, int], turns: int
) -> list[WorkloadItem]:
    by_session: dict[str, list[WorkloadItem]] = defaultdict(list)
    for row in rows:
        by_session[row.session_id].append(row)
    selected_ids: list[str] = []
    for tier, count in counts.items():
        candidates = sorted(
            session_id
            for session_id, values in by_session.items()
            if values[0].shared_prefix_tokens == tier
        )
        if len(candidates) < count:
            raise ValueError(f"need {count} sessions for {tier}, found {len(candidates)}")
        selected_ids.extend(candidates[:count])
    selected = [
        row
        for session_id in selected_ids
        for row in sorted(by_session[session_id], key=lambda value: value.turn_id)
        if row.turn_id < turns
    ]
    expected = sum(counts.values()) * turns
    if len(selected) != expected:
        raise ValueError(f"expected {expected} requests, found {len(selected)}")
    return selected


def _controlled(rows: list[WorkloadItem]) -> list[WorkloadItem]:
    wanted = {"controlled-0001", "controlled-0002"}
    selected = [
        row for row in rows if row.session_id in wanted and row.turn_id < 4
    ]
    selected.sort(key=lambda row: (row.session_id, row.turn_id))
    if len(selected) != 8:
        raise ValueError(f"expected 8 controlled requests, found {len(selected)}")
    return selected


def _hotspot(rows: list[WorkloadItem]) -> list[WorkloadItem]:
    base = sorted(
        (
            row
            for row in rows
            if row.session_id == "controlled-0001" and row.turn_id < 4
        ),
        key=lambda row: row.turn_id,
    )
    if len(base) != 4:
        raise ValueError("controlled-0001 must contain four turns")
    output: list[WorkloadItem] = []
    for index in range(16):
        session_id = f"dual-hotspot-{index:02d}"
        for row in base:
            output.append(
                row.model_copy(
                    update={
                        "dataset_name": "CONTROLLED-HOTSPOT",
                        "dataset_instance_id": session_id,
                        "request_id": f"{session_id}-t{row.turn_id:02d}",
                        "session_id": session_id,
                    }
                )
            )
    return output


def build(source_root: Path, output_dir: Path) -> dict[str, object]:
    controlled_path = source_root / "pre_rental" / "controlled.jsonl"
    swe_path = source_root / "final" / "swebench.jsonl"
    controlled = _rows(controlled_path)
    swe = _rows(swe_path)
    outputs = {
        "controlled-16k32k.jsonl": _controlled(controlled),
        "calibration-60.jsonl": _select_sessions(
            swe, {8192: 7, 16384: 7, 32768: 6}, 3
        ),
        "hotspot-16k-c16.jsonl": _hotspot(controlled),
        "failure-12.jsonl": _select_sessions(
            controlled, {8192: 1, 16384: 1, 32768: 1}, 4
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, rows in outputs.items():
        path = output_dir / name
        count = write_jsonl(path, rows)
        artifacts[name] = {
            "path": str(path),
            "rows": count,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": "1.0",
        "profile": "dual-h20-router-eval",
        "sources": {
            str(controlled_path): sha256_file(controlled_path),
            str(swe_path): sha256_file(swe_path),
        },
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/root/workload-aware-kv-cache-data/processed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/workload-aware-kv-cache-data/processed/dual_h20"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
