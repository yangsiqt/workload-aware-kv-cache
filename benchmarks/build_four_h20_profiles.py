from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from benchmarks.schemas import WorkloadItem


def _load(path: Path) -> list[WorkloadItem]:
    return [WorkloadItem.model_validate(row) for row in read_jsonl(path)]


def _select(
    rows: list[WorkloadItem], tier_sessions: dict[int, int], turns: int
) -> list[WorkloadItem]:
    sessions: dict[str, list[WorkloadItem]] = defaultdict(list)
    for row in rows:
        sessions[row.session_id].append(row)
    selected_ids: list[str] = []
    for tier, count in tier_sessions.items():
        candidates = sorted(
            session_id
            for session_id, values in sessions.items()
            if values[0].shared_prefix_tokens == tier
        )
        if len(candidates) < count:
            raise ValueError(f"need {count} sessions for tier {tier}")
        selected_ids.extend(candidates[:count])
    output = [
        row
        for session_id in selected_ids
        for row in sorted(sessions[session_id], key=lambda item: item.turn_id)
        if row.turn_id < turns
    ]
    expected = sum(tier_sessions.values()) * turns
    if len(output) != expected:
        raise ValueError(f"expected {expected} rows, found {len(output)}")
    return output


def build(source_root: Path, output_dir: Path) -> dict[str, object]:
    controlled_path = source_root / "final" / "controlled.jsonl"
    swe_path = source_root / "four_h20" / "swebench.jsonl"
    controlled = _load(controlled_path)
    swe = _load(swe_path)
    outputs = {
        "four_h20_smoke.jsonl": _select(controlled, {8192: 1, 16384: 1, 32768: 1}, 4),
        "kv_cost_controlled.jsonl": _select(
            controlled, {8192: 2, 16384: 2, 32768: 2}, 4
        ),
        "kv_threshold_screening.jsonl": _select(
            swe, {8192: 7, 16384: 7, 32768: 6}, 3
        ),
        "pd_crossover.jsonl": _select(swe, {8192: 3, 16384: 3, 32768: 2}, 6),
        "failure.jsonl": _select(controlled, {8192: 1, 32768: 1}, 4),
    }
    pd_calibration = _select(swe, {8192: 1, 16384: 2, 32768: 1}, 6)
    outputs["pd_crossover_monolithic.jsonl"] = pd_calibration
    outputs["pd_crossover_pd.jsonl"] = pd_calibration
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {}
    for name, rows in outputs.items():
        path = output_dir / name
        artifacts[name] = {
            "path": str(path),
            "rows": write_jsonl(path, rows),
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": "1.0",
        "profile": "four-h20-preflight",
        "sources": {
            str(controlled_path): sha256_file(controlled_path),
            str(swe_path): sha256_file(swe_path),
        },
        "formal_workload": {
            "path": str(swe_path),
            "rows": len(swe),
            "sha256": sha256_file(swe_path),
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
        default=Path("/root/workload-aware-kv-cache-data/processed/four_h20/profiles"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
