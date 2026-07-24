from __future__ import annotations

import argparse
import hashlib
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


def stable_backend(session_id: str, backend_count: int = 4) -> int:
    digest = hashlib.sha256(session_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % backend_count


def _sessions(rows: list[WorkloadItem]) -> dict[str, list[WorkloadItem]]:
    output: dict[str, list[WorkloadItem]] = defaultdict(list)
    for row in rows:
        output[row.session_id].append(row)
    for values in output.values():
        values.sort(key=lambda item: item.turn_id)
    return output


def _with_probe_identity(
    item: WorkloadItem, request_id: str, *, role: str
) -> WorkloadItem:
    return item.model_copy(
        update={
            "request_id": request_id,
            "turn_id": 0,
            "expected_output_tokens": 8,
            "measurement_phase": "four_h20_v2_1",
            "multitier_role": role,
        }
    )


def _pick_turn_zero(
    sessions: dict[str, list[WorkloadItem]],
    *,
    length: int,
    backend: int,
    count: int = 1,
    excluded: set[str] | None = None,
) -> list[WorkloadItem]:
    excluded = excluded or set()
    candidates = [
        values[0]
        for session_id, values in sorted(sessions.items())
        if session_id not in excluded
        and values[0].shared_prefix_tokens == length
        and stable_backend(session_id) == backend
    ]
    if len(candidates) < count:
        raise ValueError(
            f"need {count} turn-zero sessions for length={length}, backend={backend}"
        )
    return candidates[:count]


def _build_v2_1_profiles(swe: list[WorkloadItem]) -> dict[str, list[WorkloadItem]]:
    sessions = _sessions(swe)

    strict_lengths = (8192, 16384, 32768, 16384)
    strict: list[WorkloadItem] = []
    used: set[str] = set()
    for backend, length in enumerate(strict_lengths):
        item = _pick_turn_zero(sessions, length=length, backend=backend, excluded=used)[
            0
        ]
        used.add(item.session_id)
        strict.append(
            _with_probe_identity(
                item,
                f"v21-k01-strict-b{backend}-{length}",
                role=f"strict_backend_{backend}",
            )
        )

    lru_backend = 0
    target = _pick_turn_zero(
        sessions,
        length=16384,
        backend=lru_backend,
        excluded=used,
    )[0]
    used.add(target.session_id)
    fillers = _pick_turn_zero(
        sessions,
        length=32768,
        backend=lru_backend,
        count=4,
        excluded=used,
    )
    lru = [
        _with_probe_identity(target, "v21-k01-lru-target", role="target"),
        *[
            _with_probe_identity(
                item, f"v21-k01-lru-filler-{index}", role=f"lru_filler_{index}"
            )
            for index, item in enumerate(fillers)
        ],
    ]

    cost: list[WorkloadItem] = []
    for length in (8192, 16384, 32768):
        for backend in range(4):
            item = _pick_turn_zero(
                sessions, length=length, backend=backend, excluded=used
            )[0]
            used.add(item.session_id)
            cost.append(
                _with_probe_identity(
                    item,
                    f"v21-k02-{length}-b{backend}",
                    role=f"cost_{length}_backend_{backend}",
                )
            )

    capacity = _select(swe, {8192: 7, 16384: 7, 32768: 6}, 6)
    capacity_sessions = []
    seen_capacity: set[str] = set()
    for item in capacity:
        if item.session_id not in seen_capacity:
            seen_capacity.add(item.session_id)
            capacity_sessions.append(item.session_id)
    confirm_ids = set(capacity_sessions[:10])
    capacity_confirm = [item for item in capacity if item.session_id in confirm_ids]
    if len(capacity_confirm) != 60:
        raise ValueError("V2.1 2 RPS confirmation profile must contain 60 rows")

    return {
        "v2_1_k01_strict.jsonl": strict,
        "v2_1_k01_lru.jsonl": lru,
        "v2_1_k01_lru_target.jsonl": lru[:1],
        "v2_1_k01_lru_fillers.jsonl": lru[1:],
        "v2_1_k02_cost.jsonl": cost,
        "v2_1_k03_capacity.jsonl": capacity,
        "v2_1_k03_capacity_confirm.jsonl": capacity_confirm,
    }


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
        "kv_threshold_screening.jsonl": _select(swe, {8192: 7, 16384: 7, 32768: 6}, 3),
        "pd_crossover.jsonl": _select(swe, {8192: 3, 16384: 3, 32768: 2}, 6),
        "failure.jsonl": _select(controlled, {8192: 1, 32768: 1}, 4),
    }
    outputs.update(_build_v2_1_profiles(swe))
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
        "v2_1": {
            "k01_executed_requests": 20,
            "k01_lru_backend": 0,
            "k01_lru_working_set_tokens": 16384 + 4 * 32768,
            "k01_lru_estimated_gib": (16384 + 4 * 32768) * 98304 / 2**30,
            "k02_executed_requests": 36,
            "k03_capacity_requests": 120,
            "formal_rps_candidates": [2.0, 2.5, 3.0, 3.5, 4.0],
        },
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
