from __future__ import annotations

import argparse
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from benchmarks.schemas import ArrivalTraceItem


def _length_bucket(shared_prefix_tokens: int) -> str:
    if shared_prefix_tokens <= 8_192:
        return "8k"
    if shared_prefix_tokens <= 16_384:
        return "16k"
    return "32k"


def select_calibration_sessions(rows: list[dict[str, Any]]) -> set[str]:
    first_turn: dict[str, dict[str, Any]] = {}
    for row in rows:
        session_id = str(row["session_id"])
        if session_id not in first_turn or int(row["turn_id"]) < int(
            first_turn[session_id]["turn_id"]
        ):
            first_turn[session_id] = row
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for session_id, row in first_turn.items():
        by_bucket[_length_bucket(int(row["shared_prefix_tokens"]))].append(session_id)
    selected: set[str] = set()
    for bucket in ("8k", "16k", "32k"):
        sessions = sorted(by_bucket[bucket])
        if len(sessions) < 20:
            raise ValueError(f"length bucket {bucket} has fewer than 20 sessions")
        selected.update(sessions[:20])
    return selected


def _piecewise_offsets(
    count: int,
    rng: random.Random,
    *,
    high_rps: float,
    low_rps: float,
    period_s: float,
    mean_rps: float,
) -> list[float]:
    if min(high_rps, low_rps, period_s, mean_rps) <= 0:
        raise ValueError("rates and period must be positive")
    offsets: list[float] = []
    now = 0.0
    for _ in range(count):
        hazard = rng.expovariate(1.0)
        while True:
            segment = int(now // period_s)
            segment_end = (segment + 1) * period_s
            rate = high_rps if segment % 2 == 0 else low_rps
            available_hazard = (segment_end - now) * rate
            if hazard <= available_hazard:
                now += hazard / rate
                break
            hazard -= available_hazard
            now = segment_end
        offsets.append(now)
    if offsets:
        target_duration = count / mean_rps
        scale = target_duration / offsets[-1]
        offsets = [value * scale for value in offsets]
    return offsets


def generate_trace(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    high_rps: float = 3.5,
    low_rps: float = 1.5,
    period_s: float = 12.0,
    mean_rps: float = 2.5,
    cohort_size: int = 60,
) -> list[ArrivalTraceItem]:
    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive")
    rng = random.Random(seed)
    sessions: dict[str, dict[int, str]] = defaultdict(dict)
    seen: set[str] = set()
    for row in rows:
        request_id = str(row["request_id"])
        if request_id in seen:
            raise ValueError(f"duplicate request_id: {request_id}")
        seen.add(request_id)
        sessions[str(row["session_id"])][int(row["turn_id"])] = request_id

    ordered_ids: list[str] = []
    session_ids = sorted(sessions)
    rng.shuffle(session_ids)
    for start in range(0, len(session_ids), cohort_size):
        cohort = session_ids[start : start + cohort_size]
        turn_ids = sorted({turn for session_id in cohort for turn in sessions[session_id]})
        for turn_id in turn_ids:
            wave = [
                session_id
                for session_id in cohort
                if turn_id in sessions[session_id]
            ]
            rng.shuffle(wave)
            ordered_ids.extend(sessions[session_id][turn_id] for session_id in wave)

    offsets = _piecewise_offsets(
        len(ordered_ids),
        rng,
        high_rps=high_rps,
        low_rps=low_rps,
        period_s=period_s,
        mean_rps=mean_rps,
    )
    return [
        ArrivalTraceItem(request_id=request_id, offset_s=offset)
        for request_id, offset in zip(ordered_ids, offsets, strict=True)
    ]


def generate(workload: Path, output_dir: Path) -> dict[str, Path]:
    rows = [
        row
        for row in read_jsonl(workload)
        if row.get("dataset_name") == "SWE-bench Verified"
    ]
    calibration_sessions = select_calibration_sessions(rows)
    profiles = {
        "calibration": (
            [
                row
                for row in rows
                if str(row["session_id"]) in calibration_sessions
                and int(row["turn_id"]) < 3
            ],
            52,
        ),
        "formal": (rows, 53),
        "replicate": (rows, 54),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_workload = output_dir / "v2-2-calibration-workload.jsonl"
    write_jsonl(calibration_workload, profiles["calibration"][0])
    paths: dict[str, Path] = {}
    trace_records = []
    for name, (profile, seed) in profiles.items():
        path = output_dir / f"v2-2-{name}-cohort-bursty-2.5rps.jsonl"
        write_jsonl(path, generate_trace(profile, seed=seed))
        paths[name] = path
        trace_records.append(
            {
                "name": name,
                "seed": seed,
                "path": str(path.resolve()),
                "rows": len(profile),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "2.2",
        "created_at": datetime.now(UTC).isoformat(),
        "workload_path": str(workload.resolve()),
        "workload_sha256": sha256_file(workload),
        "ordering": (
            "60-session active cohorts; turns ordered within each cohort; "
            "sessions shuffled independently per turn"
        ),
        "cohort_size": 60,
        "rate_pattern": {
            "high_rps": 3.5,
            "low_rps": 1.5,
            "segment_seconds": 12.0,
            "rescaled_mean_rps": 2.5,
        },
        "calibration_workload": {
            "path": str(calibration_workload.resolve()),
            "rows": len(profiles["calibration"][0]),
            "sha256": sha256_file(calibration_workload),
        },
        "traces": trace_records,
    }
    write_json(output_dir / "manifest.json", manifest)
    paths["manifest"] = output_dir / "manifest.json"
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V2.2 wave/bursty traces")
    parser.add_argument("workload", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = generate(args.workload, args.output_dir)
    print(paths["manifest"])


if __name__ == "__main__":
    main()
