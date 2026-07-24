from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, write_json


def _requests(run_dir: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(run_dir / "requests.jsonl"))


def _backend_url(row: dict[str, Any]) -> str:
    return str(row.get("backend_id", "")).rstrip("/")


def _mooncake_read_deltas(run_dir: Path) -> dict[str, dict[str, float]]:
    by_backend: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(run_dir / "backend_metrics.jsonl"):
        if row.get("source") != "mooncake" or row.get("error"):
            continue
        by_backend.setdefault(str(row["backend_id"]), []).append(row)
    deltas = {}
    for backend, rows in by_backend.items():
        byte_values = [float(row.get("read_bytes_total", 0)) for row in rows]
        operation_values = [float(row.get("read_operations_total", 0)) for row in rows]
        deltas[backend] = {
            "bytes": max(byte_values, default=0) - min(byte_values, default=0),
            "operations": max(operation_values, default=0)
            - min(operation_values, default=0),
        }
    return deltas


def validate(
    *,
    strict_recompute: Path,
    strict_l1: Path,
    strict_l2: Path,
    cold_target: Path,
    adaptive_l1: list[Path],
    fillers: Path,
    adaptive_l2: Path,
) -> dict[str, Any]:
    runs = [
        strict_recompute,
        strict_l1,
        strict_l2,
        cold_target,
        *adaptive_l1,
        fillers,
        adaptive_l2,
    ]
    validations = {
        str(path): json.loads((path / "validation.json").read_text(encoding="utf-8"))
        for path in runs
    }
    rows = {str(path): _requests(path) for path in runs}
    total = sum(len(values) for values in rows.values())
    failures = []
    if total != 20 or any(
        not row.get("success") for values in rows.values() for row in values
    ):
        failures.append(f"K01 requires 20 successful requests, observed {total}")
    if any(not report.get("passed") for report in validations.values()):
        failures.append("one or more K01 run validation gates failed")
    strict_backends = {_backend_url(row) for row in _requests(strict_recompute)}
    expected_backends = {f"http://127.0.0.1:{port}" for port in range(8000, 8004)}
    if strict_backends != expected_backends:
        failures.append(f"strict Backend coverage mismatch: {sorted(strict_backends)}")
    target_runs = [cold_target, *adaptive_l1, fillers, adaptive_l2]
    target_backends = {
        _backend_url(row) for path in target_runs for row in _requests(path)
    }
    if target_backends != {"http://127.0.0.1:8000"}:
        failures.append(f"LRU probe escaped Backend 0: {sorted(target_backends)}")

    final_inflight: dict[str, dict[str, float]] = {}
    for path in runs:
        mooncake_rows = [
            row
            for row in read_jsonl(path / "backend_metrics.jsonl")
            if row.get("source") == "mooncake" and not row.get("error")
        ]
        latest: dict[str, dict[str, Any]] = {}
        for row in mooncake_rows:
            latest[str(row["backend_id"])] = row
        for backend, row in latest.items():
            final_inflight[f"{path.name}:{backend}"] = {
                "operations": float(row.get("inflight_read_operations", -1)),
                "bytes": float(row.get("inflight_read_bytes", -1)),
            }
    if not final_inflight or any(
        value["operations"] != 0 or value["bytes"] != 0
        for value in final_inflight.values()
    ):
        failures.append("Mooncake inflight metrics did not return to zero")
    l2_read_deltas = {
        "strict_l2": _mooncake_read_deltas(strict_l2),
        "adaptive_l2": _mooncake_read_deltas(adaptive_l2),
    }
    expected_l2_backends = {
        "strict_l2": {f"gpu{index}" for index in range(4)},
        "adaptive_l2": {"gpu0"},
    }
    for name, expected in expected_l2_backends.items():
        observed = {
            backend
            for backend, delta in l2_read_deltas[name].items()
            if delta["bytes"] > 0 and delta["operations"] > 0
        }
        if not expected.issubset(observed):
            failures.append(
                f"{name} missing positive Mooncake read evidence: "
                f"expected={sorted(expected)}, observed={sorted(observed)}"
            )
    return {
        "schema_version": "2.1",
        "scope": "FOUR_H20_FUNCTIONAL_SMOKE_NOT_PERFORMANCE",
        "requests": total,
        "strict_backend_coverage": sorted(strict_backends),
        "lru_backend_coverage": sorted(target_backends),
        "final_mooncake_inflight": final_inflight,
        "mooncake_l2_read_deltas": l2_read_deltas,
        "run_validations": validations,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-recompute", type=Path, required=True)
    parser.add_argument("--strict-l1", type=Path, required=True)
    parser.add_argument("--strict-l2", type=Path, required=True)
    parser.add_argument("--cold-target", type=Path, required=True)
    parser.add_argument("--adaptive-l1", type=Path, action="append", required=True)
    parser.add_argument("--fillers", type=Path, required=True)
    parser.add_argument("--adaptive-l2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        strict_recompute=args.strict_recompute,
        strict_l1=args.strict_l1,
        strict_l2=args.strict_l2,
        cold_target=args.cold_target,
        adaptive_l1=args.adaptive_l1,
        fillers=args.fillers,
        adaptive_l2=args.adaptive_l2,
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
