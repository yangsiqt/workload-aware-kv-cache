from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import requests

from benchmarks.io_utils import (
    read_jsonl,
    repository_state,
    sha256_file,
    write_json,
    write_jsonl,
)
from benchmarks.join_traces import join
from benchmarks.run_benchmark import _request
from benchmarks.schemas import RequestResult, WorkloadItem
from benchmarks.smoke_v2_1_2080ti import (
    SCHEDULER_METRICS,
    _metric_sum,
    _validate_lifecycle,
)
from benchmarks.tokenizer_utils import load_tokenizer


SCOPE = "SINGLE_H20_FUNCTIONAL_AND_COST_SCREENING_NOT_FOUR_H20_PERFORMANCE"
MODEL_NAME = "Qwen3-30B-A3B-Instruct-2507"
MOONCAKE_METRICS = {
    "mooncake_transfer_read_bytes",
    "mooncake_transfer_read_operation_count",
    "mooncake_transfer_inflight_read_operations",
    "mooncake_transfer_inflight_read_bytes",
    "mooncake_transfer_read_failures",
    "mooncake_transfer_read_misses",
}
MOONCAKE_EAGER_METRICS = {
    "mooncake_transfer_inflight_read_operations",
    "mooncake_transfer_inflight_read_bytes",
    "mooncake_transfer_read_failures",
    "mooncake_transfer_read_misses",
}


def _clone(
    row: dict[str, Any],
    *,
    request_id: str,
    phase: str,
    round_id: int,
) -> dict[str, Any]:
    cloned = dict(row)
    cloned.update(
        {
            "request_id": request_id,
            "turn_id": 0,
            "expected_output_tokens": 8,
            "measurement_phase": phase,
            "measurement_round": round_id,
            "source_request_id": row["request_id"],
        }
    )
    return cloned


def build_profile(data_root: Path, output_dir: Path) -> dict[str, Any]:
    profiles = data_root / "processed/four_h20/profiles"
    formal_path = data_root / "processed/four_h20/swebench.jsonl"
    smoke_path = profiles / "four_h20_smoke.jsonl"
    smoke = list(read_jsonl(smoke_path))
    base_by_length: dict[int, dict[str, Any]] = {}
    for row in smoke:
        length = int(row["shared_prefix_tokens"])
        if int(row["turn_id"]) == 0 and length in {8192, 16384, 32768}:
            base_by_length.setdefault(length, row)
    if set(base_by_length) != {8192, 16384, 32768}:
        raise ValueError("single-H20 cost profile requires 8K/16K/32K turn-0 rows")

    rows: list[dict[str, Any]] = []
    for phase in ("recompute", "lmcache_l1", "mooncake_l2"):
        for round_id in range(3):
            for length in (8192, 16384, 32768):
                rows.append(
                    _clone(
                        base_by_length[length],
                        request_id=f"h20-v21-{phase}-r{round_id}-{length // 1024}k",
                        phase=phase,
                        round_id=round_id,
                    )
                )

    formal_16k_sources = []
    seen_sessions: set[str] = set()
    for row in read_jsonl(formal_path):
        if (
            int(row["turn_id"]) == 0
            and int(row["shared_prefix_tokens"]) == 16384
            and str(row["session_id"]) not in seen_sessions
        ):
            formal_16k_sources.append(row)
            seen_sessions.add(str(row["session_id"]))
            if len(formal_16k_sources) == 15:
                break
    if len(formal_16k_sources) != 15:
        raise ValueError("single-H20 profile requires 15 distinct 16K sessions")

    for path, source in zip(
        ("local_hbm", "lmcache_l1", "mooncake_l2"),
        formal_16k_sources[:3],
        strict=True,
    ):
        rows.append(
            _clone(
                source,
                request_id=f"h20-v21-adaptive-{path}",
                phase=f"adaptive_{path}",
                round_id=0,
            )
        )

    concurrency_sources = formal_16k_sources[3:]
    for index, row in enumerate(concurrency_sources):
        rows.append(
            _clone(
                row,
                request_id=f"h20-v21-concurrency-{index:02d}",
                phase="concurrency_16k_c12",
                round_id=0,
            )
        )

    request_ids = [str(row["request_id"]) for row in rows]
    if len(rows) != 42 or len(request_ids) != len(set(request_ids)):
        raise ValueError("single-H20 profile must contain 42 unique requests")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "profile.jsonl"
    write_jsonl(profile_path, rows)
    manifest = {
        "schema_version": "2.1",
        "scope": SCOPE,
        "profile": "v2-1-single-h20-required-validation",
        "count": len(rows),
        "phase_counts": {
            phase: sum(row["measurement_phase"] == phase for row in rows)
            for phase in sorted({str(row["measurement_phase"]) for row in rows})
        },
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "source_sha256": {
            str(smoke_path): sha256_file(smoke_path),
            str(formal_path): sha256_file(formal_path),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _run(stack: Path, *args: str) -> None:
    subprocess.run([str(stack), *args], check=True)


def _switch_router(stack: Path, config: Path) -> None:
    _run(stack, "router", str(config))
    time.sleep(0.75)


def _metrics(url: str, names: set[str]) -> tuple[str, dict[str, float]]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    text = response.text
    return text, {name: _metric_sum(text, name) for name in names}


def _direct_warmup(item: WorkloadItem) -> None:
    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [message.model_dump() for message in item.messages],
            "max_tokens": 1,
            "temperature": 0,
        },
        timeout=600,
    )
    response.raise_for_status()


async def _issue_rows(
    rows: list[WorkloadItem],
    *,
    run_id: str,
    tokenizer: Any,
    concurrency: int = 1,
    stagger_after: int | None = None,
    observed_scheduler: dict[str, float] | None = None,
) -> list[RequestResult]:
    timeout = aiohttp.ClientTimeout(total=900)
    connector = aiohttp.TCPConnector(limit=64, keepalive_timeout=30)
    semaphore = asyncio.Semaphore(concurrency)
    origin = time.perf_counter()
    results: list[RequestResult] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def issue(item: WorkloadItem) -> None:
            async with semaphore:
                result = await _request(
                    session,
                    item,
                    run_id=run_id,
                    endpoint="http://127.0.0.1:9003/v1/chat/completions",
                    model=MODEL_NAME,
                    tokenizer=tokenizer,
                    route_policy="agent_slo_aware",
                    temperature=0.0,
                    offered_at=time.perf_counter() - origin,
                )
                results.append(result)

        if stagger_after is None:
            for row in rows:
                await issue(row)
        else:
            tasks = [asyncio.create_task(issue(row)) for row in rows[:stagger_after]]
            await asyncio.sleep(0.75)
            tasks.extend(
                asyncio.create_task(issue(row)) for row in rows[stagger_after:]
            )
            while not all(task.done() for task in tasks):
                if observed_scheduler is not None:
                    _, values = _metrics(
                        "http://127.0.0.1:8000/metrics", SCHEDULER_METRICS
                    )
                    for name, value in values.items():
                        observed_scheduler[name] = max(
                            observed_scheduler.get(name, 0.0), value
                        )
                await asyncio.sleep(0.05)
            await asyncio.gather(*tasks)
    return sorted(results, key=lambda row: (row.started_at_s, row.request_id))


def _rows_for_phase(raw_rows: list[dict[str, Any]], phase: str) -> list[WorkloadItem]:
    return [
        WorkloadItem.model_validate(row)
        for row in raw_rows
        if row["measurement_phase"] == phase
    ]


def _filter_trace(source: Path, target: Path, request_ids: set[str]) -> int:
    if not source.exists():
        raise FileNotFoundError(source)
    return write_jsonl(
        target,
        (
            row
            for row in read_jsonl(source)
            if str(row.get("request_id", "")) in request_ids
        ),
    )


def _median_range(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "samples": len(values),
    }


def summarize_costs(
    client_rows: list[dict[str, Any]], worker_events: list[dict[str, Any]]
) -> dict[str, Any]:
    by_request = {str(row["request_id"]): row for row in client_rows}
    load_completed = [
        row for row in worker_events if row.get("phase") == "load_completed"
    ]
    report: dict[str, Any] = {"by_path_and_length": {}}
    for path in ("recompute", "lmcache_l1", "mooncake_l2"):
        path_report: dict[str, Any] = {}
        for length in (8192, 16384, 32768):
            prefix = f"h20-v21-{path}-"
            ids = [
                request_id
                for request_id in by_request
                if request_id.startswith(prefix)
                and request_id.endswith(f"-{length // 1024}k")
            ]
            if len(ids) != 3:
                raise ValueError(f"{path}/{length} requires three client samples")
            values: dict[str, Any] = {
                "ttft_ms": _median_range(
                    [float(by_request[request_id]["ttft_ms"]) for request_id in ids]
                ),
                "e2e_ms": _median_range(
                    [float(by_request[request_id]["e2e_ms"]) for request_id in ids]
                ),
            }
            if path != "recompute":
                loads = [
                    row
                    for row in load_completed
                    if str(row.get("request_id")) in ids
                    and row.get("actual_kv_path") == path
                ]
                if len(loads) != 3:
                    raise ValueError(f"{path}/{length} requires three load samples")
                values.update(
                    {
                        "load_ms": _median_range(
                            [float(row["load_ms"]) for row in loads]
                        ),
                        "tokens_per_s": _median_range(
                            [
                                1000.0
                                * int(row["retrieved_tokens"])
                                / float(row["load_ms"])
                                for row in loads
                            ]
                        ),
                        "to_gpu_ms": _median_range(
                            [float(row.get("to_gpu_ms", 0.0)) for row in loads]
                        ),
                        "transfer_bytes": _median_range(
                            [float(row.get("transfer_bytes", 0)) for row in loads]
                        ),
                    }
                )
            path_report[str(length)] = values
        report["by_path_and_length"][path] = path_report

    points = []
    for request_id, row in by_request.items():
        if request_id.startswith("h20-v21-recompute-"):
            points.append((float(row["input_tokens"]), float(row["ttft_ms"])))
    mean_x = statistics.fmean(point[0] for point in points)
    mean_y = statistics.fmean(point[1] for point in points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in points
    ) / denominator
    if slope <= 0:
        raise ValueError("single-H20 recompute fit has non-positive slope")
    report["recompute_fit"] = {
        "prefill_tokens_per_s": 1000.0 / slope,
        "intercept_ms": max(0.0, mean_y - slope * mean_x),
        "samples": len(points),
    }
    for path in ("lmcache_l1", "mooncake_l2"):
        rates = [
            1000.0 * int(row["retrieved_tokens"]) / float(row["load_ms"])
            for row in load_completed
            if row.get("actual_kv_path") == path
            and str(row.get("request_id", "")).startswith(f"h20-v21-{path}-")
        ]
        report[f"{path}_overall_tokens_per_s"] = _median_range(rates)
    return report


def _repository_states() -> dict[str, dict[str, Any]]:
    return {
        name: repository_state(path)
        for name, path in {
            "project": Path("/root/workload-aware-kv-cache"),
            "production_stack": Path("/root/production-stack"),
            "vllm": Path("/root/vllm"),
            "lmcache": Path("/root/LMCache"),
            "mooncake": Path("/root/Mooncake"),
        }.items()
    }


def finalize_existing_run(
    run_dir: Path,
    *,
    profile_manifest: dict[str, Any],
    model_path: Path,
    log_root: Path,
) -> dict[str, Any]:
    profile_path = Path(profile_manifest["profile_path"])
    raw_profile = list(read_jsonl(profile_path))
    measured_ids = {str(row["request_id"]) for row in raw_profile}
    client_path = run_dir / "requests.jsonl"
    router_path = run_dir / "router-trace.jsonl"
    worker_lifecycle = run_dir / "worker-lifecycle.jsonl"
    worker_actual = run_dir / "worker-actual.jsonl"
    joined_path = run_dir / "joined-trace.jsonl"
    client_rows = list(read_jsonl(client_path))
    if len(client_rows) != 42 or sum(bool(row.get("success")) for row in client_rows) != 42:
        raise RuntimeError("existing single-H20 run does not have 42 successful requests")
    join(
        client_path,
        router_path,
        joined_path,
        [worker_lifecycle, worker_actual],
    )
    worker_events = sorted(
        [*read_jsonl(worker_lifecycle), *read_jsonl(worker_actual)],
        key=lambda row: float(row.get("recorded_at", 0.0)),
    )
    lifecycle = _validate_lifecycle(worker_events, measured_ids)
    result_by_id = {str(row["request_id"]): row for row in client_rows}
    for path in ("lmcache_l1", "mooncake_l2"):
        selected = {
            str(result_by_id[request_id].get("selected_kv_path", ""))
            for request_id in measured_ids
            if request_id.startswith(f"h20-v21-{path}-")
        }
        if selected != {path}:
            raise RuntimeError(f"strict {path} selected paths were {selected}")
    recompute_ids = {
        request_id
        for request_id in measured_ids
        if request_id.startswith("h20-v21-recompute-")
    }
    cold_terminals = [
        row
        for row in worker_events
        if row.get("phase") == "request_finished"
        and str(row.get("request_id", "")) in recompute_ids
        and row.get("actual_kv_path") in {"recompute", "local_hbm"}
        and int(row.get("vllm_cached_tokens", 0)) <= 16
    ]
    if len(cold_terminals) != 9:
        raise RuntimeError(
            "cold-reset phase requires nine requests with at most one cached block"
        )
    load_completed = [
        row for row in worker_events if row.get("phase") == "load_completed"
    ]
    for path in ("lmcache_l1", "mooncake_l2"):
        strict_ids = {
            request_id
            for request_id in measured_ids
            if request_id.startswith(f"h20-v21-{path}-")
        }
        actual = [
            row
            for row in load_completed
            if str(row.get("request_id", "")) in strict_ids
            and row.get("actual_kv_path") == path
            and not row.get("path_mismatch")
        ]
        if len(actual) != 9:
            raise RuntimeError(f"strict {path} requires nine actual load samples")

    router_rows = list(read_jsonl(router_path))
    decisions = [row for row in router_rows if row.get("event") == "decision"]
    if len(decisions) != 42:
        raise RuntimeError(f"expected 42 Router decisions, found {len(decisions)}")
    scheduler_max: dict[str, float] = {}
    candidate_fields = {
        "running",
        "waiting",
        "running_prefill_tokens",
        "waiting_prefill_tokens",
        "scheduled_prefill_tokens",
        "scheduled_decode_tokens",
        "skipped_waiting_prefill_tokens",
        "reserved_prefill_tokens",
        "reserved_external_load_ms",
        "reserved_kv_blocks",
        "kv_cache_total_blocks",
    }
    free_blocks = []
    for row in decisions:
        for candidate in row.get("candidates", []):
            for field in candidate_fields:
                scheduler_max[field] = max(
                    scheduler_max.get(field, 0.0), float(candidate.get(field, 0.0))
                )
            free_blocks.append(float(candidate.get("kv_cache_free_blocks", 0.0)))
    scheduler_max["kv_cache_free_blocks_min"] = min(free_blocks)

    final_probe_id = f"h20-v21-unmeasured-final-probe-{run_dir.name}"
    raw_router_path = log_root / "routing/router-trace.jsonl"
    final_probe_rows = [
        row
        for row in read_jsonl(raw_router_path)
        if row.get("event") == "decision" and row.get("request_id") == final_probe_id
    ]
    if len(final_probe_rows) != 1:
        raise RuntimeError("existing run is missing its final reservation probe")
    final_probe = final_probe_rows[0]
    final_candidates = final_probe.get("candidates", [])
    reservation_final = {
        "prefill_tokens": sum(
            int(row.get("reserved_prefill_tokens", 0)) for row in final_candidates
        ),
        "external_load_ms": sum(
            float(row.get("reserved_external_load_ms", 0.0))
            for row in final_candidates
        ),
        "kv_blocks": sum(int(row.get("reserved_kv_blocks", 0)) for row in final_candidates),
    }
    if any(value != 0 for value in reservation_final.values()):
        raise RuntimeError(f"Router reservation leaked: {reservation_final}")
    mooncake_snapshot = (
        ((final_probe.get("v2_context") or {}).get("feedback") or {})
        .get("http://127.0.0.1:8000", {})
        .get("mooncake_snapshot", {})
    )
    if mooncake_snapshot.get("stale", True) or mooncake_snapshot.get(
        "inflight_read_operations", -1
    ) != 0 or mooncake_snapshot.get("inflight_read_bytes", -1) != 0:
        raise RuntimeError(f"invalid final Mooncake snapshot: {mooncake_snapshot}")

    terminal_by_request = {
        str(row["request_id"]): row
        for row in worker_events
        if row.get("phase") == "request_finished"
    }
    adaptive_ids = {
        "local_hbm": "h20-v21-adaptive-local_hbm",
        "lmcache_l1_state": "h20-v21-adaptive-lmcache_l1",
        "mooncake_l2": "h20-v21-adaptive-mooncake_l2",
    }
    adaptive_observed = {
        label: {
            "selected_path": result_by_id[request_id].get("selected_kv_path"),
            "actual_path": terminal_by_request[request_id].get("actual_kv_path"),
            "fallback_reason": terminal_by_request[request_id].get("fallback_reason"),
        }
        for label, request_id in adaptive_ids.items()
    }
    cost_report = summarize_costs(client_rows, worker_events)
    strict_l2_loads = [
        row
        for row in load_completed
        if str(row.get("request_id", "")).startswith("h20-v21-mooncake_l2-")
    ]
    backend_log = log_root / "components/backend.log"
    allocation_warnings = (
        backend_log.read_text(encoding="utf-8", errors="replace").count(
            "Failed to allocate memory block"
        )
        if backend_log.exists()
        else 0
    )
    configs = Path(__file__).resolve().parents[1] / "configs/v2_1_single_h20"
    manifest = {
        "schema_version": "2.1",
        "scope": SCOPE,
        "status": "PASS_WITH_LIMITATION",
        "passed": True,
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "model_name": MODEL_NAME,
        "topology": "1xH20 TP1 monolithic backend",
        "max_model_len": 40960,
        "max_num_seqs": 12,
        "gpu_memory_utilization": 0.90,
        "lmcache_l1_gib": 8,
        "mooncake_l2_gib": 16,
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "request_count": len(client_rows),
        "success_rate": 1.0,
        "repository_states": _repository_states(),
        "configs": {
            path.name: sha256_file(path) for path in sorted(configs.glob("*.yaml"))
        },
        "costs": cost_report,
        "recompute_selected_path_counts": {
            path: sum(
                str(result_by_id[request_id].get("selected_kv_path", "")) == path
                for request_id in recompute_ids
            )
            for path in ("recompute", "local_hbm")
        },
        "adaptive_observed": adaptive_observed,
        "adaptive_l1_isolation_observed": False,
        "scheduler_and_reservation_max": scheduler_max,
        "concurrency_buckets_observed": sorted(
            {
                str(row.get("concurrency_bucket", ""))
                for row in worker_events
                if row.get("concurrency_bucket")
            }
        ),
        "lifecycle": lifecycle,
        "router_reservation_final": reservation_final,
        "mooncake_final_snapshot": mooncake_snapshot,
        "strict_l2_worker_load_operations": len(strict_l2_loads),
        "strict_l2_worker_transfer_bytes": sum(
            int(row.get("transfer_bytes", 0)) for row in strict_l2_loads
        ),
        "l1_capacity_allocation_warning_count": allocation_warnings,
        "storage_io_screening": {
            "sample_bytes": 2147483648,
            "file_storage_median_mib_s": 348.2,
            "data_disk_median_mib_s": 190.1,
            "file_storage_to_data_copy_mib_s": 117.1,
            "file_storage_over_data_disk_ratio": 1.8317,
        },
        "artifacts": {
            "requests": str(client_path),
            "router_trace": str(router_path),
            "worker_lifecycle": str(worker_lifecycle),
            "worker_actual": str(worker_actual),
            "joined_trace": str(joined_path),
        },
        "limitations": [
            "single backend cannot validate multi-backend routing gains",
            "Mooncake L2 uses same-host TCP and is not four-client contention cost",
            "measured costs do not overwrite four-H20 K02 frozen parameters",
            "RemoteBackend does not support online clear; Controller exposes one primary location per instance, so an Adaptive L1-only state was not isolated without restarting the backend",
            "no fixed-4096 versus Adaptive V2.1 performance conclusion",
        ],
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "report.json", manifest)
    write_json(log_root / "single-h20-report.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    stack = project_root / "scripts/v2_1_single_h20_stack.sh"
    configs = project_root / "configs/v2_1_single_h20"
    profile_manifest = build_profile(args.data_root, args.profile_dir)
    profile_path = Path(profile_manifest["profile_path"])
    raw_rows = list(read_jsonl(profile_path))
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_root = args.log_root
    tokenizer = load_tokenizer(str(args.model_path))
    results: list[RequestResult] = []
    observed_scheduler = {name: 0.0 for name in SCHEDULER_METRICS}
    measured_ids = {str(row["request_id"]) for row in raw_rows}

    if args.dry_run:
        report = {
            "scope": SCOPE,
            "dry_run": True,
            "profile": profile_manifest,
            "planned_requests": len(raw_rows),
            "model_path": str(args.model_path),
            "stack": str(stack),
            "configs": sorted(str(path) for path in configs.glob("*.yaml")),
        }
        write_json(run_dir / "dry-run.json", report)
        return report

    try:
        _run(stack, "start")
        cost_base = [
            WorkloadItem.model_validate(row)
            for row in raw_rows
            if row["measurement_phase"] == "recompute"
            and int(row["measurement_round"]) == 0
        ]
        for item in cost_base:
            _direct_warmup(item)
        _run(stack, "reset-hbm", "true")

        scheduler_text, scheduler_initial = _metrics(
            "http://127.0.0.1:8000/metrics", SCHEDULER_METRICS
        )
        missing_scheduler = sorted(
            name for name in SCHEDULER_METRICS if name not in scheduler_text
        )
        if missing_scheduler:
            raise RuntimeError(f"missing Scheduler metrics: {missing_scheduler}")
        mooncake_text, mooncake_initial = _metrics(
            "http://127.0.0.1:9300/metrics", MOONCAKE_METRICS
        )
        # Cumulative read bytes/operation metrics are registered lazily by the
        # first real read. The V2.1 gauges and error counters are eager.
        missing_mooncake = sorted(
            name for name in MOONCAKE_EAGER_METRICS if name not in mooncake_text
        )
        if missing_mooncake:
            raise RuntimeError(f"missing Mooncake metrics: {missing_mooncake}")

        phase_configs = {
            "recompute": configs / "agent-slo-recompute.yaml",
            "lmcache_l1": configs / "agent-slo-force-l1.yaml",
            "mooncake_l2": configs / "agent-slo-force-l2.yaml",
        }
        for phase in ("recompute", "lmcache_l1", "mooncake_l2"):
            _switch_router(stack, phase_configs[phase])
            phase_rows = _rows_for_phase(raw_rows, phase)
            for round_id in range(3):
                if phase == "recompute":
                    _run(stack, "reset-hbm", "true")
                elif phase == "lmcache_l1":
                    _run(stack, "reset-hbm", "false")
                else:
                    _run(stack, "clear-l1")
                    _run(stack, "reset-hbm", "false")
                batch = phase_rows[round_id * 3 : (round_id + 1) * 3]
                results.extend(
                    asyncio.run(
                        _issue_rows(batch, run_id=run_id, tokenizer=tokenizer)
                    )
                )

        mooncake_after_cost_text, mooncake_after_cost = _metrics(
            "http://127.0.0.1:9300/metrics", MOONCAKE_METRICS
        )
        missing_after_read = sorted(
            name for name in MOONCAKE_METRICS if name not in mooncake_after_cost_text
        )
        if missing_after_read:
            raise RuntimeError(
                f"Mooncake metrics missing after strict L2 reads: {missing_after_read}"
            )

        adaptive_rows = {
            str(row["measurement_phase"]): WorkloadItem.model_validate(row)
            for row in raw_rows
            if str(row["measurement_phase"]).startswith("adaptive_")
        }
        _switch_router(stack, configs / "agent-slo-adaptive-local-only.yaml")
        _run(stack, "reset-hbm", "true")
        hbm_seed = adaptive_rows["adaptive_local_hbm"].model_copy(
            update={"request_id": f"h20-v21-unmeasured-hbm-seed-{run_id}"}
        )
        asyncio.run(_issue_rows([hbm_seed], run_id=run_id, tokenizer=tokenizer))
        time.sleep(2)
        results.extend(
            asyncio.run(
                _issue_rows(
                    [adaptive_rows["adaptive_local_hbm"]],
                    run_id=run_id,
                    tokenizer=tokenizer,
                )
            )
        )

        _run(stack, "reset-hbm", "true")
        _switch_router(stack, configs / "agent-slo-adaptive.yaml")
        l1_seed = adaptive_rows["adaptive_lmcache_l1"].model_copy(
            update={"request_id": f"h20-v21-unmeasured-l1-seed-{run_id}"}
        )
        asyncio.run(_issue_rows([l1_seed], run_id=run_id, tokenizer=tokenizer))
        time.sleep(2)
        _run(stack, "reset-hbm", "false")
        _switch_router(stack, configs / "agent-slo-adaptive.yaml")
        l1_discovery = adaptive_rows["adaptive_lmcache_l1"].model_copy(
            update={"request_id": f"h20-v21-unmeasured-l1-discovery-{run_id}"}
        )
        asyncio.run(
            _issue_rows([l1_discovery], run_id=run_id, tokenizer=tokenizer)
        )
        time.sleep(1)
        _run(stack, "reset-hbm", "false")
        results.extend(
            asyncio.run(
                _issue_rows(
                    [adaptive_rows["adaptive_lmcache_l1"]],
                    run_id=run_id,
                    tokenizer=tokenizer,
                )
            )
        )

        _run(stack, "reset-hbm", "true")
        _switch_router(stack, configs / "agent-slo-adaptive.yaml")
        l2_seed = adaptive_rows["adaptive_mooncake_l2"].model_copy(
            update={"request_id": f"h20-v21-unmeasured-l2-seed-{run_id}"}
        )
        asyncio.run(_issue_rows([l2_seed], run_id=run_id, tokenizer=tokenizer))
        time.sleep(2)
        _run(stack, "clear-l1")
        _run(stack, "reset-hbm", "false")
        _switch_router(stack, configs / "agent-slo-adaptive.yaml")
        l2_discovery = adaptive_rows["adaptive_mooncake_l2"].model_copy(
            update={"request_id": f"h20-v21-unmeasured-l2-discovery-{run_id}"}
        )
        asyncio.run(
            _issue_rows([l2_discovery], run_id=run_id, tokenizer=tokenizer)
        )
        time.sleep(1)
        _run(stack, "clear-l1")
        _run(stack, "reset-hbm", "false")
        results.extend(
            asyncio.run(
                _issue_rows(
                    [adaptive_rows["adaptive_mooncake_l2"]],
                    run_id=run_id,
                    tokenizer=tokenizer,
                )
            )
        )

        _run(stack, "reset-hbm", "true")
        _switch_router(stack, configs / "agent-slo-adaptive.yaml")
        concurrency_rows = _rows_for_phase(raw_rows, "concurrency_16k_c12")
        results.extend(
            asyncio.run(
                _issue_rows(
                    concurrency_rows,
                    run_id=run_id,
                    tokenizer=tokenizer,
                    concurrency=12,
                    stagger_after=6,
                    observed_scheduler=observed_scheduler,
                )
            )
        )
        time.sleep(2)
        final_probe = concurrency_rows[0].model_copy(
            update={"request_id": f"h20-v21-unmeasured-final-probe-{run_id}"}
        )
        asyncio.run(_issue_rows([final_probe], run_id=run_id, tokenizer=tokenizer))
        time.sleep(1)

        if len(results) != 42 or any(not result.success for result in results):
            raise RuntimeError(
                f"expected 42 successful requests, got "
                f"{sum(result.success for result in results)}/{len(results)}"
            )
        client_path = run_dir / "requests.jsonl"
        write_jsonl(client_path, (result.model_dump(mode="json") for result in results))
        router_path = run_dir / "router-trace.jsonl"
        worker_lifecycle = run_dir / "worker-lifecycle.jsonl"
        worker_actual = run_dir / "worker-actual.jsonl"
        _filter_trace(
            log_root / "routing/router-trace.jsonl", router_path, measured_ids
        )
        _filter_trace(
            log_root / "serving/backend.connector-trace.jsonl",
            worker_lifecycle,
            measured_ids,
        )
        _filter_trace(
            log_root / "serving/backend.connector-actual-trace.jsonl",
            worker_actual,
            measured_ids,
        )
        return finalize_existing_run(
            run_dir,
            profile_manifest=profile_manifest,
            model_path=args.model_path,
            log_root=log_root,
        )
    finally:
        if not args.keep_stack:
            subprocess.run([str(stack), "stop"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2.1 required single-H20 validation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-stack", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--finalize-run-dir", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/workload-aware-kv-cache-data"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path(
            "/root/workload-aware-kv-cache-data/processed/single_h20_v2_1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/workload-aware-kv-cache-data/runs/single_h20_v2_1"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/root/log/workload-aware-kv-cache/v2-1-single-h20"),
    )
    args = parser.parse_args()
    if not args.model_path.is_dir():
        parser.error(f"model path does not exist: {args.model_path}")
    if args.finalize_run_dir:
        profile = build_profile(args.data_root, args.profile_dir)
        report = finalize_existing_run(
            args.finalize_run_dir,
            profile_manifest=profile,
            model_path=args.model_path,
            log_root=args.log_root,
        )
    else:
        report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
