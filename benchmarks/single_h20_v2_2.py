from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import msgspec
import requests
import zmq

from benchmarks.io_utils import (
    read_jsonl,
    repository_state,
    sha256_file,
    write_json,
    write_jsonl,
)
from benchmarks.join_traces import join
from benchmarks.schemas import RequestResult, WorkloadItem
from benchmarks.single_h20_v2_1 import (
    MODEL_NAME,
    MOONCAKE_EAGER_METRICS,
    MOONCAKE_METRICS,
    _direct_warmup,
    _filter_trace,
    _issue_rows,
    _metrics,
    _run,
    _switch_router,
)
from benchmarks.smoke_v2_1_2080ti import SCHEDULER_METRICS, _validate_lifecycle
from benchmarks.tokenizer_utils import load_tokenizer


SCOPE = "SINGLE_H20_V2_2_FUNCTIONAL_VALIDATION_NOT_FOUR_H20_PERFORMANCE"
BACKEND = "http://127.0.0.1:8000"
L1_GIB = 8
L2_GIB = 16
KV_BYTES_PER_TOKEN = 98304
MIN_CONTROLLER_TOKENS = 7936


class KVEventRecorder:
    """Capture the same real vLLM KV event stream consumed by the Router."""

    def __init__(self, endpoint: str, topic: str) -> None:
        self.endpoint = endpoint
        self.topic = topic.encode()
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        time.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._rows)

    def count(self, event_type: str) -> int:
        return sum(row.get("type") == event_type for row in self.rows())

    def _run(self) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        socket.connect(self.endpoint)
        try:
            while not self._stop.is_set():
                if not socket.poll(100):
                    continue
                frames = socket.recv_multipart()
                if len(frames) != 3:
                    continue
                sequence = int.from_bytes(frames[1], "big")
                try:
                    payload = msgspec.msgpack.decode(frames[2])
                except msgspec.DecodeError:
                    continue
                if not isinstance(payload, list) or len(payload) < 2:
                    continue
                recorded_at = time.time()
                for event in payload[1]:
                    if not isinstance(event, dict):
                        continue
                    row = dict(event)
                    row.update(
                        {
                            "sequence": sequence,
                            "recorded_at": recorded_at,
                            "publisher_timestamp": payload[0],
                        }
                    )
                    with self._lock:
                        self._rows.append(row)
        finally:
            socket.close(linger=0)


def _clone(item: WorkloadItem, request_id: str, output_tokens: int = 4) -> WorkloadItem:
    return item.model_copy(
        update={"request_id": request_id, "expected_output_tokens": output_tokens}
    )


def build_profile(data_root: Path, output_dir: Path) -> dict[str, Any]:
    """Freeze three length targets and enough unique 32K HBM fillers."""
    source = data_root / "processed/four_h20/swebench.jsonl"
    targets: dict[int, dict[str, Any]] = {}
    fillers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in read_jsonl(source):
        session_id = str(raw["session_id"])
        if int(raw["turn_id"]) != 0 or session_id in seen:
            continue
        length = int(raw["shared_prefix_tokens"])
        if length in {8192, 16384, 32768} and length not in targets:
            targets[length] = dict(raw)
            seen.add(session_id)
        elif length == 32768 and len(fillers) < 14:
            fillers.append(dict(raw))
            seen.add(session_id)
        if len(targets) == 3 and len(fillers) == 14:
            break
    if len(targets) != 3 or len(fillers) != 14:
        raise RuntimeError("V2.2 single-H20 profile requires 3 targets and 14 fillers")

    rows: list[dict[str, Any]] = []
    for length, raw in sorted(targets.items()):
        row = dict(raw)
        row.update(
            {
                "request_id": f"h20-v22-target-{length // 1024}k",
                "expected_output_tokens": 4,
                "v2_2_role": "target",
            }
        )
        rows.append(row)
    for index, raw in enumerate(fillers):
        row = dict(raw)
        row.update(
            {
                "request_id": f"h20-v22-filler-{index:02d}",
                "expected_output_tokens": 1,
                "v2_2_role": "hbm_filler",
            }
        )
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = output_dir / "profile.jsonl"
    write_jsonl(profile, rows)
    manifest = {
        "schema_version": "2.2",
        "scope": SCOPE,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "profile_path": str(profile),
        "profile_sha256": sha256_file(profile),
        "targets": sorted(targets),
        "filler_count": len(fillers),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def normalize_layout_v3(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Return maximum tokens and revision for every Controller location."""
    values = payload.get("layout_info_v3")
    if not isinstance(values, list):
        raise RuntimeError("Controller response is missing layout_info_v3")
    result: dict[str, dict[str, int]] = {}
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 5:
            continue
        location = str(value[2])
        tokens = max(0, int(value[3]))
        revision = max(0, int(value[4]))
        current = result.get(location, {"tokens": 0, "revision": 0})
        if (tokens, revision) > (current["tokens"], current["revision"]):
            result[location] = {"tokens": tokens, "revision": revision}
    return result


def _token_ids(item: WorkloadItem, tokenizer: Any) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [message.model_dump() for message in item.messages],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "get"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _controller_lookup(item: WorkloadItem, tokenizer: Any) -> dict[str, Any]:
    response = requests.post(
        "http://127.0.0.1:9000/lookup",
        json={"tokens": _token_ids(item, tokenizer)},
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json()
    return {"raw": raw, "by_location": normalize_layout_v3(raw)}


def _wait_for_layout(
    item: WorkloadItem,
    tokenizer: Any,
    predicate: Callable[[dict[str, dict[str, int]]], bool],
    timeout_s: float = 45,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = _controller_lookup(item, tokenizer)
        if predicate(latest["by_location"]):
            return latest
        time.sleep(0.5)
    raise RuntimeError(f"Controller layout did not converge: {latest}")


def _event_sequences_are_contiguous(rows: list[dict[str, Any]]) -> bool:
    sequences: list[int] = []
    for row in rows:
        sequence = int(row["sequence"])
        if not sequences or sequence != sequences[-1]:
            sequences.append(sequence)
    return all(right == left + 1 for left, right in zip(sequences, sequences[1:]))


def _wait_for_event_count(
    recorder: KVEventRecorder, event_type: str, minimum: int, timeout_s: float = 15
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if recorder.count(event_type) >= minimum:
            return
        time.sleep(0.1)
    raise RuntimeError(f"did not observe {minimum} {event_type} events")


def _wait_for_decision(trace: Path, request_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if trace.exists():
            rows = [
                row
                for row in read_jsonl(trace)
                if row.get("event") == "decision"
                and row.get("request_id") == request_id
            ]
            if rows:
                return rows[-1]
        time.sleep(0.1)
    raise RuntimeError(f"Router decision missing for {request_id}")


def _scheduler_seen_generation(trace: Path, request_id: str) -> str:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if trace.exists():
            rows = [
                row
                for row in read_jsonl(trace)
                if row.get("phase") == "scheduler_seen"
                and row.get("request_id") == request_id
            ]
            if rows:
                return str(rows[-1].get("backend_generation", ""))
        time.sleep(0.1)
    raise RuntimeError(f"scheduler_seen missing for {request_id}")


def _issue_one(
    item: WorkloadItem, run_id: str, tokenizer: Any
) -> RequestResult:
    result = asyncio.run(_issue_rows([item], run_id=run_id, tokenizer=tokenizer))[0]
    if not result.success:
        raise RuntimeError(f"request failed: {item.request_id}: {result.error}")
    return result


def _active_reset_probe(
    item: WorkloadItem,
    probe: WorkloadItem,
    run_id: str,
    tokenizer: Any,
    recorder: KVEventRecorder,
) -> tuple[list[RequestResult], dict[str, Any]]:
    clear_before = recorder.count("AllBlocksCleared")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_issue_one, item, run_id, tokenizer)
        deadline = time.monotonic() + 20
        observed_running = False
        while time.monotonic() < deadline and not future.done():
            _, values = _metrics(
                "http://127.0.0.1:8000/metrics", {"vllm:num_requests_running"}
            )
            if values["vllm:num_requests_running"] >= 1:
                observed_running = True
                break
            time.sleep(0.02)
        if not observed_running:
            raise RuntimeError("active reset probe never observed a running request")
        response = requests.post(
            "http://127.0.0.1:8000/reset_prefix_cache",
            timeout=30,
        )
        response.raise_for_status()
        active_result = future.result(timeout=900)
    time.sleep(0.25)
    clear_after = recorder.count("AllBlocksCleared")
    probe_result = _issue_one(probe, run_id, tokenizer)
    return [active_result, probe_result], {
        "running_observed": observed_running,
        "all_cleared_before": clear_before,
        "all_cleared_after": clear_after,
        "reset_emitted_clear": clear_after != clear_before,
    }


def _repository_states() -> dict[str, dict[str, Any]]:
    return {
        name: repository_state(path)
        for name, path in {
            "project": Path("/root/workload-aware-kv-cache"),
            "production_stack": Path("/root/production-stack"),
            "lmcache": Path("/root/LMCache"),
            "vllm": Path("/root/vllm"),
            "mooncake": Path("/root/Mooncake"),
        }.items()
    }


def _validate_and_report(
    run_dir: Path,
    results: list[RequestResult],
    measured_ids: set[str],
    runtime: dict[str, Any],
    configs: Path,
    log_root: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    client = run_dir / "requests.jsonl"
    router = run_dir / "router-trace.jsonl"
    lifecycle = run_dir / "worker-lifecycle.jsonl"
    actual = run_dir / "worker-actual.jsonl"
    joined = run_dir / "joined-trace.jsonl"
    raw_router = log_root / "routing/router-trace.jsonl"
    raw_lifecycle = log_root / "serving/backend.connector-trace.jsonl"
    raw_actual = log_root / "serving/backend.connector-actual-trace.jsonl"
    write_jsonl(client, (row.model_dump(mode="json") for row in results))
    _filter_trace(raw_router, router, measured_ids)
    _filter_trace(raw_lifecycle, lifecycle, measured_ids)
    _filter_trace(raw_actual, actual, measured_ids)
    join(client, router, joined, [lifecycle, actual])

    if len(results) != len(measured_ids) or any(not row.success for row in results):
        raise RuntimeError("measured requests are incomplete or unsuccessful")
    router_rows = [row for row in read_jsonl(router) if row.get("event") == "decision"]
    if len(router_rows) != len(measured_ids):
        raise RuntimeError("Router decision count does not match measured requests")
    # Lifecycle and actual execution feedback are written by separate
    # producers. Their file concatenation order is not their causal order.
    worker_rows = sorted(
        [*read_jsonl(lifecycle), *read_jsonl(actual)],
        key=lambda row: float(row.get("recorded_at", 0.0)),
    )
    lifecycle_report = _validate_lifecycle(worker_rows, measured_ids)
    decisions = {str(row["request_id"]): row for row in router_rows}

    strict_expected = {
        **{f"h20-v22-s03-recompute-{n}k": "recompute" for n in (8, 16, 32)},
        **{f"h20-v22-s03-l1-{n}k": "lmcache_l1" for n in (8, 16, 32)},
        **{f"h20-v22-s03-l2-{n}k": "mooncake_l2" for n in (8, 16, 32)},
    }
    load_rows = [row for row in worker_rows if row.get("phase") == "load_completed"]
    terminal_rows = {
        str(row["request_id"]): row
        for row in worker_rows
        if row.get("phase") == "request_finished"
    }
    strict_observed: dict[str, dict[str, str]] = {}
    for request_id, expected in strict_expected.items():
        selected = str((decisions[request_id].get("kv_path") or {}).get("selected_path"))
        loads = [row for row in load_rows if row.get("request_id") == request_id]
        actual_path = str(
            loads[-1].get("actual_kv_path")
            if loads
            else terminal_rows[request_id].get("actual_kv_path")
        )
        strict_observed[request_id] = {"selected": selected, "actual": actual_path}
        if selected != expected or actual_path != expected:
            raise RuntimeError(
                f"strict path mismatch for {request_id}: {selected}/{actual_path}"
            )

    hbm_before = decisions["h20-v22-s02-hbm-before"]["candidates"][0]
    hbm_after = decisions["h20-v22-s02-hbm-after"]["candidates"][0]
    if (
        hbm_before.get("cache_source") != "vllm_kv_event"
        or int(hbm_before.get("cached_tokens", 0)) < 8192
    ):
        raise RuntimeError(f"Router did not consume the real HBM store: {hbm_before}")
    after_path = str(
        (decisions["h20-v22-s02-hbm-after"].get("kv_path") or {}).get(
            "selected_path"
        )
    )
    if (
        int(hbm_after.get("local_hbm_cached_tokens", -1)) > 256
        or after_path == "local_hbm"
    ):
        raise RuntimeError(f"Router did not consume the real HBM removal: {hbm_after}")

    refresh_rows = [
        row
        for request_id, row in decisions.items()
        if request_id.startswith("h20-v22-s04-")
    ]
    refreshes = [row.get("cache_tier_refresh") or {} for row in refresh_rows]
    waits = [float(row.get("wait_ms", 0.0)) for row in refreshes]
    if len(refreshes) != 12 or any(
        not row.get("attempted")
        or not row.get("completed")
        or row.get("timed_out")
        or row.get("error")
        or int(row.get("observations", 0)) <= 0
        for row in refreshes
    ):
        raise RuntimeError(f"Controller refresh gate failed: {refreshes}")
    if max(waits) >= 200:
        raise RuntimeError(f"Controller refresh exceeded 200ms: {waits}")
    p90_wait = statistics.quantiles(waits, n=10, method="inclusive")[8]
    refresh_status = "PASS" if p90_wait <= 150 else "PASS_WITH_RISK"
    refresh_selected = [
        str((row.get("kv_path") or {}).get("selected_path")) for row in refresh_rows
    ]
    if any(path not in {"lmcache_l1", "mooncake_l2"} for path in refresh_selected):
        raise RuntimeError(
            f"current request did not use refreshed external state: {refresh_selected}"
        )

    event_rows = list(read_jsonl(run_dir / "vllm-kv-events.jsonl"))
    event_counts = {
        event_type: sum(row.get("type") == event_type for row in event_rows)
        for event_type in ("BlockStored", "BlockRemoved", "AllBlocksCleared")
    }
    if any(event_counts[name] <= 0 for name in event_counts):
        raise RuntimeError(f"incomplete real KV event evidence: {event_counts}")
    if not _event_sequences_are_contiguous(event_rows):
        raise RuntimeError("raw vLLM KV event stream contains a sequence gap")
    if runtime["active_reset"]["reset_emitted_clear"]:
        raise RuntimeError("non-forced reset cleared HBM during an active request")

    generations = {
        str(row.get("backend_generation", ""))
        for row in worker_rows
        if row.get("phase") == "scheduler_seen" and row.get("backend_generation")
    }
    active_generation = _scheduler_seen_generation(
        raw_lifecycle, "h20-v22-s02-active-reset"
    )
    probe_generation = _scheduler_seen_generation(
        raw_lifecycle, "h20-v22-s02-active-probe"
    )
    if len(generations) < 2 or active_generation != probe_generation:
        raise RuntimeError(
            "HBM generation did not advance on successful clears or changed "
            "after a rejected active reset"
        )

    _, mooncake = _metrics("http://127.0.0.1:9300/metrics", MOONCAKE_METRICS)
    if (
        mooncake.get("mooncake_transfer_read_operation_count", 0) < 3
        or mooncake.get("mooncake_transfer_read_bytes", 0) <= 0
        or mooncake.get("mooncake_transfer_inflight_read_operations", -1) != 0
        or mooncake.get("mooncake_transfer_inflight_read_bytes", -1) != 0
    ):
        raise RuntimeError(f"invalid final Mooncake metrics: {mooncake}")

    report = {
        "schema_version": "2.2",
        "scope": SCOPE,
        "status": refresh_status,
        "passed": True,
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "topology": "1xH20 TP1 monolithic backend",
        "model": MODEL_NAME,
        "max_model_len": 40960,
        "max_num_seqs": 12,
        "gpu_memory_utilization": 0.90,
        "lmcache_l1_gib": L1_GIB,
        "mooncake_l2_gib": L2_GIB,
        "request_count": len(results),
        "success_rate": 1.0,
        "profile": profile,
        "repository_states": _repository_states(),
        "config_sha256": {
            path.name: sha256_file(path) for path in sorted(configs.glob("*.yaml"))
        },
        "event_counts": event_counts,
        "event_sequences_contiguous": True,
        "hbm_router_evidence": {"before": hbm_before, "after": hbm_after},
        "backend_generations": sorted(generations),
        "active_reset": runtime["active_reset"],
        "strict_paths": strict_observed,
        "controller_snapshots": runtime["controller_snapshots"],
        "controller_refresh": {
            "status": refresh_status,
            "samples": len(waits),
            "p50_ms": statistics.median(waits),
            "p90_ms": p90_wait,
            "max_ms": max(waits),
            "selected_paths": refresh_selected,
        },
        "kv_capacity": runtime["kv_capacity"],
        "hbm_lru_fillers_used": runtime["hbm_lru_fillers_used"],
        "mooncake_final": mooncake,
        "lifecycle": lifecycle_report,
        "artifacts": {
            "requests": str(client),
            "router_trace": str(router),
            "worker_lifecycle": str(lifecycle),
            "worker_actual": str(actual),
            "joined_trace": str(joined),
            "vllm_kv_events": str(run_dir / "vllm-kv-events.jsonl"),
        },
        "limitations": [
            "single backend cannot validate cross-GPU routing gains",
            "same-host TCP Mooncake is not four-client contention cost",
            "this run does not compare fixed and adaptive performance",
            "this run does not overwrite four-H20 frozen cost parameters",
        ],
    }
    write_json(run_dir / "report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    stack = root / "scripts/v2_2_single_h20_stack.sh"
    configs = root / "configs/v2_2_single_h20"
    profile = build_profile(args.data_root, args.profile_dir)
    rows = [WorkloadItem.model_validate(row) for row in read_jsonl(Path(profile["profile_path"]))]
    targets = {row.shared_prefix_tokens: row for row in rows if "target" in row.request_id}
    fillers = [row for row in rows if "filler" in row.request_id]
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.dry_run:
        result = {
            "schema_version": "2.2",
            "scope": SCOPE,
            "dry_run": True,
            "planned_measured_requests": 26,
            "profile": profile,
            "configs": sorted(str(path) for path in configs.glob("*.yaml")),
        }
        write_json(run_dir / "dry-run.json", result)
        return result

    tokenizer = load_tokenizer(str(args.model_path))
    results: list[RequestResult] = []
    measured_ids: set[str] = set()
    runtime: dict[str, Any] = {"controller_snapshots": {}}
    recorder: KVEventRecorder | None = None

    def measured(item: WorkloadItem) -> RequestResult:
        if item.request_id in measured_ids:
            raise RuntimeError(f"duplicate measured request ID: {item.request_id}")
        result = _issue_one(item, run_id, tokenizer)
        measured_ids.add(item.request_id)
        results.append(result)
        return result

    try:
        _run(stack, "start")
        recorder = KVEventRecorder("tcp://127.0.0.1:9400", "workload-aware-kv")
        recorder.start()
        trace = args.log_root / "routing/router-trace.jsonl"
        lifecycle = args.log_root / "serving/backend.connector-trace.jsonl"

        scheduler_text, scheduler_values = _metrics(
            "http://127.0.0.1:8000/metrics", SCHEDULER_METRICS
        )
        missing_scheduler = [name for name in SCHEDULER_METRICS if name not in scheduler_text]
        if missing_scheduler:
            raise RuntimeError(f"missing Scheduler metrics: {missing_scheduler}")
        mooncake_text, _ = _metrics(
            "http://127.0.0.1:9300/metrics", MOONCAKE_EAGER_METRICS
        )
        missing_mooncake = [name for name in MOONCAKE_EAGER_METRICS if name not in mooncake_text]
        if missing_mooncake:
            raise RuntimeError(f"missing Mooncake metrics: {missing_mooncake}")
        runtime["kv_capacity"] = {
            "total_blocks": int(scheduler_values["vllm:kv_cache_total_blocks"]),
            "block_size": 16,
            "tokens": int(scheduler_values["vllm:kv_cache_total_blocks"]) * 16,
        }

        for target in targets.values():
            _direct_warmup(target)
        _run(stack, "reset-hbm", "true")

        _switch_router(stack, configs / "agent-slo-adaptive.yaml")
        seed = _clone(targets[16384], "h20-v22-s02-hbm-seed")
        before = _clone(targets[16384], "h20-v22-s02-hbm-before")
        stored_before = recorder.count("BlockStored")
        measured(seed)
        # vLLM drains reset/cache events while processing model output.  The
        # first post-reset request therefore publishes AllBlocksCleared and its
        # new BlockStored event in the same scheduler cycle.
        _wait_for_event_count(recorder, "AllBlocksCleared", 1)
        _wait_for_event_count(recorder, "BlockStored", stored_before + 1)
        time.sleep(0.25)
        measured(before)
        before_decision = _wait_for_decision(trace, before.request_id)
        if before_decision["candidates"][0].get("cache_source") != "vllm_kv_event":
            raise RuntimeError("Router did not use vLLM events for the warm HBM target")

        active = _clone(targets[32768], "h20-v22-s02-active-reset", 512)
        active_probe = _clone(targets[8192], "h20-v22-s02-active-probe")
        active_results, runtime["active_reset"] = _active_reset_probe(
            active, active_probe, run_id, tokenizer, recorder
        )
        for result in active_results:
            measured_ids.add(result.request_id)
            results.append(result)
        active_generation = _scheduler_seen_generation(lifecycle, active.request_id)
        probe_generation = _scheduler_seen_generation(lifecycle, active_probe.request_id)
        runtime["active_reset"].update(
            {
                "active_generation": active_generation,
                "probe_generation": probe_generation,
                "generation_unchanged": active_generation == probe_generation,
            }
        )
        if active_generation != probe_generation:
            raise RuntimeError("active non-forced reset unexpectedly changed generation")

        total_tokens = runtime["kv_capacity"]["tokens"]
        planned_fillers = min(14, max(10, math.ceil(total_tokens / 32768) + 2))
        removed_before = recorder.count("BlockRemoved")
        used = 0
        for filler in fillers:
            _direct_warmup(filler)
            used += 1
            if used >= planned_fillers and recorder.count("BlockRemoved") > removed_before:
                break
        runtime["hbm_lru_fillers_used"] = used
        _wait_for_event_count(recorder, "BlockRemoved", removed_before + 1)
        time.sleep(0.25)
        after = _clone(targets[16384], "h20-v22-s02-hbm-after")
        measured(after)
        after_decision = _wait_for_decision(trace, after.request_id)
        after_candidate = after_decision["candidates"][0]
        after_path = str((after_decision.get("kv_path") or {}).get("selected_path"))
        if (
            int(after_candidate.get("local_hbm_cached_tokens", -1)) > 256
            or after_path == "local_hbm"
        ):
            raise RuntimeError("oldest HBM target was not evicted after capacity fill")

        _run(stack, "reset-hbm", "true")
        _switch_router(stack, configs / "agent-slo-recompute.yaml")
        for length in (8192, 16384, 32768):
            measured(_clone(targets[length], f"h20-v22-s03-recompute-{length // 1024}k"))
        for length, target in targets.items():
            runtime["controller_snapshots"][f"seed_{length}"] = _wait_for_layout(
                target,
                tokenizer,
                lambda layout, minimum=max(MIN_CONTROLLER_TOKENS, length - 256): (
                    layout.get("LocalCPUBackend", {}).get("tokens", 0) >= minimum
                    and layout.get("RemoteBackend", {}).get("tokens", 0) >= minimum
                ),
            )

        _run(stack, "reset-hbm", "false")
        _switch_router(stack, configs / "agent-slo-force-l1.yaml")
        for length in (8192, 16384, 32768):
            measured(_clone(targets[length], f"h20-v22-s03-l1-{length // 1024}k"))

        for filler in fillers[:3]:
            _direct_warmup(filler)
        for length, target in targets.items():
            minimum = max(MIN_CONTROLLER_TOKENS, length - 256)
            runtime["controller_snapshots"][f"l2_only_{length}"] = _wait_for_layout(
                target,
                tokenizer,
                lambda layout, minimum=minimum: (
                    layout.get("LocalCPUBackend", {}).get("tokens", 0) == 0
                    and layout.get("RemoteBackend", {}).get("tokens", 0) >= minimum
                ),
            )
        _run(stack, "reset-hbm", "false")
        _switch_router(stack, configs / "agent-slo-force-l2.yaml")
        for length in (8192, 16384, 32768):
            measured(_clone(targets[length], f"h20-v22-s03-l2-{length // 1024}k"))

        for policy in ("fixed", "adaptive"):
            _switch_router(stack, configs / f"agent-slo-{policy}.yaml")
            for repeat in range(2):
                for length in (8192, 16384, 32768):
                    _run(stack, "reset-hbm", "false")
                    item = _clone(
                        targets[length],
                        f"h20-v22-s04-{policy}-r{repeat}-{length // 1024}k",
                    )
                    measured(item)
                    decision = _wait_for_decision(trace, item.request_id)
                    refresh = decision.get("cache_tier_refresh") or {}
                    if not refresh.get("completed") or refresh.get("timed_out"):
                        raise RuntimeError(
                            f"current request Controller refresh failed: {refresh}"
                        )

        recorder.stop()
        write_jsonl(run_dir / "vllm-kv-events.jsonl", recorder.rows())
        report = _validate_and_report(
            run_dir,
            results,
            measured_ids,
            runtime,
            configs,
            args.log_root,
            profile,
        )
        write_json(args.log_root / "single-h20-v2-2-report.json", report)
        return report
    finally:
        if recorder is not None:
            recorder.stop()
            event_path = run_dir / "vllm-kv-events.jsonl"
            if not event_path.exists() and recorder.rows():
                write_jsonl(event_path, recorder.rows())
        if not args.keep_stack:
            subprocess.run([str(stack), "stop"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2.2 single-H20 validation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-stack", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/root/workload-aware-kv-cache-data")
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("/root/workload-aware-kv-cache-data/processed/single_h20_v2_2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/workload-aware-kv-cache-data/runs/single_h20_v2_2"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/root/log/workload-aware-kv-cache/v2-2-single-h20"),
    )
    args = parser.parse_args()
    if not args.model_path.is_dir():
        parser.error(f"model path does not exist: {args.model_path}")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
