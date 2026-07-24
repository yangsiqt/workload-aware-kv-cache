from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import requests

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
    _direct_warmup,
    _filter_trace,
    _issue_rows,
)
from benchmarks.smoke_v2_1_2080ti import _validate_lifecycle
from benchmarks.tokenizer_utils import load_tokenizer


SCOPE = "SINGLE_H20_MULTI_TIER_ADAPTIVE_FUNCTIONAL_SMOKE_NOT_FOUR_H20_PERFORMANCE"
MIN_PREFIX_TOKENS = 8192
L1_GIB = 8
L2_GIB = 16
KV_BYTES_PER_TOKEN = 98304


def build_profile(data_root: Path, output_dir: Path) -> dict[str, Any]:
    """Freeze one 16K target and four distinct 32K LRU fillers."""
    formal_path = data_root / "processed/four_h20/swebench.jsonl"
    target: dict[str, Any] | None = None
    fillers: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    for row in read_jsonl(formal_path):
        if int(row["turn_id"]) != 0 or str(row["session_id"]) in seen_sessions:
            continue
        length = int(row["shared_prefix_tokens"])
        if length == 16384 and target is None:
            target = dict(row)
            seen_sessions.add(str(row["session_id"]))
        elif length == 32768 and len(fillers) < 4:
            fillers.append(dict(row))
            seen_sessions.add(str(row["session_id"]))
        if target is not None and len(fillers) == 4:
            break
    if target is None or len(fillers) != 4:
        raise ValueError(
            "multi-tier smoke requires one 16K target and four 32K fillers"
        )

    rows = [target, *fillers]
    roles = ["target", *(f"lru_filler_{index}" for index in range(4))]
    for row, role in zip(rows, roles, strict=True):
        row["request_id"] = f"h20-v21-multitier-{role}"
        row["turn_id"] = 0
        row["expected_output_tokens"] = 8
        row["measurement_phase"] = "adaptive_multitier_lru"
        row["multitier_role"] = role

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "profile.jsonl"
    write_jsonl(profile_path, rows)
    working_set_tokens = sum(int(row["shared_prefix_tokens"]) for row in rows)
    manifest = {
        "schema_version": "2.1",
        "scope": SCOPE,
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "source_path": str(formal_path),
        "source_sha256": sha256_file(formal_path),
        "count": len(rows),
        "target_shared_prefix_tokens": int(target["shared_prefix_tokens"]),
        "filler_shared_prefix_tokens": [
            int(row["shared_prefix_tokens"]) for row in fillers
        ],
        "working_set_tokens": working_set_tokens,
        "estimated_working_set_gib": (working_set_tokens * KV_BYTES_PER_TOKEN / 2**30),
        "l1_gib": L1_GIB,
        "l2_gib": L2_GIB,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def normalize_layout(payload: dict[str, Any]) -> dict[str, int]:
    """Return the longest cached prefix reported for each storage location."""
    if "layout_info_v2" not in payload:
        raise RuntimeError("Controller response is missing layout_info_v2")
    layout: dict[str, int] = {}
    values = payload.get("layout_info_v2")
    if not isinstance(values, list):
        raise RuntimeError("Controller layout_info_v2 is not a list")
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        location = str(value[2])
        try:
            cached_tokens = max(0, int(value[3]))
        except (TypeError, ValueError):
            continue
        layout[location] = max(layout.get(location, 0), cached_tokens)
    return layout


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
    payload = response.json()
    return {"raw": payload, "by_location": normalize_layout(payload)}


def _wait_for_layout(
    item: WorkloadItem,
    tokenizer: Any,
    predicate: Callable[[dict[str, int]], bool],
    *,
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


def _run(stack: Path, *args: str) -> None:
    subprocess.run([str(stack), *args], check=True)


def _lookup_only_refresh(item: WorkloadItem, request_id: str) -> None:
    """Refresh Router metadata without executing a model request.

    The Router performs its Controller lookup before vLLM validates sampling
    parameters. vLLM then rejects ``max_tokens=-1`` with HTTP 400, so this
    control request cannot repopulate HBM or L1.
    """
    headers = {
        "X-Request-ID": request_id,
        "X-Session-ID": item.session_id,
        "X-Prefix-Hash": item.prefix_hash,
        "X-Priority": str(item.priority),
        "X-Prompt-Tokens": str(item.prompt_tokens),
        "X-Shared-Prefix-Tokens": str(item.shared_prefix_tokens),
        "X-Expected-Output-Tokens": "0",
        "X-Route-Policy": "agent_slo_aware",
    }
    response = requests.post(
        "http://127.0.0.1:9003/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [message.model_dump() for message in item.messages],
            "max_tokens": -1,
            "temperature": 0,
            "stream": False,
        },
        headers=headers,
        timeout=30,
    )
    if response.status_code != 400:
        raise RuntimeError(
            f"lookup-only refresh expected HTTP 400, got {response.status_code}"
        )


def _wait_for_router_path(trace_path: Path, request_id: str, expected: str) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if trace_path.exists():
            rows = [
                row
                for row in read_jsonl(trace_path)
                if row.get("event") == "decision"
                and row.get("request_id") == request_id
            ]
            if rows:
                selected = str((rows[-1].get("kv_path") or {}).get("selected_path"))
                if selected != expected:
                    raise RuntimeError(
                        f"Router refresh probe selected {selected}, expected {expected}"
                    )
                return
        time.sleep(0.1)
    raise RuntimeError(f"Router decision not found for {request_id}")


def _refresh_router_tier(
    item: WorkloadItem,
    *,
    expected_path: str,
    run_id: str,
    trace_path: Path,
) -> None:
    first = f"h20-v21-multitier-refresh-{expected_path}-a-{run_id}"
    second = f"h20-v21-multitier-refresh-{expected_path}-b-{run_id}"
    _lookup_only_refresh(item, first)
    time.sleep(1)
    _lookup_only_refresh(item, second)
    _wait_for_router_path(trace_path, second, expected_path)
    time.sleep(0.5)


def _measured_item(target: WorkloadItem, path: str, index: int) -> WorkloadItem:
    return target.model_copy(
        update={"request_id": f"h20-v21-multitier-adaptive-{path}-{index}"}
    )


def _actual_paths(
    worker_events: list[dict[str, Any]], request_ids: set[str]
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for row in worker_events:
        request_id = str(row.get("request_id", ""))
        if request_id not in request_ids or row.get("phase") != "load_completed":
            continue
        path = str(row.get("actual_kv_path", ""))
        if path in {"lmcache_l1", "mooncake_l2"}:
            if request_id in actual:
                raise RuntimeError(f"multiple load_completed events for {request_id}")
            actual[request_id] = path
    return actual


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    stack = project_root / "scripts/v2_1_single_h20_stack.sh"
    smoke_config = (
        project_root / "configs/v2_1_single_h20/agent-slo-adaptive-multitier-smoke.yaml"
    )
    profile_manifest = build_profile(args.data_root, args.profile_dir)
    raw_profile = list(read_jsonl(Path(profile_manifest["profile_path"])))
    target = WorkloadItem.model_validate(raw_profile[0])
    fillers = [WorkloadItem.model_validate(row) for row in raw_profile[1:]]
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.dry_run:
        report = {
            "scope": SCOPE,
            "dry_run": True,
            "profile": profile_manifest,
            "adaptive_measurements": 3,
        }
        write_json(run_dir / "dry-run.json", report)
        return report

    tokenizer = load_tokenizer(str(args.model_path))
    trace_root = args.log_root
    router_trace = trace_root / "routing/router-trace.jsonl"
    results: list[RequestResult] = []
    snapshots: dict[str, Any] = {}
    try:
        _run(stack, "start")
        _run(stack, "router", str(smoke_config))
        _run(stack, "reset-hbm", "true")

        _direct_warmup(target)
        snapshots["before_lru"] = _wait_for_layout(
            target,
            tokenizer,
            lambda layout: layout.get("LocalCPUBackend", 0) >= MIN_PREFIX_TOKENS
            and layout.get("RemoteBackend", 0) >= MIN_PREFIX_TOKENS,
        )
        _run(stack, "reset-hbm", "false")
        _refresh_router_tier(
            target,
            expected_path="lmcache_l1",
            run_id=run_id,
            trace_path=router_trace,
        )
        for index in range(2):
            _run(stack, "reset-hbm", "false")
            results.extend(
                asyncio.run(
                    _issue_rows(
                        [_measured_item(target, "lmcache_l1", index)],
                        run_id=run_id,
                        tokenizer=tokenizer,
                    )
                )
            )

        for filler in fillers:
            _direct_warmup(filler)
        _run(stack, "reset-hbm", "false")
        snapshots["after_lru"] = _wait_for_layout(
            target,
            tokenizer,
            lambda layout: layout.get("LocalCPUBackend", 0) == 0
            and layout.get("RemoteBackend", 0) >= MIN_PREFIX_TOKENS,
        )
        _refresh_router_tier(
            target,
            expected_path="mooncake_l2",
            run_id=run_id,
            trace_path=router_trace,
        )
        _run(stack, "reset-hbm", "false")
        results.extend(
            asyncio.run(
                _issue_rows(
                    [_measured_item(target, "mooncake_l2", 0)],
                    run_id=run_id,
                    tokenizer=tokenizer,
                )
            )
        )

        if len(results) != 3 or any(not result.success for result in results):
            raise RuntimeError(
                f"expected 3 successful Adaptive requests, got "
                f"{sum(result.success for result in results)}/{len(results)}"
            )
        selected = {result.request_id: result.selected_kv_path for result in results}
        expected = {
            "h20-v21-multitier-adaptive-lmcache_l1-0": "lmcache_l1",
            "h20-v21-multitier-adaptive-lmcache_l1-1": "lmcache_l1",
            "h20-v21-multitier-adaptive-mooncake_l2-0": "mooncake_l2",
        }
        if selected != expected:
            raise RuntimeError(f"Adaptive selected paths mismatch: {selected}")

        measured_ids = set(expected)
        requests_path = run_dir / "requests.jsonl"
        router_path = run_dir / "router-trace.jsonl"
        lifecycle_path = run_dir / "worker-lifecycle.jsonl"
        actual_path = run_dir / "worker-actual.jsonl"
        joined_path = run_dir / "joined-trace.jsonl"
        write_jsonl(
            requests_path, (result.model_dump(mode="json") for result in results)
        )
        _filter_trace(router_trace, router_path, measured_ids)
        _filter_trace(
            trace_root / "serving/backend.connector-trace.jsonl",
            lifecycle_path,
            measured_ids,
        )
        _filter_trace(
            trace_root / "serving/backend.connector-actual-trace.jsonl",
            actual_path,
            measured_ids,
        )
        join(requests_path, router_path, joined_path, [lifecycle_path, actual_path])
        worker_events = sorted(
            [*read_jsonl(lifecycle_path), *read_jsonl(actual_path)],
            key=lambda row: float(row.get("recorded_at", 0)),
        )
        lifecycle = _validate_lifecycle(worker_events, measured_ids)
        actual = _actual_paths(worker_events, measured_ids)
        if actual != expected:
            raise RuntimeError(f"Adaptive actual paths mismatch: {actual}")

        wheel_manifest = Path(
            "/root/wheels/workload-aware-kv-cache/v2-1/lmcache/lmcache-v2-1.sha256"
        )
        wheel_sha, wheel_path = wheel_manifest.read_text().split()
        repositories = {
            name: repository_state(path)
            for name, path in {
                "project": Path("/root/workload-aware-kv-cache"),
                "production_stack": Path("/root/production-stack"),
                "vllm": Path("/root/vllm"),
                "lmcache": Path("/root/LMCache"),
                "mooncake": Path("/root/Mooncake"),
            }.items()
        }
        report = {
            "schema_version": "2.1",
            "scope": SCOPE,
            "status": "PASS",
            "passed": True,
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "model_path": str(args.model_path),
            "profile": profile_manifest,
            "capacities": {
                "lmcache_l1_gib": L1_GIB,
                "mooncake_l2_gib": L2_GIB,
                "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            },
            "controller_snapshots": snapshots,
            "adaptive_selected_paths": selected,
            "adaptive_actual_paths": actual,
            "request_count": len(results),
            "success_rate": 1.0,
            "lifecycle": lifecycle,
            "repositories": repositories,
            "lmcache_wheel": {"path": wheel_path, "sha256": wheel_sha},
            "artifacts": {
                "requests": str(requests_path),
                "router_trace": str(router_path),
                "worker_lifecycle": str(lifecycle_path),
                "worker_actual": str(actual_path),
                "joined_trace": str(joined_path),
            },
            "limitations": [
                "single backend cannot validate cross-GPU routing or throughput gains",
                "same-host TCP Mooncake cost is not four-client contention cost",
                "lookup-only HTTP 400 requests are control probes and are excluded from measured requests",
                "this functional smoke contains no formal performance conclusion",
            ],
        }
        write_json(run_dir / "report.json", report)
        write_json(run_dir / "controller-snapshots.json", snapshots)
        return report
    finally:
        if not args.keep_stack:
            subprocess.run([str(stack), "stop"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2.1 multi-tier Adaptive smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-stack", action="store_true")
    parser.add_argument("--run-id")
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
            "/root/workload-aware-kv-cache-data/processed/single_h20_v2_1_multitier"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/workload-aware-kv-cache-data/runs/single_h20_v2_1_multitier"
        ),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/root/log/workload-aware-kv-cache/v2-1-single-h20"),
    )
    args = parser.parse_args()
    if not args.model_path.is_dir():
        parser.error(f"model path does not exist: {args.model_path}")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
