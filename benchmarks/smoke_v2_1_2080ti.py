from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from benchmarks.io_utils import read_jsonl, write_json


SCHEDULER_METRICS = {
    "vllm:waiting_prefill_tokens",
    "vllm:running_prefill_tokens",
    "vllm:active_decode_sequences",
    "vllm:scheduled_prefill_tokens",
    "vllm:scheduled_decode_tokens",
    "vllm:skipped_waiting_prefill_tokens",
    "vllm:kv_cache_free_blocks",
    "vllm:kv_cache_total_blocks",
}


def _prompt(model_path: Path, target_tokens: int) -> tuple[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    source = "def workload_aware_cache(request):\n    return request.session_id\n"
    text = source * max(1, target_tokens // 12)
    token_ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(token_ids), len(token_ids)


def _metric_sum(text: str, name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = line.split(maxsplit=1)
        if len(metric) != 2 or metric[0].split("{", 1)[0] != name:
            continue
        total += float(metric[1])
    return total


def _metrics(url: str) -> tuple[str, dict[str, float]]:
    text = requests.get(url, timeout=10).text
    names = SCHEDULER_METRICS | {
        "mooncake_transfer_read_bytes",
        "mooncake_transfer_read_operation_count",
        "mooncake_transfer_inflight_read_operations",
        "mooncake_transfer_inflight_read_bytes",
        "mooncake_transfer_read_failures",
        "mooncake_transfer_read_misses",
    }
    return text, {name: _metric_sum(text, name) for name in names}


def _switch_router(stack: Path, config: Path) -> None:
    subprocess.run([str(stack), "router", str(config)], check=True)
    time.sleep(0.5)


def _request(
    request_id: str,
    prompt: str,
    prompt_tokens: int,
    prefix_hash: str,
    *,
    max_tokens: int = 8,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        "http://127.0.0.1:9003/v1/completions",
        headers={
            "X-Request-Id": request_id,
            "X-Session-ID": "v2-1-smoke-session",
            "X-Prefix-Hash": prefix_hash,
            "X-Prompt-Tokens": str(prompt_tokens),
            "X-Shared-Prefix-Tokens": str(prompt_tokens),
            "X-Expected-Output-Tokens": str(max_tokens),
        },
        json={
            "model": "Qwen3-0.6B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=180,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    return {
        "request_id": request_id,
        "elapsed_ms": elapsed_ms,
        "backend_id": response.headers.get("X-Backend-ID", ""),
        "selected_kv_path": response.headers.get("X-KV-Path", ""),
        "route_reason": response.headers.get("X-Route-Reason", ""),
    }


def _events(log_root: Path, request_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (
        log_root / "serving/backend.connector-trace.jsonl",
        log_root / "serving/backend.connector-actual-trace.jsonl",
    ):
        if path.exists():
            rows.extend(
                row
                for row in read_jsonl(path)
                if row.get("schema_version") == "2.1"
                and str(row.get("request_id", "")) in request_ids
            )
    return sorted(rows, key=lambda row: float(row.get("recorded_at", 0.0)))


def _validate_lifecycle(
    events: list[dict[str, Any]], request_ids: set[str]
) -> dict[str, Any]:
    by_request: dict[str, list[dict[str, Any]]] = {
        request_id: [] for request_id in request_ids
    }
    for row in events:
        by_request[str(row["request_id"])].append(row)
    errors: list[str] = []
    for request_id, rows in by_request.items():
        phases = [str(row.get("phase", "")) for row in rows]
        if phases.count("scheduler_seen") != 1:
            errors.append(f"{request_id}: scheduler_seen count")
        terminals = [row for row in rows if row.get("terminal") is True]
        if phases.count("request_finished") != 1 or len(terminals) != 1:
            errors.append(f"{request_id}: terminal count")
        identities = {
            (
                str(row.get("attempt_id", "")),
                str(row.get("decision_id", "")),
                str(row.get("backend_id", "")),
            )
            for row in rows
        }
        if len(identities) != 1 or any(
            not value for value in next(iter(identities), ())
        ):
            errors.append(f"{request_id}: inconsistent identity")
        timestamps = [float(row.get("recorded_at", 0.0)) for row in rows]
        if timestamps != sorted(timestamps):
            errors.append(f"{request_id}: phase order")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "request_count": len(by_request),
        "event_count": len(events),
        "unique_terminal_per_request": True,
        "identity_complete": True,
        "phase_order_valid": True,
    }


def run(model_path: Path, log_root: Path, target_tokens: int) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    stack = project_root / "scripts/v2_1_2080ti_stack.sh"
    configs = project_root / "configs/v2_1"
    prompt, prompt_tokens = _prompt(model_path, target_tokens)
    prefix_hash = hashlib.sha256(prompt.encode()).hexdigest()
    run_id = str(int(time.time() * 1000))
    results: list[dict[str, Any]] = []

    _switch_router(stack, configs / "agent-slo-2080ti-recompute.yaml")
    subprocess.run([str(stack), "reset-hbm", "true"], check=True)
    cold_id = f"v2-1-smoke-{run_id}-cold"
    results.append(_request(cold_id, prompt, prompt_tokens, prefix_hash))
    time.sleep(1)

    _switch_router(stack, configs / "agent-slo-2080ti-force-l1.yaml")
    subprocess.run([str(stack), "reset-hbm", "false"], check=True)
    l1_id = f"v2-1-smoke-{run_id}-l1"
    results.append(_request(l1_id, prompt, prompt_tokens, prefix_hash))
    time.sleep(1)

    _switch_router(stack, configs / "agent-slo-2080ti-force-l2.yaml")
    subprocess.run([str(stack), "reset-hbm", "false"], check=True)
    subprocess.run([str(stack), "clear-l1"], check=True)
    time.sleep(1)
    _, mooncake_before = _metrics("http://127.0.0.1:9300/metrics")
    l2_id = f"v2-1-smoke-{run_id}-l2"
    results.append(_request(l2_id, prompt, prompt_tokens, prefix_hash))
    time.sleep(1)
    mooncake_text, mooncake_after = _metrics("http://127.0.0.1:9300/metrics")

    _switch_router(stack, configs / "agent-slo-2080ti.yaml")
    concurrent_ids = [f"v2-1-smoke-{run_id}-concurrent-{index}" for index in range(2)]
    observed_scheduler: dict[str, float] = {name: 0.0 for name in SCHEDULER_METRICS}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _request,
                request_id,
                prompt,
                prompt_tokens,
                prefix_hash,
                max_tokens=16,
            )
            for request_id in concurrent_ids
        ]
        while not all(future.done() for future in futures):
            metrics_text = requests.get(
                "http://127.0.0.1:8000/metrics", timeout=10
            ).text
            for name in SCHEDULER_METRICS:
                observed_scheduler[name] = max(
                    observed_scheduler[name], _metric_sum(metrics_text, name)
                )
            time.sleep(0.05)
        results.extend(future.result() for future in futures)
    time.sleep(2)

    probe_id = f"v2-1-smoke-{run_id}-reservation-probe"
    results.append(_request(probe_id, prompt, prompt_tokens, prefix_hash))
    time.sleep(2)

    scheduler_text = requests.get("http://127.0.0.1:8000/metrics", timeout=10).text
    missing_scheduler = sorted(
        name for name in SCHEDULER_METRICS if name not in scheduler_text
    )
    if missing_scheduler:
        raise RuntimeError(f"missing V2.1 scheduler metrics: {missing_scheduler}")
    missing_mooncake = sorted(
        name
        for name in {
            "mooncake_transfer_inflight_read_operations",
            "mooncake_transfer_inflight_read_bytes",
            "mooncake_transfer_read_failures",
            "mooncake_transfer_read_misses",
        }
        if name not in mooncake_text
    )
    if missing_mooncake:
        raise RuntimeError(f"missing Mooncake V2.1 metrics: {missing_mooncake}")

    request_ids = {str(row["request_id"]) for row in results}
    events = _events(log_root, request_ids)
    lifecycle = _validate_lifecycle(events, request_ids)
    load_events = {
        str(row["request_id"]): row
        for row in events
        if row.get("phase") == "load_completed"
    }
    for request_id, expected in ((l1_id, "lmcache_l1"), (l2_id, "mooncake_l2")):
        row = load_events.get(request_id)
        if (
            row is None
            or row.get("actual_kv_path") != expected
            or row.get("path_mismatch")
        ):
            raise RuntimeError(f"{request_id} did not execute strict {expected}")

    read_bytes_delta = (
        mooncake_after["mooncake_transfer_read_bytes"]
        - mooncake_before["mooncake_transfer_read_bytes"]
    )
    read_ops_delta = (
        mooncake_after["mooncake_transfer_read_operation_count"]
        - mooncake_before["mooncake_transfer_read_operation_count"]
    )
    if read_bytes_delta <= 0 or read_ops_delta <= 0:
        raise RuntimeError("Mooncake L2 read metrics did not increase")
    if mooncake_after["mooncake_transfer_inflight_read_operations"] != 0:
        raise RuntimeError("Mooncake inflight read operations leaked")
    if mooncake_after["mooncake_transfer_inflight_read_bytes"] != 0:
        raise RuntimeError("Mooncake inflight read bytes leaked")

    router_rows = [
        row
        for row in read_jsonl(log_root / "routing/router-trace.jsonl")
        if row.get("event") == "decision" and row.get("request_id") == probe_id
    ]
    if len(router_rows) != 1:
        raise RuntimeError("reservation probe Router decision is missing")
    probe_backend = str(router_rows[0].get("backend_url", ""))
    router_mooncake_snapshot = (
        (router_rows[0].get("v2_context") or {})
        .get("feedback", {})
        .get(probe_backend, {})
        .get("mooncake_snapshot", {})
    )
    if router_mooncake_snapshot.get("stale", True):
        raise RuntimeError("Router did not ingest fresh Mooncake client metrics")
    reserved_after_completion = sum(
        int(candidate.get("reserved_prefill_tokens", 0))
        for candidate in router_rows[0].get("candidates", [])
    )
    if reserved_after_completion != 0:
        raise RuntimeError("Router reservation leaked after completed requests")

    report = {
        "schema_version": "2.1",
        "scope": "FUNCTIONAL_SMOKE_NOT_PERFORMANCE",
        "run_id": run_id,
        "model_path": str(model_path),
        "prompt_tokens": prompt_tokens,
        "requests": results,
        "strict_l1_actual_path": load_events[l1_id]["actual_kv_path"],
        "strict_l2_actual_path": load_events[l2_id]["actual_kv_path"],
        "scheduler_metrics_present": sorted(SCHEDULER_METRICS),
        "scheduler_metrics_observed_max": observed_scheduler,
        "lifecycle": lifecycle,
        "mooncake_read_bytes_delta": read_bytes_delta,
        "mooncake_read_operations_delta": read_ops_delta,
        "mooncake_inflight_operations_final": mooncake_after[
            "mooncake_transfer_inflight_read_operations"
        ],
        "mooncake_inflight_bytes_final": mooncake_after[
            "mooncake_transfer_inflight_read_bytes"
        ],
        "router_reserved_prefill_tokens_final": reserved_after_completion,
        "router_mooncake_snapshot_fresh": True,
        "passed": True,
        "performance_conclusion": "NOT_MEASURED_ON_2080_TI",
    }
    write_json(log_root / "smoke-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2080 Ti V2.1 smoke")
    parser.add_argument(
        "--model-path", type=Path, default=Path("/root/autodl-fs/models/Qwen3-0.6B")
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/root/log/workload-aware-kv-cache/v2-1-2080ti"),
    )
    parser.add_argument("--target-tokens", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(run(args.model_path, args.log_root, args.target_tokens), indent=2))


if __name__ == "__main__":
    main()
