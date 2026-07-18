from __future__ import annotations

import argparse
import asyncio
import math
import random
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.analyze_results import summarize
from benchmarks.io_utils import (
    load_yaml,
    project_commit,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from benchmarks.schemas import (
    ArrivalTraceItem,
    RequestResult,
    RunManifest,
    WorkloadItem,
)
from benchmarks.sse import SSEAccumulator
from benchmarks.tokenizer_utils import load_tokenizer


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def _request(
    session: aiohttp.ClientSession,
    item: WorkloadItem,
    *,
    run_id: str,
    endpoint: str,
    model: str,
    tokenizer: Any,
    route_policy: str,
    temperature: float,
    offered_at: float,
) -> RequestResult:
    started = time.perf_counter()
    wall_started = time.time()
    chunk_times: list[float] = []
    status: int | None = None
    response_headers: dict[str, str] = {}
    error: str | None = None
    stream = SSEAccumulator()
    try:
        payload = {
            "model": model,
            "messages": [message.model_dump() for message in item.messages],
            "max_tokens": item.expected_output_tokens,
            "temperature": temperature,
            "stream": True,
        }
        headers = {
            "X-Request-ID": item.request_id,
            "X-Session-ID": item.session_id,
            "X-Prefix-Hash": item.prefix_hash,
            "X-Priority": str(item.priority),
            "X-Prompt-Tokens": str(item.prompt_tokens),
            "X-Expected-Output-Tokens": str(item.expected_output_tokens),
            "X-Route-Policy": route_policy,
        }
        async with session.post(endpoint, json=payload, headers=headers) as response:
            status = response.status
            response_headers = {
                key.lower(): value for key, value in response.headers.items()
            }
            if status < 200 or status >= 300:
                error = (await response.text())[:1000]
            else:
                async for raw in response.content.iter_any():
                    emitted = stream.feed(raw)
                    chunk_times.extend(time.perf_counter() for _ in emitted)
                error = stream.validation_error()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    completed = time.perf_counter()
    output_text = stream.text
    output_tokens = (
        len(tokenizer.encode(output_text, add_special_tokens=False))
        if output_text
        else 0
    )
    ttft = (chunk_times[0] - started) * 1000 if chunk_times else None
    e2e = (completed - started) * 1000
    tpot = None
    if output_tokens > 1 and ttft is not None:
        tpot = max(0.0, (e2e - ttft) / (output_tokens - 1))
    itl = [(right - left) * 1000 for left, right in zip(chunk_times, chunk_times[1:])]
    hit_header = response_headers.get("x-mock-cache-hit")
    return RequestResult(
        run_id=run_id,
        request_id=item.request_id,
        session_id=item.session_id,
        turn_id=item.turn_id,
        dataset_name=item.dataset_name,
        request_type=item.request_type,
        prefix_hash=item.prefix_hash,
        priority=item.priority,
        route_policy=response_headers.get("x-route-policy", route_policy),
        backend_id=response_headers.get("x-backend-id"),
        route_reason=response_headers.get("x-route-reason"),
        cache_hit=None if hit_header is None else hit_header.lower() == "true",
        offered_at_s=offered_at,
        started_at_s=wall_started,
        completed_at_s=time.time(),
        ttft_ms=ttft,
        e2e_ms=e2e,
        tpot_ms=tpot,
        inter_chunk_latencies_ms=itl,
        input_tokens=item.prompt_tokens,
        output_tokens=output_tokens,
        status_code=status,
        success=error is None,
        error=error,
    )


async def run(args: argparse.Namespace) -> Path:
    config = load_yaml(args.config)
    items = [WorkloadItem.model_validate(row) for row in read_jsonl(args.workload)]
    tokenizer = load_tokenizer(config["tokenizer_path"])
    run_id = (
        args.run_id
        or f"{args.mode}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    )
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    timeout = aiohttp.ClientTimeout(total=float(config.get("timeout_s", 600)))
    connector = aiohttp.TCPConnector(limit=int(config.get("max_in_flight", 64)))
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[RequestResult] = []
    origin = time.perf_counter()

    by_session: dict[str, list[WorkloadItem]] = defaultdict(list)
    for item in items:
        by_session[item.session_id].append(item)
    for session_items in by_session.values():
        session_items.sort(key=lambda value: value.turn_id)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def issue(item: WorkloadItem, offered: float) -> None:
            async with semaphore:
                result = await _request(
                    session,
                    item,
                    run_id=run_id,
                    endpoint=config["endpoint"],
                    model=config["model"],
                    tokenizer=tokenizer,
                    route_policy=args.route_policy,
                    temperature=float(config.get("temperature", 0.0)),
                    offered_at=offered,
                )
                results.append(result)

        if args.mode == "closed_loop":

            async def run_session(session_items: list[WorkloadItem]) -> None:
                for item in session_items:
                    await issue(item, time.perf_counter() - origin)

            await asyncio.gather(*(run_session(value) for value in by_session.values()))
        else:
            if args.arrival_trace:
                trace = [
                    ArrivalTraceItem.model_validate(row)
                    for row in read_jsonl(args.arrival_trace)
                ]
                by_id = {item.request_id: item for item in items}
                trace_ids = {value.request_id for value in trace}
                if trace_ids != set(by_id):
                    raise ValueError(
                        "arrival trace request IDs must exactly match workload"
                    )
                schedule = [
                    (value.offset_s, by_id[value.request_id]) for value in trace
                ]
            else:
                rng = random.Random(args.seed)
                schedule = []
                delay = 0.0
                for item in sorted(
                    items, key=lambda value: (value.session_id, value.turn_id)
                ):
                    delay += rng.expovariate(args.request_rate)
                    schedule.append((delay, item))
            previous: dict[str, asyncio.Task[None]] = {}

            async def scheduled(
                delay_s: float,
                item: WorkloadItem,
                predecessor: asyncio.Task[None] | None,
            ) -> None:
                await asyncio.sleep(max(0.0, origin + delay_s - time.perf_counter()))
                if predecessor:
                    await predecessor
                await issue(item, delay_s)

            tasks: list[asyncio.Task[None]] = []
            for delay_s, item in schedule:
                task = asyncio.create_task(
                    scheduled(delay_s, item, previous.get(item.session_id))
                )
                previous[item.session_id] = task
                tasks.append(task)
            await asyncio.gather(*tasks)

    results.sort(key=lambda value: (value.started_at_s, value.request_id))
    results_path = output_dir / "requests.jsonl"
    write_jsonl(results_path, (result.model_dump() for result in results))
    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        project_commit=project_commit(Path(__file__).resolve().parents[1]),
        mode=args.mode,
        endpoint=config["endpoint"],
        model=config["model"],
        workload_path=str(args.workload.resolve()),
        workload_sha256=sha256_file(args.workload),
        workload_count=len(items),
        max_concurrency=args.concurrency,
        request_rate=args.request_rate if args.mode == "poisson" else None,
        seed=args.seed,
        arrival_trace_path=(
            str(args.arrival_trace.resolve()) if args.arrival_trace else None
        ),
        arrival_trace_sha256=(
            sha256_file(args.arrival_trace) if args.arrival_trace else None
        ),
        simulated=args.simulated,
        config=config,
    )
    write_json(output_dir / "run_manifest.json", manifest.model_dump())
    summarize(results_path, output_dir, args.simulated)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/workload-aware-kv-cache-data/runs"),
    )
    parser.add_argument(
        "--mode", choices=["closed_loop", "poisson"], default="closed_loop"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--route-policy", default="direct")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-trace", type=Path)
    parser.add_argument("--simulated", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.request_rate <= 0:
        parser.error("concurrency and request-rate must be positive")
    print(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
