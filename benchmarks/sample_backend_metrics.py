from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from prometheus_client.parser import text_string_to_metric_families


VLLM_METRICS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:prefix_cache_queries": "prefix_queries",
    "vllm:prefix_cache_queries_total": "prefix_queries",
    "vllm:prefix_cache_hits": "prefix_hits",
    "vllm:prefix_cache_hits_total": "prefix_hits",
    "vllm:kv_cache_usage_perc": "kv_usage",
    "vllm:waiting_prefill_tokens": "waiting_prefill_tokens",
    "vllm:running_prefill_tokens": "running_prefill_tokens",
    "vllm:active_decode_sequences": "active_decode_sequences",
    "vllm:scheduled_prefill_tokens": "scheduled_prefill_tokens",
    "vllm:scheduled_decode_tokens": "scheduled_decode_tokens",
    "vllm:skipped_waiting_prefill_tokens": "skipped_waiting_prefill_tokens",
    "vllm:kv_cache_free_blocks": "kv_cache_free_blocks",
    "vllm:kv_cache_total_blocks": "kv_cache_total_blocks",
    "vllm:num_preemptions": "preemptions_total",
    "vllm:num_preemptions_total": "preemptions_total",
}

MOONCAKE_METRICS = {
    "mooncake_transfer_read_bytes": "read_bytes_total",
    "mooncake_transfer_read_operation_count": "read_operations_total",
    "mooncake_transfer_inflight_read_operations": "inflight_read_operations",
    "mooncake_transfer_inflight_read_bytes": "inflight_read_bytes",
    "mooncake_transfer_read_failures": "read_failures_total",
    "mooncake_transfer_read_misses": "read_misses_total",
}


def parse_metrics(
    text: str, metric_map: dict[str, str] | None = None
) -> dict[str, float]:
    selected = metric_map or VLLM_METRICS
    values = {name: 0.0 for name in set(selected.values())}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in selected:
                values[selected[sample.name]] = float(sample.value)
    return values


def scrape(
    backend_id: str,
    url: str,
    *,
    source: str = "vllm",
) -> dict[str, object]:
    metric_map = VLLM_METRICS if source == "vllm" else MOONCAKE_METRICS
    row: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "monotonic_s": time.monotonic(),
        "backend_id": backend_id,
        "url": url,
        "source": source,
    }
    try:
        with urlopen(f"{url.rstrip('/')}/metrics", timeout=0.2) as response:
            row.update(parse_metrics(response.read().decode("utf-8"), metric_map))
        row["error"] = None
    except Exception as exc:  # Metrics failure belongs in the experiment trace.
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--mooncake", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    endpoints = []
    for source, values in (("vllm", args.backend), ("mooncake", args.mooncake)):
        for value in values:
            backend_id, separator, url = value.partition("=")
            if not separator:
                parser.error(f"--{source} must be ID=URL")
            endpoints.append((backend_id, url, source))
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        while True:
            started = time.monotonic()
            for backend_id, url, source in endpoints:
                handle.write(
                    json.dumps(scrape(backend_id, url, source=source), sort_keys=True)
                    + "\n"
                )
            handle.flush()
            if args.once or stop:
                break
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
