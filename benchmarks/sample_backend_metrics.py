from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from prometheus_client.parser import text_string_to_metric_families


METRICS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:prefix_cache_queries": "prefix_queries",
    "vllm:prefix_cache_queries_total": "prefix_queries",
    "vllm:prefix_cache_hits": "prefix_hits",
    "vllm:prefix_cache_hits_total": "prefix_hits",
    "vllm:kv_cache_usage_perc": "kv_usage",
}


def parse_metrics(text: str) -> dict[str, float]:
    values = {name: 0.0 for name in set(METRICS.values())}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in METRICS:
                values[METRICS[sample.name]] = float(sample.value)
    return values


def scrape(backend_id: str, url: str) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "monotonic_s": time.monotonic(),
        "backend_id": backend_id,
        "url": url,
    }
    try:
        with urlopen(f"{url.rstrip('/')}/metrics", timeout=0.2) as response:
            row.update(parse_metrics(response.read().decode("utf-8")))
        row["error"] = None
    except Exception as exc:  # Metrics failure belongs in the experiment trace.
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    endpoints = []
    for value in args.backend:
        backend_id, separator, url = value.partition("=")
        if not separator:
            parser.error("--backend must be ID=URL")
        endpoints.append((backend_id, url))
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
            for backend_id, url in endpoints:
                handle.write(json.dumps(scrape(backend_id, url), sort_keys=True) + "\n")
            handle.flush()
            if args.once or stop:
                break
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
