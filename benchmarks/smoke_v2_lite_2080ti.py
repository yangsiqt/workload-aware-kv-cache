from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from benchmarks.io_utils import read_jsonl, write_json


def _prompt(model_path: Path, target_tokens: int) -> tuple[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    source = "def workload_aware_cache(request):\n    return request.session_id\n"
    text = source * max(1, target_tokens // 12)
    token_ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(token_ids), len(token_ids)


def run(model_path: Path, log_root: Path, target_tokens: int) -> dict:
    prompt, prompt_tokens = _prompt(model_path, target_tokens)
    prefix_hash = hashlib.sha256(prompt.encode()).hexdigest()
    run_id = str(int(time.time() * 1000))
    results = []
    for index in range(5):
        request_id = f"v2-lite-smoke-{run_id}-{index}"
        if index:
            response = requests.post(
                "http://127.0.0.1:8000/reset_prefix_cache?reset_external=false",
                timeout=10,
            )
            response.raise_for_status()
            time.sleep(2)
        started = time.perf_counter()
        response = requests.post(
            "http://127.0.0.1:9003/v1/completions",
            headers={
                "X-Request-Id": request_id,
                "X-Session-ID": "v2-lite-smoke-session",
                "X-Prefix-Hash": prefix_hash,
                "X-Prompt-Tokens": str(prompt_tokens),
                "X-Shared-Prefix-Tokens": str(prompt_tokens),
                "X-Expected-Output-Tokens": "8",
            },
            json={
                "model": "Qwen3-0.6B",
                "prompt": prompt,
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=180,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()
        results.append(
            {
                "request_id": request_id,
                "elapsed_ms": elapsed_ms,
                "backend_id": response.headers.get("X-Backend-ID", ""),
                "selected_kv_path": response.headers.get("X-KV-Path", ""),
                "route_reason": response.headers.get("X-Route-Reason", ""),
            }
        )
        time.sleep(2)

    metrics = requests.get("http://127.0.0.1:8000/metrics", timeout=10).text
    required_metrics = {
        "vllm:waiting_prefill_tokens",
        "vllm:running_prefill_tokens",
        "vllm:active_decode_sequences",
    }
    missing_metrics = sorted(name for name in required_metrics if name not in metrics)
    if missing_metrics:
        raise RuntimeError(f"missing V2 scheduler metrics: {missing_metrics}")

    actual_path = log_root / "serving/backend.connector-actual-trace.jsonl"
    actual_rows = list(read_jsonl(actual_path)) if actual_path.exists() else []
    identities_ok = all(
        row.get("request_id")
        and row.get("attempt_id") != ""
        and row.get("backend_id")
        and row.get("selected_path")
        for row in actual_rows
        if row.get("event_type") == "kv_execution_feedback"
    )
    report = {
        "schema_version": "1.0",
        "scope": "FUNCTIONAL_SMOKE_NOT_PERFORMANCE",
        "run_id": run_id,
        "model_path": str(model_path),
        "prompt_tokens": prompt_tokens,
        "requests": results,
        "scheduler_metrics_present": True,
        "actual_feedback_rows": len(actual_rows),
        "actual_feedback_identity_complete": identities_ok,
        "actual_kv_paths": sorted(
            {
                str(row.get("actual_kv_path"))
                for row in actual_rows
                if row.get("actual_kv_path")
            }
        ),
    }
    write_json(log_root / "smoke-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2080 Ti V2 Lite smoke")
    parser.add_argument(
        "--model-path", type=Path, default=Path("/root/autodl-fs/models/Qwen3-0.6B")
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/root/log/workload-aware-kv-cache/v2-lite-2080ti"),
    )
    parser.add_argument("--target-tokens", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(run(args.model_path, args.log_root, args.target_tokens), indent=2))


if __name__ == "__main__":
    main()
