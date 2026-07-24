from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.io_utils import read_jsonl
from benchmarks.schemas import WorkloadItem
from benchmarks.single_h20_v2_1_multitier import (
    _refresh_router_tier,
    _wait_for_layout,
)
from benchmarks.tokenizer_utils import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--expected-path", choices=("lmcache_l1", "mooncake_l2"), required=True
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507"),
    )
    args = parser.parse_args()
    rows = list(read_jsonl(args.profile))
    if len(rows) != 1:
        raise ValueError("Router tier refresh requires a one-request profile")
    item = WorkloadItem.model_validate(rows[0])
    tokenizer = load_tokenizer(str(args.model_path))
    if args.expected_path == "lmcache_l1":
        _wait_for_layout(
            item,
            tokenizer,
            lambda layout: layout.get("LocalCPUBackend", 0) >= 8192
            and layout.get("RemoteBackend", 0) >= 8192,
        )
    else:
        _wait_for_layout(
            item,
            tokenizer,
            lambda layout: layout.get("LocalCPUBackend", 0) == 0
            and layout.get("RemoteBackend", 0) >= 8192,
        )
    _refresh_router_tier(
        item,
        expected_path=args.expected_path,
        run_id=args.run_id,
        trace_path=args.trace,
    )


if __name__ == "__main__":
    main()
