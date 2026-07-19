from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, sha256_file, write_json, write_jsonl
from benchmarks.schemas import WorkloadItem
from benchmarks.tokenizer_utils import load_tokenizer, tokenizer_fingerprint

CONTROLLED_SESSIONS = ("controlled-0000", "controlled-0001", "controlled-0002")
CHUNKED_SESSIONS = (
    "controlled-0001",
    "controlled-0004",
    "controlled-0007",
    "controlled-0010",
)
SWE_SESSIONS = (
    "swebench-astropy__astropy-13398",
    "swebench-astropy__astropy-14096",
    "swebench-astropy__astropy-12907",
)


def select_session_turns(
    rows: list[WorkloadItem], session_ids: tuple[str, ...], turns: int
) -> list[WorkloadItem]:
    selected = [
        row for row in rows if row.session_id in session_ids and row.turn_id < turns
    ]
    selected.sort(key=lambda row: (session_ids.index(row.session_id), row.turn_id))
    expected = len(session_ids) * turns
    if len(selected) != expected:
        raise ValueError(f"expected {expected} requests, found {len(selected)}")
    return selected


def official_custom_rows(
    rows: list[WorkloadItem], tokenizer: Any
) -> list[dict[str, Any]]:
    return [
        {
            "prompt": tokenizer.apply_chat_template(
                [message.model_dump() for message in row.messages],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "output_tokens": row.expected_output_tokens,
            "request_id": row.request_id,
        }
        for row in rows
    ]


def build(source_root: Path, output_dir: Path, model_path: Path) -> dict[str, Any]:
    pre_rental = source_root / "pre_rental"
    final = source_root / "final"
    small = source_root / "small"
    controlled_rows = [
        WorkloadItem.model_validate(row)
        for row in read_jsonl(pre_rental / "controlled.jsonl")
    ]
    swe_rows = [
        WorkloadItem.model_validate(row) for row in read_jsonl(final / "swebench.jsonl")
    ]
    sharegpt_rows = [
        WorkloadItem.model_validate(row) for row in read_jsonl(small / "sharegpt.jsonl")
    ]
    if len(sharegpt_rows) != 8:
        raise ValueError(f"expected 8 ShareGPT requests, found {len(sharegpt_rows)}")

    outputs: dict[str, list[Any]] = {
        "controlled-8k16k32k.jsonl": select_session_turns(
            controlled_rows, CONTROLLED_SESSIONS, 4
        ),
        "swe-8k16k32k.jsonl": select_session_turns(swe_rows, SWE_SESSIONS, 4),
        "sharegpt-8.jsonl": sharegpt_rows,
        "chunked-16k-c4.jsonl": select_session_turns(
            controlled_rows, CHUNKED_SESSIONS, 1
        ),
    }
    tokenizer = load_tokenizer(str(model_path))
    outputs["sharegpt-8-vllm-custom.jsonl"] = official_custom_rows(
        sharegpt_rows, tokenizer
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_outputs = {}
    for name, rows in outputs.items():
        path = output_dir / name
        count = write_jsonl(path, rows)
        manifest_outputs[name] = {
            "path": str(path),
            "count": count,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": "1.0",
        "profile": "single-h20-calibration-screening",
        "model_path": str(model_path),
        "tokenizer_fingerprint": tokenizer_fingerprint(str(model_path)),
        "source_files": {
            str(path): sha256_file(path)
            for path in (
                pre_rental / "controlled.jsonl",
                final / "swebench.jsonl",
                small / "sharegpt.jsonl",
            )
        },
        "outputs": manifest_outputs,
        "limitations": [
            "single GPU screening only",
            "no p99 claims",
            "no multi-backend routing conclusions",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/root/workload-aware-kv-cache-data/processed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/workload-aware-kv-cache-data/processed/single_h20_calibration"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            os.environ.get(
                "MODEL_PATH",
                "/root/autodl-fs/models/Qwen3-30B-A3B-Instruct-2507",
            )
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(build(args.source_root, args.output_dir, args.model_path), indent=2)
    )


if __name__ == "__main__":
    main()
