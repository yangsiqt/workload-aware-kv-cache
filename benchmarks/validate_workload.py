from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.io_utils import read_jsonl, sha256_file
from benchmarks.schemas import WorkloadItem
from benchmarks.tokenizer_utils import chat_tokens, load_tokenizer, token_ids_sha256


def validate(
    workload_path: Path, tokenizer_path: Path, raw_swebench: Path | None = None
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    errors: list[str] = []
    sessions: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    datasets: set[str] = set()
    prefix_token_lengths: set[int] = set()
    request_count = 0
    patches: dict[str, str] = {}
    if raw_swebench and raw_swebench.exists():
        patches = {
            str(row["instance_id"]): str(row.get("patch", ""))
            for row in read_jsonl(raw_swebench)
        }

    for raw_row in read_jsonl(workload_path):
        row = WorkloadItem.model_validate(raw_row)
        request_count += 1
        datasets.add(row.dataset_name)
        prefix_token_lengths.add(row.shared_prefix_tokens)
        sessions[row.session_id].append(
            (row.turn_id, row.prefix_hash, len(row.messages))
        )
        messages = [message.model_dump() for message in row.messages]
        actual_prompt = len(
            chat_tokens(tokenizer, messages, add_generation_prompt=True)
        )
        if actual_prompt != row.prompt_tokens:
            errors.append(
                f"{row.request_id}: prompt token mismatch {actual_prompt} != {row.prompt_tokens}"
            )
        system = [messages[0]] if messages and messages[0]["role"] == "system" else []
        if system:
            prefix_ids = chat_tokens(tokenizer, system, add_generation_prompt=False)
            if len(prefix_ids) != row.shared_prefix_tokens:
                errors.append(f"{row.request_id}: shared prefix token mismatch")
            if token_ids_sha256(prefix_ids) != row.prefix_hash:
                errors.append(f"{row.request_id}: prefix hash mismatch")

        if row.dataset_name == "SWE-bench Verified":
            prompt = "\n".join(message.content for message in row.messages)
            patch = patches.get(row.dataset_instance_id, "")
            if patch and patch in prompt:
                errors.append(f"{row.request_id}: full gold patch leaked into prompt")

    for session_id, values in sessions.items():
        values.sort(key=lambda item: item[0])
        turns = [item[0] for item in values]
        if turns != list(range(len(values))):
            errors.append(f"{session_id}: non-contiguous turns {turns}")
        if len({item[1] for item in values}) != 1:
            errors.append(f"{session_id}: unstable prefix hash")
        message_counts = [item[2] for item in values]
        if message_counts != sorted(message_counts) or len(message_counts) != len(
            set(message_counts)
        ):
            errors.append(f"{session_id}: history is not strictly growing")

    report = {
        "valid": not errors,
        "path": str(workload_path),
        "sha256": sha256_file(workload_path),
        "requests": request_count,
        "sessions": len(sessions),
        "datasets": sorted(datasets),
        "prefix_token_lengths": sorted(prefix_token_lengths),
        "errors": errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--raw-swebench", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.workload, args.tokenizer, args.raw_swebench)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
