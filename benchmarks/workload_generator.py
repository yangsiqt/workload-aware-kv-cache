from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import ijson

from benchmarks.io_utils import (
    load_yaml,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from benchmarks.profile_utils import profile_instance_ids
from benchmarks.repo_context import GitRepoCache
from benchmarks.sampling import stratified_sample
from benchmarks.schemas import ChatMessage, SourceInfo, WorkloadItem
from benchmarks.tokenizer_utils import (
    chat_tokens,
    fit_system_content,
    load_tokenizer,
    token_ids_sha256,
    tokenizer_fingerprint,
)

CODE_AGENT_SYSTEM = """You are a code-analysis agent working on a fixed repository snapshot.
Use only the issue and repository context below. Do not claim to have edited files.
Available conceptual tools: list_files, read_file, search_code, inspect_symbol, run_tests.
The benchmark evaluates serving behavior, not patch correctness.

Repository: {repo}
Base commit: {commit}
Issue:
{problem}

Repository context:
{context}
"""

AGENT_STAGES = [
    "Identify the most relevant files and symbols. Explain what should be inspected first.",
    "Trace the important call path and state transitions related to the issue.",
    "Form a concrete root-cause hypothesis and cite the provided code context.",
    "Describe a minimal patch strategy without writing the final patch.",
    "Design focused regression tests and identify likely edge cases.",
    "Review the proposed approach for compatibility, performance, and rollback risks.",
    "Describe instrumentation that would confirm the diagnosis in production.",
    "Summarize the implementation plan and the evidence still missing.",
]

CHECKPOINT = "Checkpoint stored. Continue from the fixed repository context and prior analysis request."
DEFAULT_EXPECTED_OUTPUT_TOKENS = [128, 128, 128, 128, 128, 512]


def _priority(index: int) -> int:
    bucket = index % 10
    return 0 if bucket == 0 else (2 if bucket >= 8 else 1)


def _source_revision(manifest: dict[str, Any], name: str, fallback: str) -> str:
    artifact = manifest.get("artifacts", {}).get(name)
    if isinstance(artifact, dict):
        return str(artifact.get("revision") or artifact.get("sha256") or fallback)
    if isinstance(artifact, list) and artifact:
        return str(artifact[0].get("revision") or artifact[0].get("sha256") or fallback)
    return fallback


def _build_swebench(
    rows: list[dict[str, Any]],
    *,
    count: int,
    turns: int,
    prefix_tiers: list[int],
    seed: int,
    revision: str,
    snapshot_id: str,
    tokenizer: Any,
    repo_cache: GitRepoCache,
    preselected: bool = False,
    expected_output_tokens: list[int] | None = None,
) -> Iterator[WorkloadItem]:
    selected = rows if preselected else stratified_sample(rows, count, seed)
    output_tokens = expected_output_tokens or DEFAULT_EXPECTED_OUTPUT_TOKENS
    if turns > len(output_tokens):
        raise ValueError("expected_output_tokens must cover every agent turn")
    for index, row in enumerate(selected):
        instance_id = str(row["instance_id"])
        repo = str(row["repo"])
        commit = str(row["base_commit"])
        target = prefix_tiers[index % len(prefix_tiers)]
        context = repo_cache.build_context(
            repo,
            commit,
            str(row.get("patch", "")),
            target_chars=target * 7,
        )
        raw_system = CODE_AGENT_SYSTEM.format(
            repo=repo,
            commit=commit,
            problem=str(row["problem_statement"]),
            context=context,
        )
        system_text, prefix_ids = fit_system_content(tokenizer, raw_system, target)
        prefix_hash = token_ids_sha256(prefix_ids)
        history: list[dict[str, str]] = [{"role": "system", "content": system_text}]
        session_id = f"swebench-{instance_id}"
        for turn_id, stage in enumerate(AGENT_STAGES[:turns]):
            messages = history + [{"role": "user", "content": stage}]
            prompt_tokens = len(
                chat_tokens(tokenizer, messages, add_generation_prompt=True)
            )
            yield WorkloadItem(
                dataset_name="SWE-bench Verified",
                dataset_revision=revision,
                dataset_instance_id=instance_id,
                request_id=f"{session_id}-t{turn_id:02d}",
                session_id=session_id,
                turn_id=turn_id,
                priority=_priority(index),
                request_type="code_agent_multiturn",
                prefix_hash=prefix_hash,
                messages=[ChatMessage(**message) for message in messages],
                prompt_tokens=prompt_tokens,
                shared_prefix_tokens=len(prefix_ids),
                expected_output_tokens=output_tokens[turn_id],
                source=SourceInfo(
                    dataset="SWE-bench/SWE-bench_Verified",
                    license="SWE-bench terms plus source repository license",
                    snapshot_id=snapshot_id,
                    repo=repo,
                    base_commit=commit,
                    public_id=instance_id,
                    url=f"https://github.com/{repo}",
                ),
            )
            history.extend(
                [
                    {"role": "user", "content": stage},
                    {"role": "assistant", "content": CHECKPOINT},
                ]
            )


def _build_longbench(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    revision: str,
    snapshot_id: str,
    tokenizer: Any,
) -> Iterator[WorkloadItem]:
    for index, row in enumerate(rows):
        if index >= count:
            break
        context = str(row["context"])
        system = "Repository-level code context follows.\n\n" + context
        prefix_messages = [{"role": "system", "content": system}]
        prefix_ids = chat_tokens(
            tokenizer, prefix_messages, add_generation_prompt=False
        )
        messages = prefix_messages + [{"role": "user", "content": str(row["input"])}]
        prompt_tokens = len(
            chat_tokens(tokenizer, messages, add_generation_prompt=True)
        )
        public_id = str(row.get("_id", index))
        yield WorkloadItem(
            dataset_name="LongBench RepoBench-P",
            dataset_revision=revision,
            dataset_instance_id=public_id,
            request_id=f"longbench-{public_id}",
            session_id=f"longbench-{public_id}",
            turn_id=0,
            priority=_priority(index),
            request_type="repo_code_completion",
            prefix_hash=token_ids_sha256(prefix_ids),
            messages=[ChatMessage(**message) for message in messages],
            prompt_tokens=prompt_tokens,
            shared_prefix_tokens=len(prefix_ids),
            expected_output_tokens=128,
            source=SourceInfo(
                dataset="THUDM/LongBench:repobench-p",
                license="LongBench and RepoBench dataset terms",
                snapshot_id=snapshot_id,
                public_id=public_id,
                url="https://github.com/THUDM/LongBench",
            ),
        )


def _normalize_sharegpt_turn(turn: dict[str, Any]) -> dict[str, str] | None:
    role = str(turn.get("from", turn.get("role", ""))).lower()
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
    }
    mapped = role_map.get(role)
    content = turn.get("value", turn.get("content"))
    if not mapped or not isinstance(content, str) or not content.strip():
        return None
    return {"role": mapped, "content": content}


def _build_sharegpt(
    conversations: Iterable[dict[str, Any]],
    *,
    count: int,
    revision: str,
    snapshot_id: str,
    tokenizer: Any,
) -> Iterator[WorkloadItem]:
    emitted = 0
    for row in conversations:
        turns = [
            item
            for turn in row.get("conversations", [])
            if (item := _normalize_sharegpt_turn(turn)) is not None
        ]
        user_positions = [i for i, turn in enumerate(turns) if turn["role"] == "user"]
        if not user_positions:
            continue
        stop = user_positions[min(1, len(user_positions) - 1)]
        messages = turns[: stop + 1]
        if not messages or messages[-1]["role"] != "user":
            continue
        index = emitted
        prompt_tokens = len(
            chat_tokens(tokenizer, messages, add_generation_prompt=True)
        )
        prefix = messages[:-1]
        prefix_ids = (
            chat_tokens(tokenizer, prefix, add_generation_prompt=False)
            if prefix
            else []
        )
        public_id = str(row.get("id", index))
        yield WorkloadItem(
            dataset_name="ShareGPT",
            dataset_revision=revision,
            dataset_instance_id=public_id,
            request_id=f"sharegpt-{index:05d}",
            session_id=f"sharegpt-{index:05d}",
            turn_id=0,
            priority=_priority(index),
            request_type="standard_chat",
            prefix_hash=token_ids_sha256(prefix_ids),
            messages=[ChatMessage(**message) for message in messages],
            prompt_tokens=prompt_tokens,
            shared_prefix_tokens=len(prefix_ids),
            expected_output_tokens=128,
            source=SourceInfo(
                dataset="ShareGPT_V3_unfiltered_cleaned_split",
                license="ShareGPT source dataset terms",
                snapshot_id=snapshot_id,
                public_id=public_id,
                url="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered",
            ),
        )
        emitted += 1
        if emitted >= count:
            break


def _build_controlled(
    *,
    count: int,
    turns: int,
    prefix_tiers: list[int],
    tokenizer: Any,
) -> Iterator[WorkloadItem]:
    for index in range(count):
        target = prefix_tiers[index % len(prefix_tiers)]
        line = f"# controlled repository session {index:04d}\nclass CacheProbe{index}: pass\n"
        raw = "Controlled code prefix.\n" + line * (target // 8 + 32)
        system, prefix_ids = fit_system_content(tokenizer, raw, target)
        prefix_hash = token_ids_sha256(prefix_ids)
        session_id = f"controlled-{index:04d}"
        history: list[dict[str, str]] = [{"role": "system", "content": system}]
        for turn_id in range(turns):
            question = f"Inspect controlled symbol {index} at analysis turn {turn_id}."
            messages = history + [{"role": "user", "content": question}]
            yield WorkloadItem(
                dataset_name="Controlled Prefix",
                dataset_revision="generated-v1",
                dataset_instance_id=session_id,
                request_id=f"{session_id}-t{turn_id:02d}",
                session_id=session_id,
                turn_id=turn_id,
                priority=_priority(index),
                request_type="controlled_shared_prefix",
                prefix_hash=prefix_hash,
                messages=[ChatMessage(**message) for message in messages],
                prompt_tokens=len(
                    chat_tokens(tokenizer, messages, add_generation_prompt=True)
                ),
                shared_prefix_tokens=len(prefix_ids),
                expected_output_tokens=32,
                source=SourceInfo(
                    dataset="controlled-prefix-v1",
                    license="generated",
                    snapshot_id="seed-42",
                    public_id=session_id,
                ),
            )
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": CHECKPOINT},
                ]
            )


def generate(
    config_path: Path, profile_name: str, selected: set[str]
) -> dict[str, Any]:
    config = load_yaml(config_path)
    profile = config["profiles"][profile_name]
    data_root = Path(config["data_root"])
    raw_dir = data_root / "raw"
    output_dir = data_root / "processed" / profile_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config["tokenizer_path"])
    dataset_manifest_path = data_root / "manifests" / "datasets.json"
    dataset_manifest = (
        json.loads(dataset_manifest_path.read_text())
        if dataset_manifest_path.exists()
        else {}
    )
    snapshot_id = (
        sha256_file(dataset_manifest_path)
        if dataset_manifest_path.exists()
        else "unmanifested"
    )
    repo_cache = GitRepoCache(data_root / "repo-cache")
    artifact_paths: list[Path] = []
    artifacts: dict[str, Any] = {}

    def materialize(name: str, items: Iterable[WorkloadItem]) -> None:
        path = output_dir / f"{name}.jsonl"
        count = write_jsonl(path, items)
        artifact_paths.append(path)
        artifacts[name] = {
            "path": str(path),
            "rows": count,
            "sha256": sha256_file(path),
        }

    if "swebench" in selected:
        path = raw_dir / "swebench_verified.jsonl"
        swebench_rows = list(read_jsonl(path))
        instance_ids = profile_instance_ids(profile, config_path)
        if instance_ids:
            if len(instance_ids) != int(profile["swebench_sessions"]):
                raise ValueError("swebench_instance_ids must match swebench_sessions")
            by_id = {str(row["instance_id"]): row for row in swebench_rows}
            missing = [
                instance_id for instance_id in instance_ids if instance_id not in by_id
            ]
            if missing:
                raise ValueError(f"Unknown SWE-bench instance IDs: {missing}")
            swebench_rows = [by_id[instance_id] for instance_id in instance_ids]
        if profile.get("require_prefetched_snapshots", False):
            snapshot_root = data_root / "repo-cache" / "snapshots"
            swebench_rows = [
                row
                for row in swebench_rows
                if (
                    snapshot_root
                    / str(row["repo"]).replace("/", "__")
                    / f"{row['base_commit']}.tar.gz"
                ).exists()
            ]
            if len(swebench_rows) < int(profile["swebench_sessions"]):
                raise ValueError(
                    f"Only {len(swebench_rows)} prefetched SWE-bench snapshots are available"
                )
        items = _build_swebench(
            swebench_rows,
            count=int(profile["swebench_sessions"]),
            turns=int(profile["turns_per_swebench_session"]),
            prefix_tiers=[int(value) for value in profile["prefix_tiers"]],
            seed=int(config["seed"]),
            revision=_source_revision(dataset_manifest, "swebench", "unknown"),
            snapshot_id=snapshot_id,
            tokenizer=tokenizer,
            repo_cache=repo_cache,
            preselected=bool(instance_ids),
            expected_output_tokens=[
                int(value)
                for value in profile.get(
                    "expected_output_tokens", DEFAULT_EXPECTED_OUTPUT_TOKENS
                )
            ],
        )
        materialize("swebench", items)

    if "longbench" in selected:
        path = raw_dir / "longbench_repobench-p.jsonl"
        items = _build_longbench(
            read_jsonl(path),
            count=int(profile["longbench_samples"]),
            revision=_source_revision(dataset_manifest, "longbench", "unknown"),
            snapshot_id=snapshot_id,
            tokenizer=tokenizer,
        )
        materialize("longbench", items)

    if "sharegpt" in selected:
        path = raw_dir / "sharegpt.json"
        with path.open("rb") as handle:
            items = _build_sharegpt(
                ijson.items(handle, "item"),
                count=int(profile["sharegpt_sessions"]),
                revision=_source_revision(
                    dataset_manifest, "sharegpt", sha256_file(path)
                ),
                snapshot_id=snapshot_id,
                tokenizer=tokenizer,
            )
            materialize("sharegpt", items)

    if "controlled" in selected:
        items = _build_controlled(
            count=int(profile["controlled_sessions"]),
            turns=int(profile["turns_per_swebench_session"]),
            prefix_tiers=[int(value) for value in profile["prefix_tiers"]],
            tokenizer=tokenizer,
        )
        materialize("controlled", items)

    combined_path = output_dir / "combined.jsonl"
    combined_rows = 0
    with combined_path.open("w", encoding="utf-8") as combined_handle:
        for path in artifact_paths:
            with path.open(encoding="utf-8") as artifact_handle:
                for line in artifact_handle:
                    combined_handle.write(line)
                    combined_rows += 1
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile_name,
        "seed": config["seed"],
        "tokenizer_path": config["tokenizer_path"],
        "tokenizer_fingerprint": tokenizer_fingerprint(config["tokenizer_path"]),
        "combined": {
            "path": str(combined_path),
            "rows": combined_rows,
            "sha256": sha256_file(combined_path),
        },
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible serving workloads"
    )
    parser.add_argument("--config", default="configs/workloads.yaml")
    parser.add_argument(
        "--profile", default="small", help="Profile name from configs/workloads.yaml"
    )
    parser.add_argument(
        "--datasets",
        default="swebench,longbench,sharegpt,controlled",
        help="Comma-separated: swebench,longbench,sharegpt,controlled",
    )
    args = parser.parse_args()
    available = load_yaml(Path(args.config))["profiles"]
    if args.profile not in available:
        parser.error(
            f"unknown profile {args.profile!r}; choose from {', '.join(available)}"
        )
    selected = {name.strip() for name in args.datasets.split(",") if name.strip()}
    manifest = generate(Path(args.config), args.profile, selected)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
