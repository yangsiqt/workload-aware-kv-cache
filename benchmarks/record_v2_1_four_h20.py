from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from benchmarks.io_utils import sha256_file


DETAIL = Path("/root/V2.1四卡项目记录.md")
SUMMARIES = (Path("/root/四卡项目记录.md"), Path("/root/项目记录.md"))
REPOSITORIES = {
    "main": Path("/root/workload-aware-kv-cache"),
    "router": Path("/root/production-stack"),
    "lmcache": Path("/root/LMCache"),
    "vllm": Path("/root/vllm"),
    "mooncake": Path("/root/Mooncake"),
}


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return next(csv.DictReader(handle), None)


def _append_once(path: Path, marker: str, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if not existing:
            handle.write("# V2.1 四卡项目记录\n\n")
            handle.write(
                "> 本文只记录4×H20 V2.1真实实验。正式K04/K05完成前，"
                "不得宣称性能正收益。\n\n"
            )
        handle.write(text)


def record(
    *,
    stage: str,
    tag: str,
    status: str,
    run_dirs: list[Path],
    artifacts: list[Path],
    note: str,
) -> None:
    now = datetime.now(UTC)
    beijing = now.astimezone(timezone(timedelta(hours=8)))
    marker = f"<!-- V21:{tag}:{stage}:{status} -->"
    commits = {name: _git_commit(path) for name, path in REPOSITORIES.items()}
    lines = [
        marker,
        f"## {now.strftime('%Y-%m-%d %H:%M:%S')} UTC：{stage}（{status}）",
        "",
        f"- run tag：`{tag}`。",
        f"- 北京时间：`{beijing.strftime('%Y-%m-%d %H:%M:%S')}`。",
        "- Commit："
        + "；".join(f"{name}=`{value}`" for name, value in commits.items())
        + "。",
    ]
    if note:
        lines.append(f"- 说明：{note}")
    for run_dir in run_dirs:
        validation = _json(run_dir / "validation.json")
        summary = _summary(run_dir / "summary.csv")
        lines.append(f"- Run：`{run_dir}`。")
        if validation:
            lines.append(
                "  - 门禁："
                f"passed={validation.get('passed')}，"
                f"success={validation.get('successful_requests')}/"
                f"{validation.get('expected_requests')}，"
                f"join={validation.get('joined_trace_rows')}，"
                f"selected={validation.get('selected_kv_paths')}，"
                f"actual={validation.get('actual_kv_paths')}，"
                f"mismatch={len(validation.get('path_mismatches') or [])}。"
            )
        if summary:
            lines.append(
                "  - 指标："
                f"request/s={summary.get('request_per_s')}，"
                f"TTFT p90={summary.get('ttft_ms_p90')} ms，"
                f"E2E p90={summary.get('e2e_ms_p90')} ms，"
                f"SLO Goodput={summary.get('slo_goodput_request_per_s')} req/s。"
            )
    for artifact in artifacts:
        payload = _json(artifact)
        lines.append(
            f"- Artifact：`{artifact}`，SHA256=`{sha256_file(artifact) if artifact.is_file() else 'missing'}`。"
        )
        if payload and "selection" in payload:
            lines.append(
                f"  - RPS选择：`{json.dumps(payload['selection'], ensure_ascii=False, sort_keys=True)}`。"
            )
        if payload and "measured" in payload:
            lines.append(
                f"  - 冻结成本：`{json.dumps(payload['measured'], ensure_ascii=False, sort_keys=True)}`。"
            )
    lines.extend(
        [
            f"- 是否允许继续：`{'YES' if status == 'PASS' else 'NO'}`。",
            "- 结论边界：本节点本身不代表V2.1四卡整体性能正收益。",
            "",
        ]
    )
    _append_once(DETAIL, marker, "\n".join(lines))

    compact = (
        f"\n{marker}\n- V2.1四卡 `{stage}`：`{status}`，run tag=`{tag}`，"
        f"详细证据见`/root/V2.1四卡项目记录.md`。"
        f"{' ' + note if note else ''}\n"
    )
    for path in SUMMARIES:
        _append_once(path, marker, compact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--status", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    record(
        stage=args.stage,
        tag=args.tag,
        status=args.status,
        run_dirs=args.run_dir,
        artifacts=args.artifact,
        note=args.note,
    )


if __name__ == "__main__":
    main()
