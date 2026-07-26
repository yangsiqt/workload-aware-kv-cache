from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.record_v2_2_stage import append_once

DETAIL = Path("/root/V2.3项目记录.md")
SUMMARIES = (Path("/root/四卡项目记录.md"), Path("/root/项目记录.md"))


def record(tag: str, stage: str, status: str, detail: str) -> int:
    timestamp = datetime.now(UTC).isoformat()
    marker = f"<!-- v2.3-stage:{tag}:{stage}:{status} -->"
    body = (
        f"### {timestamp} — V2.3 {stage} {status}\n\n"
        f"- Run tag：`{tag}`\n"
        f"- 结果：{detail}\n"
        "- 性能结论：至少两条独立Trace完成公平配对后才允许生成。"
    )
    changed = int(append_once(DETAIL, marker, body))
    summary = (
        f"- {timestamp}：V2.3 `{tag}` 的 `{stage}` 为 **{status}**；{detail}。"
        "详细证据见`/root/V2.3项目记录.md`。"
    )
    for path in SUMMARIES:
        changed += int(append_once(path, marker, summary))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one V2.3 experiment stage")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--detail", required=True)
    args = parser.parse_args()
    print(record(args.tag, args.stage, args.status, args.detail))


if __name__ == "__main__":
    main()
