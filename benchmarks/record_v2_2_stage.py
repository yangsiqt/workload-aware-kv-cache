from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path


DETAIL = Path("/root/V2.2项目记录.md")
SUMMARIES = (Path("/root/四卡项目记录.md"), Path("/root/项目记录.md"))


def append_once(path: Path, marker: str, text: str) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write(f"\n{marker}\n{text.rstrip()}\n")
    return True


def record(tag: str, stage: str, status: str, detail: str) -> int:
    timestamp = datetime.now(UTC).isoformat()
    marker = f"<!-- v2.2-stage:{tag}:{stage}:{status} -->"
    body = (
        f"### {timestamp} — V2.2 {stage} {status}\n\n"
        f"- Run tag：`{tag}`\n"
        f"- 结果：{detail}\n"
        "- 性能结论：仅K04/K05公平配对及K07门禁通过后允许生成。"
    )
    changed = int(append_once(DETAIL, marker, body))
    summary = (
        f"- {timestamp}：V2.2 `{tag}` 的 `{stage}` 为 **{status}**；{detail}。"
        "详细证据见`/root/V2.2项目记录.md`。"
    )
    for path in SUMMARIES:
        changed += int(append_once(path, marker, summary))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one V2.2 experiment stage")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--detail", required=True)
    args = parser.parse_args()
    print(record(args.tag, args.stage, args.status, args.detail))


if __name__ == "__main__":
    main()
