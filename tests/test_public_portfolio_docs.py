from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "README_ZH.md",
    ROOT / "reports/four-h20/v2.3-primary-trace.md",
)

PRIMARY_HEADLINE_VALUES = (
    "1567.3",
    "553.6",
    "9631.9",
    "5414.1",
    "16543.8",
    "13190.5",
    "2.5014",
    "2.5013",
    "1.5008",
    "2.0302",
    "35.3%",
)

REPLICATE_PERFORMANCE_VALUES = (
    "12.8%",
    "5.9%",
    "7.8%",
    "29603.5",
)


def test_public_readmes_share_primary_trace_headlines() -> None:
    english = (ROOT / "README.md").read_text()
    chinese = (ROOT / "README_ZH.md").read_text()

    for value in PRIMARY_HEADLINE_VALUES:
        assert value in english
        assert value in chinese


def test_public_docs_do_not_publish_replicate_performance_values() -> None:
    public_text = "\n".join(path.read_text() for path in PUBLIC_DOCS)
    for value in REPLICATE_PERFORMANCE_VALUES:
        assert value not in public_text


def test_public_markdown_relative_links_exist() -> None:
    markdown_link = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in PUBLIC_DOCS:
        for target in markdown_link.findall(document.read_text()):
            if target.startswith(("http://", "https://", "#")):
                continue
            resolved = (document.parent / target).resolve()
            assert resolved.exists(), f"{document}: missing link target {target}"


def test_component_lock_pins_window1_v2_3_commits() -> None:
    lock = yaml.safe_load((ROOT / "components.lock.yaml").read_text())
    assert lock["scope"] == "window1-adaptive-kv-v2.3"
    assert {
        name: value["commit"]
        for name, value in lock["components"].items()
    } == {
        "production_stack": "eca26980bb6666e28980e02aa0da2cfe4f7fe610",
        "vllm": "48658ab048227caf3b952ad96d1a6221783f644a",
        "lmcache": "78edee49b24db7846261b97bb43abb10c7fecd6c",
        "mooncake": "e198f435df1ac4149d22e6e2dc6136dc86810e3a",
    }
