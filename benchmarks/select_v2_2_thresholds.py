from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from benchmarks.io_utils import read_jsonl, sha256_file, write_json


def _contexts(path: Path) -> list[dict[str, Any]]:
    contexts = []
    for row in read_jsonl(path):
        attempts = row.get("attempts") or []
        if attempts:
            context = (attempts[-1].get("decision") or {}).get("v2_context")
            if isinstance(context, dict):
                contexts.append(context)
    return contexts


def select_thresholds(joined_path: Path) -> dict[str, Any]:
    contexts = _contexts(joined_path)
    total = len(contexts)
    overrides = sum(
        context.get("guard_reason") == "v2_override_fixed" for context in contexts
    )
    potential_250 = sum(
        float(context.get("gain_ms", 0.0)) >= 250.0
        and float(context.get("gain_ratio", 0.0)) >= 0.10
        and bool(context.get("telemetry_valid"))
        and bool(context.get("cache_confidence_valid"))
        for context in contexts
    )
    ratio = overrides / total if total else 0.0
    if ratio < 0.05 and total and potential_250 / total >= 0.05:
        gain_ms, gain_ratio, reason = 250.0, 0.10, "under_activation"
    elif ratio > 0.30:
        gain_ms, gain_ratio, reason = 750.0, 0.20, "over_activation"
    else:
        gain_ms, gain_ratio, reason = 500.0, 0.15, "default_accepted"
    return {
        "schema_version": "2.2",
        "calibration_rows": total,
        "default_overrides": overrides,
        "default_override_ratio": ratio,
        "potential_250ms_10pct": potential_250,
        "selected_gain_ms": gain_ms,
        "selected_gain_ratio": gain_ratio,
        "selection_reason": reason,
        "requires_one_recalibration": (gain_ms, gain_ratio) != (500.0, 0.15),
    }


def freeze_config(template: Path, output: Path, report: dict[str, Any]) -> None:
    config = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    config["v2_min_gain_ms"] = report["selected_gain_ms"]
    config["v2_min_gain_ratio"] = report["selected_gain_ratio"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report["template"] = str(template.resolve())
    report["template_sha256"] = sha256_file(template)
    report["frozen_config"] = str(output.resolve())
    report["frozen_config_sha256"] = sha256_file(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze V2.2 guard thresholds")
    parser.add_argument("joined_trace", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = select_thresholds(args.joined_trace)
    freeze_config(args.template, args.output_config, report)
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
