from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from benchmarks.io_utils import read_jsonl, sha256_file, write_json


def _summary(run_dir: Path) -> dict[str, str]:
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def select_kv(
    specs: list[str], output_config: Path, report_path: Path
) -> dict[str, Any]:
    candidates = []
    for spec in specs:
        label, separator, remainder = spec.partition("=")
        config_raw, separator2, run_raw = remainder.partition("=")
        if not separator or not separator2:
            raise ValueError("KV candidate must be LABEL=CONFIG=RUN_DIR")
        run_dir = Path(run_raw)
        summary = _summary(run_dir)
        requests = int(summary["requests"])
        successes = int(summary["successful_requests"])
        candidates.append(
            {
                "label": label,
                "config": str(Path(config_raw).resolve()),
                "run_dir": str(run_dir.resolve()),
                "success_rate": successes / requests,
                "slo_goodput": float(summary["slo_goodput_request_per_s"]),
                "ttft_p95_ms": float(summary["ttft_ms_p95"]),
            }
        )
    eligible = [item for item in candidates if item["success_rate"] >= 0.99]
    if not eligible:
        raise ValueError("no KV threshold candidate reached 99% success")
    selected = max(
        eligible, key=lambda item: (item["slo_goodput"], -item["ttft_p95_ms"])
    )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected["config"], output_config)
    report = {
        "mode": "kv_fixed_baseline",
        "selection_rule": "success>=99%, max SLO goodput, then min TTFT p95",
        "candidates": candidates,
        "selected": selected,
        "frozen_config": str(output_config.resolve()),
        "frozen_config_sha256": sha256_file(output_config),
    }
    write_json(report_path, report)
    return report


def select_pd(
    monolithic_dir: Path,
    pd_dir: Path,
    template: Path,
    output_config: Path,
    report_path: Path,
) -> dict[str, Any]:
    grouped: dict[str, dict[int, list[float]]] = {
        "monolithic": defaultdict(list),
        "pd": defaultdict(list),
    }
    for label, run_dir in (("monolithic", monolithic_dir), ("pd", pd_dir)):
        for row in read_jsonl(run_dir / "requests.jsonl"):
            if row.get("success") and row.get("e2e_ms") is not None:
                grouped[label][int(row["input_tokens"])].append(float(row["e2e_ms"]))
    common = sorted(set(grouped["monolithic"]) & set(grouped["pd"]))
    comparisons = []
    threshold = 40960
    for prompt_tokens in common:
        mono = sorted(grouped["monolithic"][prompt_tokens])
        pd_values = sorted(grouped["pd"][prompt_tokens])
        mono_median = mono[len(mono) // 2]
        pd_median = pd_values[len(pd_values) // 2]
        comparisons.append(
            {
                "prompt_tokens": prompt_tokens,
                "monolithic_e2e_median_ms": mono_median,
                "pd_e2e_median_ms": pd_median,
            }
        )
        if pd_median <= mono_median:
            threshold = min(threshold, prompt_tokens)
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["pd_fixed_min_prompt_tokens"] = threshold
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {
        "mode": "pd_fixed_baseline",
        "selection_rule": "first prompt length where PD median E2E <= monolithic",
        "comparisons": comparisons,
        "selected_threshold": threshold,
        "frozen_config": str(output_config.resolve()),
        "frozen_config_sha256": sha256_file(output_config),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    kv = subparsers.add_parser("kv")
    kv.add_argument("--candidate", action="append", required=True)
    kv.add_argument("--output-config", type=Path, required=True)
    kv.add_argument("--report", type=Path, required=True)
    pd = subparsers.add_parser("pd")
    pd.add_argument("--monolithic-run", type=Path, required=True)
    pd.add_argument("--pd-run", type=Path, required=True)
    pd.add_argument("--template", type=Path, required=True)
    pd.add_argument("--output-config", type=Path, required=True)
    pd.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "kv":
        report = select_kv(args.candidate, args.output_config, args.report)
    else:
        report = select_pd(
            args.monolithic_run,
            args.pd_run,
            args.template,
            args.output_config,
            args.report,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
