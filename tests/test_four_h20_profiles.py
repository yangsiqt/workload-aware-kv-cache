from collections import Counter
from pathlib import Path

from benchmarks.build_four_h20_profiles import build
from benchmarks.io_utils import read_jsonl


def test_four_h20_profiles_have_frozen_sizes_and_unique_ids(tmp_path: Path) -> None:
    manifest = build(
        Path("/root/workload-aware-kv-cache-data/processed"), tmp_path
    )
    expected = {
        "four_h20_smoke.jsonl": 12,
        "kv_cost_controlled.jsonl": 24,
        "kv_threshold_screening.jsonl": 60,
        "pd_crossover.jsonl": 48,
        "failure.jsonl": 8,
    }
    actual = {
        name: artifact["rows"]
        for name, artifact in manifest["artifacts"].items()
    }
    assert {name: actual[name] for name in expected} == expected
    assert actual["pd_crossover_monolithic.jsonl"] == 24
    assert actual["pd_crossover_pd.jsonl"] == 24
    for name, count in expected.items():
        rows = list(read_jsonl(tmp_path / name))
        assert len(rows) == count
        assert len({row["request_id"] for row in rows}) == count
        turn_counts = Counter(row["session_id"] for row in rows)
        assert len(set(turn_counts.values())) == 1
