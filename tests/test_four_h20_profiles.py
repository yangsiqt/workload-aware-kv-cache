from collections import Counter
from pathlib import Path

from benchmarks.build_four_h20_profiles import build, stable_backend
from benchmarks.io_utils import read_jsonl


def test_four_h20_profiles_have_frozen_sizes_and_unique_ids(tmp_path: Path) -> None:
    manifest = build(Path("/root/workload-aware-kv-cache-data/processed"), tmp_path)
    expected = {
        "four_h20_smoke.jsonl": 12,
        "kv_cost_controlled.jsonl": 24,
        "kv_threshold_screening.jsonl": 60,
        "pd_crossover.jsonl": 48,
        "failure.jsonl": 8,
    }
    actual = {
        name: artifact["rows"] for name, artifact in manifest["artifacts"].items()
    }
    assert {name: actual[name] for name in expected} == expected
    assert actual["pd_crossover_monolithic.jsonl"] == 24
    assert actual["pd_crossover_pd.jsonl"] == 24
    assert actual["v2_1_k01_strict.jsonl"] == 4
    assert actual["v2_1_k01_lru.jsonl"] == 5
    assert actual["v2_1_k01_lru_target.jsonl"] == 1
    assert actual["v2_1_k01_lru_fillers.jsonl"] == 4
    assert actual["v2_1_k02_cost.jsonl"] == 12
    assert actual["v2_1_k03_capacity.jsonl"] == 120
    assert actual["v2_1_k03_capacity_confirm.jsonl"] == 60
    for name, count in expected.items():
        rows = list(read_jsonl(tmp_path / name))
        assert len(rows) == count
        assert len({row["request_id"] for row in rows}) == count
        turn_counts = Counter(row["session_id"] for row in rows)
        assert len(set(turn_counts.values())) == 1

    strict = list(read_jsonl(tmp_path / "v2_1_k01_strict.jsonl"))
    assert {stable_backend(row["session_id"]) for row in strict} == {0, 1, 2, 3}

    lru = list(read_jsonl(tmp_path / "v2_1_k01_lru.jsonl"))
    assert [row["shared_prefix_tokens"] for row in lru] == [
        16384,
        32768,
        32768,
        32768,
        32768,
    ]
    assert {stable_backend(row["session_id"]) for row in lru} == {0}
    assert sum(row["shared_prefix_tokens"] for row in lru) * 98304 > 8 * 2**30
    assert sum(row["shared_prefix_tokens"] for row in lru) * 98304 < 16 * 2**30

    cost = list(read_jsonl(tmp_path / "v2_1_k02_cost.jsonl"))
    coverage = {
        (row["shared_prefix_tokens"], stable_backend(row["session_id"])) for row in cost
    }
    assert coverage == {
        (length, backend) for length in (8192, 16384, 32768) for backend in range(4)
    }

    capacity = list(read_jsonl(tmp_path / "v2_1_k03_capacity.jsonl"))
    capacity_counts = Counter(row["session_id"] for row in capacity)
    assert len(capacity_counts) == 20
    assert set(capacity_counts.values()) == {6}
