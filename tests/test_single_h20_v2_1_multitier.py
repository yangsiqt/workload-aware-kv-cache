from pathlib import Path

from benchmarks.io_utils import read_jsonl
from benchmarks.single_h20_v2_1_multitier import build_profile, normalize_layout


def test_multitier_profile_has_target_and_lru_pressure(tmp_path: Path) -> None:
    manifest = build_profile(
        Path("/root/workload-aware-kv-cache-data"), tmp_path / "profile"
    )
    rows = list(read_jsonl(Path(manifest["profile_path"])))
    assert len(rows) == 5
    assert len({row["session_id"] for row in rows}) == 5
    assert [row["shared_prefix_tokens"] for row in rows] == [
        16384,
        32768,
        32768,
        32768,
        32768,
    ]
    assert manifest["estimated_working_set_gib"] > manifest["l1_gib"]
    assert manifest["estimated_working_set_gib"] < manifest["l2_gib"]


def test_normalize_layout_keeps_longest_per_tier() -> None:
    layout = normalize_layout(
        {
            "layout_info_v2": [
                ["a", 0, "LocalCPUBackend", 8192],
                ["a", 0, "RemoteBackend", 16384],
                ["b", 0, "RemoteBackend", 8192],
            ]
        }
    )
    assert layout == {"LocalCPUBackend": 8192, "RemoteBackend": 16384}
