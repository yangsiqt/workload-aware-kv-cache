from pathlib import Path

from benchmarks.io_utils import write_jsonl
from benchmarks.validate_v2_1_k01 import _mooncake_read_deltas


def test_mooncake_read_deltas_require_real_counter_growth(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "backend_metrics.jsonl",
        [
            {
                "source": "mooncake",
                "backend_id": "gpu0",
                "read_bytes_total": 0,
                "read_operations_total": 0,
                "error": None,
            },
            {
                "source": "mooncake",
                "backend_id": "gpu0",
                "read_bytes_total": 4096,
                "read_operations_total": 1,
                "error": None,
            },
            {
                "source": "vllm",
                "backend_id": "gpu0",
                "read_bytes_total": 9999,
                "read_operations_total": 9,
                "error": None,
            },
        ],
    )
    assert _mooncake_read_deltas(tmp_path) == {
        "gpu0": {"bytes": 4096.0, "operations": 1.0}
    }
