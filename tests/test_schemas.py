import pytest
from pydantic import ValidationError

from benchmarks.schemas import WorkloadItem


def test_shared_prefix_cannot_exceed_prompt() -> None:
    with pytest.raises(ValidationError):
        WorkloadItem.model_validate({
            "dataset_name": "test", "dataset_revision": "r", "dataset_instance_id": "i",
            "request_id": "q", "session_id": "s", "turn_id": 0, "priority": 1,
            "request_type": "test", "prefix_hash": "hash", "messages": [{"role": "user", "content": "x"}],
            "prompt_tokens": 2, "shared_prefix_tokens": 3, "expected_output_tokens": 1,
            "source": {"dataset": "test", "license": "test", "snapshot_id": "snapshot"},
        })
