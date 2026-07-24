import asyncio
from pathlib import Path

import pytest

from benchmarks.run_benchmark import _wait_for_start_gate


def test_start_gate_requires_ready_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start-ready-file"):
        asyncio.run(_wait_for_start_gate(tmp_path / "gate", None, 0.1))


def test_start_gate_reports_ready_and_releases(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    gate = tmp_path / "gate"

    async def exercise() -> None:
        task = asyncio.create_task(_wait_for_start_gate(gate, ready, 1.0))
        while not ready.exists():
            await asyncio.sleep(0.01)
        assert not task.done()
        gate.touch()
        await task

    asyncio.run(exercise())
