from types import SimpleNamespace

import pytest

from benchmarks.build_single_h20_profile import select_session_turns


def test_select_session_turns_is_ordered_and_exact() -> None:
    rows = [
        SimpleNamespace(session_id=session, turn_id=turn)
        for session in ("b", "a")
        for turn in range(3)
    ]
    selected = select_session_turns(rows, ("a", "b"), 2)
    assert [(row.session_id, row.turn_id) for row in selected] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]


def test_select_session_turns_rejects_missing_rows() -> None:
    with pytest.raises(ValueError, match="expected 2 requests"):
        select_session_turns([SimpleNamespace(session_id="a", turn_id=0)], ("a",), 2)
