from benchmarks.run_benchmark import percentile


def test_percentile_interpolates() -> None:
    assert percentile([], 0.5) is None
    assert percentile([1.0], 0.99) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
