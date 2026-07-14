from benchmarks.tokenizer_utils import tokenizer_fingerprint


def test_tokenizer_fingerprint_changes_with_content(tmp_path) -> None:
    artifact = tmp_path / "tokenizer.json"
    artifact.write_text("one")
    first = tokenizer_fingerprint(tmp_path)
    artifact.write_text("two")
    assert tokenizer_fingerprint(tmp_path) != first
