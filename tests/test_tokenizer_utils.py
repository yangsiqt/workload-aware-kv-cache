from benchmarks.tokenizer_utils import chat_tokens, token_ids_sha256


class FakeTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        return [1, 2, len(messages)]


def test_chat_tokens_are_integer_ids_and_hash_is_stable() -> None:
    ids = chat_tokens(FakeTokenizer(), [{"role": "user", "content": "x"}], add_generation_prompt=True)
    assert ids == [1, 2, 1]
    assert token_ids_sha256(ids) == token_ids_sha256(list(ids))
