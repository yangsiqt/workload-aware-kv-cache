from benchmarks.sse import SSEAccumulator


def test_split_event_and_multi_token_content() -> None:
    stream = SSEAccumulator()
    assert stream.feed(b'data: {"choices":[{"delta":{"content":"two') == []
    assert stream.feed(b' tokens"}}]}\n\n') == ["two tokens"]
    assert stream.feed(b"data: [DONE]\n\n") == []
    assert stream.text == "two tokens"
    assert stream.validation_error() is None


def test_empty_stream_and_missing_done() -> None:
    stream = SSEAccumulator()
    assert stream.validation_error() == "stream ended without [DONE]"
    stream.feed(b"data: [DONE]\r\n\r\n")
    assert stream.validation_error() == "stream contained no non-empty content delta"


def test_malformed_events_are_ignored() -> None:
    stream = SSEAccumulator()
    stream.feed(b"event: ping\n\ndata: not-json\n\ndata: [DONE]\n\n")
    assert stream.validation_error() == "stream contained no non-empty content delta"
