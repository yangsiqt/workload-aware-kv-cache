import asyncio
import time

import aiohttp
import pytest
from aiohttp import web

from benchmarks.run_benchmark import _request
from benchmarks.schemas import ChatMessage, SourceInfo, WorkloadItem


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def item() -> WorkloadItem:
    return WorkloadItem(
        dataset_name="test",
        dataset_revision="r",
        dataset_instance_id="i",
        request_id="request",
        session_id="session",
        turn_id=0,
        priority=1,
        request_type="test",
        prefix_hash="hash",
        messages=[ChatMessage(role="user", content="test")],
        prompt_tokens=1,
        shared_prefix_tokens=0,
        expected_output_tokens=2,
        source=SourceInfo(dataset="test", license="test", snapshot_id="snapshot"),
    )


async def run_request(handler, timeout=1.0):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            return await _request(
                session,
                item(),
                run_id="run",
                endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
                model="model",
                tokenizer=FakeTokenizer(),
                route_policy="direct",
                temperature=0,
                offered_at=time.perf_counter(),
            )
    finally:
        await server.cleanup()


@pytest.mark.asyncio
async def test_http_error_is_not_retried() -> None:
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.json_response({"error": "failed"}, status=503)

    result = await run_request(handler)
    assert calls == 1
    assert not result.success
    assert result.status_code == 503


@pytest.mark.asyncio
async def test_stream_without_done_is_an_error() -> None:
    async def handler(request):
        return web.Response(
            text='data: {"choices":[{"delta":{"content":"hello world"}}]}\n\n',
            content_type="text/event-stream",
        )

    result = await run_request(handler)
    assert not result.success
    assert result.error == "stream ended without [DONE]"
    assert result.output_tokens == 2


@pytest.mark.asyncio
async def test_timeout_is_recorded() -> None:
    async def handler(request):
        await asyncio.sleep(0.1)
        return web.Response(text="data: [DONE]\n\n", content_type="text/event-stream")

    result = await run_request(handler, timeout=0.01)
    assert not result.success
    assert result.error.startswith("TimeoutError")


@pytest.mark.asyncio
async def test_coalesced_sse_events_are_one_client_chunk() -> None:
    async def handler(request):
        assert request.headers["X-Shared-Prefix-Tokens"] == "0"
        body = (
            'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return web.Response(text=body, content_type="text/event-stream")

    result = await run_request(handler)
    assert result.success
    assert result.output_tokens == 2
    assert result.inter_chunk_latencies_ms == []
