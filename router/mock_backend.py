from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def create_app() -> FastAPI:
    app = FastAPI(title="KV cache mock backend")
    app.state.backend_id = os.getenv("BACKEND_ID", "mock-a")
    app.state.miss_ttft_ms = float(os.getenv("MISS_TTFT_MS", "180"))
    app.state.hit_ttft_ms = float(os.getenv("HIT_TTFT_MS", "25"))
    app.state.chunk_delay_ms = float(os.getenv("CHUNK_DELAY_MS", "4"))
    app.state.output_chunks = int(os.getenv("OUTPUT_CHUNKS", "8"))
    app.state.failure_rate = float(os.getenv("FAILURE_RATE", "0"))
    app.state.seed = int(os.getenv("MOCK_SEED", "42"))
    app.state.rng = random.Random(app.state.seed)
    app.state.seen_prefix_hashes = set()

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"healthy": True, "backend_id": app.state.backend_id}

    @app.get("/stats")
    async def stats() -> dict[str, object]:
        return {"backend_id": app.state.backend_id, "seen_prefixes": len(app.state.seen_prefix_hashes)}

    @app.post("/reset")
    async def reset() -> dict[str, object]:
        app.state.seen_prefix_hashes.clear()
        app.state.rng.seed(app.state.seed)
        return {"reset": True}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        prefix_hash = request.headers.get("X-Prefix-Hash", "")
        hit = prefix_hash in app.state.seen_prefix_hashes
        if app.state.rng.random() < app.state.failure_rate:
            return JSONResponse({"error": "configured mock failure"}, status_code=503)
        app.state.seen_prefix_hashes.add(prefix_hash)
        delay_ms = app.state.hit_ttft_ms if hit else app.state.miss_ttft_ms
        expected = int(request.headers.get("X-Expected-Output-Tokens", body.get("max_tokens", 8)))
        chunk_count = max(1, min(app.state.output_chunks, expected))
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"

        async def events() -> AsyncIterator[bytes]:
            await asyncio.sleep(delay_ms / 1000)
            for index in range(chunk_count):
                content = f" token{index}"
                event = {"id": stream_id, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
                yield f"data: {json.dumps(event)}\n\n".encode()
                await asyncio.sleep(app.state.chunk_delay_ms / 1000)
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "X-Backend-ID": app.state.backend_id,
            "X-Mock-Cache-Hit": str(hit).lower(),
        })

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "9101")))
    args = parser.parse_args()
    uvicorn.run("router.mock_backend:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
