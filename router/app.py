from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from benchmarks.io_utils import load_yaml
from router.cache_registry import AffinityRegistry
from router.routing_policy import Backend, RouterPolicy


def create_app(config_path: str | Path | None = None) -> FastAPI:
    config = load_yaml(Path(config_path or os.getenv("ROUTER_CONFIG", "configs/router.yaml")))
    app = FastAPI(title="Workload-aware KV cache router")
    backends = [Backend(id=value["id"], url=value["url"].rstrip("/")) for value in config["backends"]]
    registry = AffinityRegistry(float(config.get("registry_ttl_s", 3600)))
    policy_engine = RouterPolicy(backends, registry, int(config.get("seed", 42)))
    app.state.config = config
    app.state.policy = policy_engine
    app.state.registry = registry
    app.state.trace_path = Path(config["trace_path"])
    app.state.trace_path.parent.mkdir(parents=True, exist_ok=True)
    app.state.http = None

    @app.on_event("startup")
    async def startup() -> None:
        app.state.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if app.state.http:
            await app.state.http.close()

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"healthy": True, "policy": app.state.config["policy"], "backends": [value.id for value in backends]}

    @app.post("/reset")
    async def reset() -> dict[str, object]:
        registry.clear()
        async with aiohttp.ClientSession() as session:
            for backend in backends:
                try:
                    await session.post(f"{backend.url}/reset")
                except aiohttp.ClientError:
                    pass
        return {"reset": True}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.body()
        prefix_hash = request.headers.get("X-Prefix-Hash", "")
        session_id = request.headers.get("X-Session-ID", "")
        request_id = request.headers.get("X-Request-ID", "")
        route_policy = request.headers.get("X-Route-Policy", app.state.config["policy"])
        try:
            decision = policy_engine.choose(route_policy, prefix_hash, session_id)
        except (RuntimeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        backend = policy_engine.backends[decision.backend_id]
        backend.active_requests += 1
        trace = {
            "timestamp": datetime.now(UTC).isoformat(), "request_id": request_id,
            "session_id": session_id, "prefix_hash": prefix_hash, "priority": request.headers.get("X-Priority"),
            "policy": route_policy, "backend_id": backend.id, "reason": decision.reason,
            "active_at_decision": backend.active_requests - 1, "success": False,
        }
        headers = {key: value for key, value in request.headers.items() if key.lower() not in {"host", "content-length"}}
        try:
            upstream = await app.state.http.post(f"{backend.url}/v1/chat/completions", data=body, headers=headers)
        except aiohttp.ClientError as exc:
            backend.active_requests -= 1
            trace["error"] = str(exc)
            _append_trace(app.state.trace_path, trace)
            return JSONResponse({"error": "backend unavailable"}, status_code=502)
        response_headers = {
            "X-Backend-ID": backend.id,
            "X-Route-Policy": route_policy,
            "X-Route-Reason": decision.reason,
        }
        if "X-Mock-Cache-Hit" in upstream.headers:
            response_headers["X-Mock-Cache-Hit"] = upstream.headers["X-Mock-Cache-Hit"]

        async def stream() -> AsyncIterator[bytes]:
            completed = False
            try:
                async for chunk in upstream.content.iter_any():
                    if b"data: [DONE]" in chunk:
                        completed = True
                    yield chunk
            finally:
                upstream.release()
                backend.active_requests -= 1
                success = upstream.status < 300 and completed
                trace["success"] = success
                trace["status_code"] = upstream.status
                if success:
                    registry.commit(prefix_hash, session_id, backend.id)
                _append_trace(app.state.trace_path, trace)

        return StreamingResponse(stream(), status_code=upstream.status, media_type="text/event-stream", headers=response_headers)

    return app


def _append_trace(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    uvicorn.run("router.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
