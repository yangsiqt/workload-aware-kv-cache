from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RouteMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    session_id: str = ""
    prefix_hash: str = ""
    priority: int = Field(default=1, ge=0, le=2)
    prompt_tokens: int = Field(default=0, ge=0)
    shared_prefix_tokens: int = Field(default=0, ge=0)
    expected_output_tokens: int = Field(default=0, ge=0)
    metadata_source: str = "fallback"


class CandidateTrace(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend_url: str
    running: int = Field(ge=0)
    waiting: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    cache_source: Literal["none", "affinity", "vllm_event", "lmcache_lookup"]
    cache_confidence: float = Field(ge=0, le=1)
    queue_ms: float = Field(ge=0)
    prefill_ms: float = Field(ge=0)
    external_kv_ms: float = Field(default=0, ge=0)
    pd_transfer_ms: float = Field(default=0, ge=0)
    slo_penalty_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    stale: bool


class RouteTraceEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["1.0", "1.1"] = "1.1"
    event: Literal["decision", "completion"]
    request_id: str
    attempt_id: int = Field(default=0, ge=0)
    decision_id: str = ""
    policy: str
    backend_url: str
    reason: str
    metadata: RouteMetadata
    candidates: list[CandidateTrace]
    decided_at: float
    success: bool | None = None
    error: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.decision_id:
            self.decision_id = f"{self.request_id}:{self.attempt_id}"


class RouteAttempt(BaseModel):
    attempt_id: int = Field(ge=0)
    decision: RouteTraceEvent
    completion: RouteTraceEvent


class JoinedTrace(BaseModel):
    request_id: str
    client: dict[str, Any]
    route: RouteTraceEvent
    attempts: list[RouteAttempt]
