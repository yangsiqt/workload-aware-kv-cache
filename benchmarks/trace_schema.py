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
    waiting_prefill_tokens: int = Field(default=0, ge=0)
    running_prefill_tokens: int = Field(default=0, ge=0)
    reserved_prefill_tokens: int = Field(default=0, ge=0)
    reserved_external_load_ms: float = Field(default=0, ge=0)
    reserved_kv_blocks: int = Field(default=0, ge=0)
    active_decode_sequences: int = Field(default=0, ge=0)
    scheduled_prefill_tokens: int = Field(default=0, ge=0)
    scheduled_decode_tokens: int = Field(default=0, ge=0)
    skipped_waiting_prefill_tokens: int = Field(default=0, ge=0)
    kv_cache_free_blocks: int = Field(default=0, ge=0)
    kv_cache_total_blocks: int = Field(default=0, ge=0)
    kv_pressure_ms: float = Field(default=0, ge=0)
    preemptions_total: int = Field(default=0, ge=0)
    workload_metrics_available: bool = False
    v2_1_metrics_available: bool = False
    cached_tokens: int = Field(ge=0)
    local_hbm_cached_tokens: int = Field(default=0, ge=0)
    required_kv_blocks: int = Field(default=0, ge=0)
    cache_source: Literal[
        "none",
        "affinity",
        "vllm_event",
        "vllm_kv_event",
        "vllm_event_unverified",
        "lmcache_lookup",
        "lmcache_l1",
        "mooncake_l2",
    ]
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

    schema_version: Literal["1.0", "1.1", "1.2", "2.0", "2.1", "2.2"] = "1.2"
    event: Literal["decision", "completion"]
    request_id: str
    attempt_id: int = Field(default=0, ge=0)
    decision_id: str = ""
    policy: str
    backend_url: str
    reason: str
    metadata: RouteMetadata
    candidates: list[CandidateTrace]
    kv_path: dict[str, Any] | None = None
    execution_mode: dict[str, Any] | None = None
    v2_context: dict[str, Any] | None = None
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
    worker_events: list[dict[str, Any]] = Field(default_factory=list)


class JoinedTrace(BaseModel):
    request_id: str
    client: dict[str, Any]
    route: RouteTraceEvent
    attempts: list[RouteAttempt]
