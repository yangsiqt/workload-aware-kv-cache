from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset: str
    license: str
    snapshot_id: str
    repo: str | None = None
    base_commit: str | None = None
    public_id: str | None = None
    url: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class WorkloadItem(BaseModel):
    schema_version: str = "1.0"
    dataset_name: str
    dataset_revision: str
    dataset_instance_id: str
    transform_version: str = "1.0"
    request_id: str
    session_id: str
    turn_id: int = Field(ge=0)
    priority: int = Field(ge=0, le=2)
    request_type: str
    prefix_hash: str
    messages: list[ChatMessage]
    prompt_tokens: int = Field(gt=0)
    shared_prefix_tokens: int = Field(ge=0)
    expected_output_tokens: int = Field(gt=0)
    source: SourceInfo

    @model_validator(mode="after")
    def validate_token_counts(self) -> "WorkloadItem":
        if self.shared_prefix_tokens > self.prompt_tokens:
            raise ValueError("shared_prefix_tokens cannot exceed prompt_tokens")
        return self


class RequestResult(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    request_id: str
    session_id: str
    turn_id: int
    dataset_name: str
    request_type: str
    prefix_hash: str
    priority: int
    route_policy: str
    backend_id: str | None = None
    route_reason: str | None = None
    cache_hit: bool | None = None
    offered_at_s: float | None = None
    started_at_s: float
    completed_at_s: float
    ttft_ms: float | None = None
    e2e_ms: float
    tpot_ms: float | None = None
    inter_chunk_latencies_ms: list[float] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    status_code: int | None = None
    success: bool
    error: str | None = None


class RunManifest(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    created_at: str
    project_commit: str
    mode: Literal["closed_loop", "poisson"]
    endpoint: str
    model: str
    workload_path: str
    workload_sha256: str
    workload_count: int
    max_concurrency: int
    request_rate: float | None = None
    seed: int = 42
    arrival_trace_path: str | None = None
    arrival_trace_sha256: str | None = None
    simulated: bool = False
    config: dict[str, Any] = Field(default_factory=dict)


class ArrivalTraceItem(BaseModel):
    schema_version: str = "1.0"
    request_id: str
    offset_s: float = Field(ge=0)
