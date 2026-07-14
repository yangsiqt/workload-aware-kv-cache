# Workload-aware KV Cache for Long-context Code Agents

This project studies repeated prefill and cross-instance KV cache misses in
long-context code-agent serving. It uses Qwen3-30B-A3B-Instruct-2507 with
vLLM as the primary engine and SGLang as a later comparison baseline.

## Workloads

- **SWE-bench Verified** supplies real repository snapshots and issue text for
  the primary multi-turn code-agent workload.
- **LongBench RepoBench-P** supplies a standardized repository-level
  long-context workload.
- **ShareGPT** supplies a conventional serving baseline compatible with vLLM
  benchmark tooling.
- **Controlled prefix** workloads isolate routing and cache-locality effects.

This is a serving-systems benchmark. It does not report SWE-bench solve rates
or use gold patches in model prompts.

## Current Boundary

Before renting the multi-GPU instance, the project builds and tests the data
pipeline, benchmark client, router, mock backends, result analysis, and launch
templates. Mock results are always marked `SIMULATED` and are not performance
claims.

## Layout

```text
benchmarks/   dataset adapters, workload generation, load client, analysis
router/       workload-aware proxy, policies, registry, mock backend
configs/      workload, benchmark, router, and vLLM templates
scripts/      environment checks, launch helpers, and smoke tests
tests/        unit and local end-to-end tests
data/         manifests and tiny test fixtures only
reports/      generated summaries and figures
```

All non-model artifacts default to `/root/workload-aware-kv-cache-data` so they
are included in the AutoDL system image. Only model weights live on the data
disk/file storage. Override the data location with `WORKLOAD_DATA_ROOT`.

## Metric Definitions

- TTFT: request start to first non-empty streamed content delta.
- E2E: request start to stream completion.
- TPOT: `(E2E - TTFT) / (output_tokens - 1)`.
- ITL: inter-SSE-chunk latency. It is not presented as exact per-token latency.

See `configs/` for reproducible defaults. GPU measurements must record the
hardware topology, model revision, project commit, engine version, and full
server arguments.
