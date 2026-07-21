# Workload-aware KV Cache for Long-context Code Agents

[![Tests](https://github.com/yangsiqt/workload-aware-kv-cache/actions/workflows/tests.yml/badge.svg)](https://github.com/yangsiqt/workload-aware-kv-cache/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

This project studies repeated prefill and cross-instance KV cache misses in
long-context code-agent serving. It uses Qwen3-30B-A3B-Instruct-2507 with
vLLM as the primary engine and Production Stack as the routing framework.

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

## Project Status

The data pipeline, asynchronous benchmark client, mock backends, and pre-GPU
Router implementation are complete. The project extends the Production Stack
Python Router with an `agent_slo_aware` policy. Results currently tracked in
this repository are explicitly marked `SIMULATED`; real multi-H20 Before/After
measurements remain a separate milestone.

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
- ITL: client-observed interval between transport chunks containing non-empty SSE content. It is not exact server-side token emission latency.

See `configs/` for reproducible defaults. GPU measurements must record the
hardware topology, model revision, project commit, engine version, and full
server arguments.

## Local Preparation

```bash
./scripts/install_system_dependencies.sh
python -m pip install -r requirements.txt
python -m benchmarks.dataset_adapters
python -m benchmarks.workload_generator --profile small
python -m benchmarks.validate_workload \
  /root/workload-aware-kv-cache-data/processed/small/combined.jsonl \
  --tokenizer /root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507 \
  --raw-swebench /root/workload-aware-kv-cache-data/raw/swebench_verified.jsonl
pytest -q
```

The public dataset adapter pins official Hugging Face revisions and hashes.
On AutoDL it uses hash-equivalent ModelScope LFS mirrors for LongBench and
ShareGPT because those mirrors are directly reachable from the domestic
network. `git-lfs` and `ffmpeg` are required system packages; FFmpeg is also
needed for the vLLM CLI to load TorchCodec successfully.

Run `scripts/run_mock_stack.sh` in one terminal and
`scripts/run_simulated_experiment.sh` in another to reproduce the local router
experiment. These outputs carry a `SIMULATED` marker. They validate the
measurement pipeline but are not GPU performance results.

The Production Stack integration is maintained in the companion fork on the
`feature/agent-slo-aware-router` branch. With that branch installed in
`/root/.venvs/vllm-router`, run the official-policy and project-policy matrix:

```bash
./scripts/run_guarded.sh /root/log/workload-aware-kv-cache/router-matrix.log \
  -- ./scripts/run_production_policy_matrix.sh
```

The script starts two cold Mock backends for each policy, exercises
`roundrobin`, `session`, `prefixaware`, and `agent_slo_aware`, writes a
watermarked comparison, and verifies that no Router or Mock process remains.

## Multi-GPU Handoff

1. Run `scripts/check_environment.sh` and preserve its log.
2. Copy and verify the 28 model files with `scripts/copy_model_to_data_disk.sh`.
3. Start one baseline using `scripts/serve_vllm_baseline.sh`.
4. Run `scripts/smoke_test_api.sh` before any workload sweep.
5. Execute the configured closed-loop and Poisson matrices only after the
   single-request baseline is stable.

Do not install `/root/vllm` in editable mode during baseline collection.

## Four-H20 Adaptive KV and PD Windows

The four-GPU stage uses a frozen 1200-request SWE-bench-derived workload and
separate, resumable Adaptive KV and Hybrid PD windows. Pre-rental validation is:

```bash
./scripts/check_four_h20_readiness.sh
```

On the 4 x H20 host, start with the mandatory runtime smoke in each window:

```bash
./scripts/run_four_h20_kv_window.sh
./scripts/run_four_h20_pd_window.sh
```

Both entrypoints support `--dry-run`, `--from K03/P02`, `--resume`, and an
explicit `--run-tag`. Failure injection is skipped by default and can be enabled
with `--include-failure`. The KV
window compares the best measured fixed LMCache retrieval threshold with the
adaptive Local HBM/L1/L2/Recompute selector. The PD window compares a measured
fixed prompt-length rule with Adaptive Monolithic/PD selection in the same
2M+1P1D layout. Formal Before/After runs share one frozen arrival-trace SHA and
report p50/p90/p95/p99, SLO goodput, path distributions, connector results,
fallbacks, prediction error, and Router decision overhead.

K02 records worker-observed LocalCPU/Mooncake retrieval locations and fits
Prefill/L1/L2 costs into a frozen Adaptive KV config. P02 similarly fits the
Prefill, PD transfer, and decode costs. Every stage gates request count, success
rate, Trace completeness, and required execution-path evidence before it can be
marked complete.

`READY FOR 4xH20` means all pre-rental artifacts pass. It does not replace K01
or P01: LMCache, Mooncake Store, and MooncakeConnector must still pass a real
four-H20 runtime smoke before formal results are collected.

## Workload Profiles

The pinned `pre_rental` screening profile contains 330 requests across 105
sessions:

- 15 SWE-bench Verified sessions and 90 deterministic agent turns spanning 11 repositories;
- five SWE sessions at each shared-prefix tier: 8K, 16K, and 32K tokens;
- 30 LongBench RepoBench-P requests and 30 ShareGPT requests;
- 30 controlled sessions and 180 requests for routing causality checks.

The exact SWE-bench instance IDs are pinned in `configs/workloads.yaml`.
Generated data is stored under
`/root/workload-aware-kv-cache-data/processed/pre_rental`; the tracked manifest
and validation summary are under `data/manifests/`.

The `final` profile pins 50 SWE-bench sessions and 300 deterministic Agent
turns. Three fixed Poisson arrival traces provide at least 900 business
requests for final repeated experiments. The earlier screening profile is not
used alone for a p99 headline.

```bash
GITHUB_ARCHIVE_MIRROR=https://ghfast.top/https://github.com \
  GITHUB_DOWNLOAD_MODE=direct \
  ./scripts/run_guarded.sh /root/log/workload-aware-kv-cache/prefetch-final.log \
  -- python -m benchmarks.prefetch_repos --profile final --workers 1

./scripts/run_guarded.sh /root/log/workload-aware-kv-cache/generate-final.log \
  -- python -m benchmarks.workload_generator --profile final

python -m benchmarks.generate_arrival_traces \
  /root/workload-aware-kv-cache-data/processed/final/swebench.jsonl \
  --output-dir /root/workload-aware-kv-cache-data/arrival-traces/final \
  --request-rate 2 --seeds 42,43,44
```

## Repository and Artifact Policy

The repository tracks source code, configs, tests, dataset revisions and hashes,
small result summaries, and reproducible figures. Model weights, raw public
datasets, repository snapshots, per-request traces, logs, credentials, and local
caches stay outside Git. Set `WORKLOAD_DATA_ROOT`, `MODEL_PATH`, and
`GITHUB_SSH_KEY` to adapt the examples to another machine.

SWE-bench Verified, LongBench, ShareGPT, Qwen, vLLM, Production Stack, LMCache,
and Mooncake remain subject
to their respective upstream licenses and terms. This repository does not
redistribute their raw data, source snapshots, or model weights.

## License

Project-authored source code is available under the Apache License 2.0. See
`LICENSE` for details.
