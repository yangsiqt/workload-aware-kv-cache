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

## Multi-GPU Handoff

1. Run `scripts/check_environment.sh` and preserve its log.
2. Copy and verify the 28 model files with `scripts/copy_model_to_data_disk.sh`.
3. Start one baseline using `scripts/serve_vllm_baseline.sh`.
4. Run `scripts/smoke_test_api.sh` before any workload sweep.
5. Execute the configured closed-loop and Poisson matrices only after the
   single-request baseline is stable.

Do not install `/root/vllm` in editable mode during baseline collection.
