# MoE expert placement implementation progress

This file is the source of truth for **Version 1: complete static per-expert MoE placement**.

Version 2 and adaptive rebalancing are out of scope until Version 1 is complete, benchmarked, and accepted.

Detailed design:

- `docs/moe-expert-static-placement-spec.md`
- `docs/moe-expert-static-placement-design-v1.md`
- `docs/moe-expert-placement-user-checks.md`

Active work:

- profiling branch: `agent/moe-expert-stats`
- profiling PR: https://github.com/erukolya/llama-cpp-turboquant/pull/1
- implementation branch: `agent/static-moe-expert-placement`
- implementation draft PR: https://github.com/erukolya/llama-cpp-turboquant/pull/2

## Status legend

| Marker | Meaning |
|---|---|
| `[x]` | Implemented and committed |
| `[~]` | Active or awaiting automated validation |
| `[!]` | Blocked by a required gate |
| `[ ]` | Not started |

## User involvement

**Current status: USER NOT NEEDED.**

The implementation agent continues independently until a result depends on the user's real KAT GGUF and RTX 5070 Ti.

Version 1 uses at most three planned hardware checkpoints:

1. **U1:** exact KAT model/plan dry run;
2. **U2:** exclusive split-loader RAM/VRAM validation;
3. **U3:** final correctness, PCIe, and performance validation.

A user check is requested only with a ready CUDA artifact and one exact PowerShell command.

## Mandatory invariants

- hot routed expert weights exist only in VRAM;
- cold routed expert weights exist only in RAM;
- the complete routed-expert bank is not duplicated in RAM;
- ordinary decode performs no routed-expert weight transfer over PCIe;
- GPU experts execute on CUDA;
- CPU experts execute on CPU;
- mixed top-k assignments are partitioned between CPU and CUDA;
- CPU and CUDA branches overlap where possible;
- router scores, top-k, quantization, shared experts, and model semantics are unchanged;
- stock behavior without the feature remains unchanged;
- no LRU, miss loading, migration, rebalance, or NVMe tier is part of Version 1.

## Current project state

**Current phase:** V1.1 exact model-aware dry run and U1 packaging.

**Current decision:** do not start split-pool allocation until the V1.1 automated checks and U1 real-model validation pass.

Confirmed current baseline:

```text
CUDA_Host packed expert tensor
-> read selected top-k IDs
-> copy selected expert slices over PCIe
-> CUDA MUL_MAT_ID
```

The future static CPU pool must explicitly avoid this operation-offload path.

Measured KAT planning result:

| Metric | Value |
|---|---:|
| Logical routed experts | 10,240 |
| Existing VRAM expert count | 4,864 |
| Selected by byte-aware plan | 4,863 |
| Routed-expert VRAM budget | 10,338 MiB |
| Used by generated plan | 10,337.664 MiB |
| Existing estimated GPU selection coverage | about 47.5% |
| Planned estimated GPU selection coverage | about 71.242% |

These values are routing-selection coverage, not expected tokens/s improvement.

---

# Version 1 roadmap

## P0 — Requirements and architecture

- [x] Write the complete technical specification.
- [x] Define Version 1 as the only active scope.
- [x] Record exclusive residency.
- [x] Prohibit decode-path weight streaming.
- [x] Define correctness, memory, PCIe, and benchmark gates.
- [x] Collect llama.cpp, ik_llama.cpp, KTransformers, and Colibrì references.

## P1 — Profiler and source data

- [x] Collect per `(layer, expert)` routing counts.
- [x] Support exact and sampled profiling.
- [x] Export packed tensor placement.
- [x] Confirm KAT topology: 40 layers, 256 experts, top-8.
- [x] Export exact `size_bytes`.
- [x] Export `n_experts`.
- [x] Export `expert_stride_bytes = nb[2]`.
- [x] Export `ggml_type`, `ne[4]`, and `nb[4]`.

## P2 — Existing runtime path

- [x] Locate model tensor allocation and loading.
- [x] Locate `build_moe_ffn` and `MUL_MAT_ID`.
- [x] Confirm backend preference follows weight residency.
- [x] Confirm operation offload can override CPU residency.
- [x] Confirm selected expert slices are copied RAM -> CUDA.
- [x] Confirm tensor views cannot provide independent residency.
- [~] Add scheduler/public API counters for selective MoE weight copies and CPU/accelerator `MUL_MAT_ID` execution; awaiting CI.

## V1.1 — Offline planner

- [x] Parse profiler CSV.
- [x] Parse placement CSV.
- [x] Read routed tensor metadata directly from GGUF with bundled `gguf-py`.
- [x] Compute logical expert bytes from gate/up/down strides.
- [x] Rank by estimated hits per byte.
- [x] Select under an exact routed-expert VRAM budget.
- [x] Treat missing sampled rows as zero-hit with low confidence.
- [x] Generate legacy schema v1 where exact metadata is unavailable.
- [x] Generate schema v2 with an exact tensor manifest.
- [x] Generate a sampled model fingerprint from file size, prefix, and suffix.
- [x] Canonicalize GGUF type names to `ggml_type_name()` format.
- [x] Write per-expert CSV report.
- [x] Add deterministic tie-breaking.
- [x] Add Python unit tests.
- [x] Validate the selection algorithm with real KAT statistics.

## V1.1 — C++ plan contract

- [x] Add reusable plan structures.
- [x] Parse schema v1 and schema v2.
- [x] Validate budgets, counts, IDs, and hit totals.
- [x] Validate tensor manifest names, ranges, types, shapes, strides, and sizes.
- [x] Add deterministic JSON round trip.
- [x] Add `llama-moe-plan-check`.
- [x] Add C++ unit tests.
- [~] Complete all current CI configurations.

Current automated state for the latest schema-v2 work:

- [x] Windows server build compiles.
- [x] Ubuntu server build compiles.
- [~] Windows and Ubuntu tests running.
- [~] CPU matrix running/queued.
- [~] Python type-check running/queued.
- [~] flake8 running/queued.
- [~] self-hosted CI running/queued.

## V1.1 — Runtime model-aware dry run

- [x] Add `--moe-expert-plan FNAME`.
- [x] Add `--moe-expert-plan-strict`.
- [x] Add `--moe-expert-plan-dry-run`.
- [x] Load the plan after normal model loading without changing allocation.
- [x] Compare the sampled model fingerprint.
- [x] Enumerate real routed tensors from `llama_model::layers`.
- [x] Validate tensor names and count.
- [x] Validate canonical `ggml_type`.
- [x] Validate `ne[4]` and `nb[4]`.
- [x] Validate routed expert axis `ne[2]`.
- [x] Validate expert stride `nb[2]`.
- [x] Validate exact per-layer GPU expert bytes.
- [x] Support separate gate/up/down and merged gate-up layouts.
- [x] Detect unsupported per-expert bias/scale/input-scale tensors.
- [x] Print intended per-layer CPU/VRAM counts and bytes.
- [x] Fail explicitly in strict mode.
- [x] Keep allocation and inference unchanged.
- [~] Complete exact KAT U1 validation.

### U1 packaging

- [x] Add one-command `run-moe-u1.ps1`.
- [x] Make the script create schema-v2 plan and report automatically.
- [x] Make the script wait for the validation marker and collect logs.
- [x] Make the script create a single result ZIP.
- [~] Build Windows CUDA 13.3 artifact containing server, planner, script, and `gguf-py`.
- [!] User runs U1 only after the artifact is available.

**V1.1 exit gate:**

- all relevant automated checks are green;
- Python schema-v2 output is accepted by C++;
- exact KAT manifest matches runtime tensors;
- dry run changes no allocation or inference;
- unsupported layouts fail explicitly;
- U1 result package is accepted.

## V1.2 — Exclusive split storage and selective loading

**Status:** blocked until the V1.1 exit gate.

- [~] Design ownership and lifecycle of CPU/GPU expert pools.
- [x] Locate Qwen3.5 MoE packed tensor creation in `src/models/qwen35moe.cpp`.
- [x] Confirm the graph enters one `build_moe_ffn` path.
- [x] Identify that the plan must reach model loading before tensor creation.
- [x] Identify `load_all_data` support for runtime tensors without source weights as a compact-pool extension point.
- [ ] Move plan access from server-only integration to the model-loading boundary.
- [ ] Add per-layer global-to-local location tables.
- [ ] Create compact CPU routed-expert pools.
- [ ] Create compact CUDA routed-expert pools.
- [ ] Support merged gate-up tensors.
- [ ] Copy quantized slices without dequantization.
- [ ] Validate quant-block and backend alignment.
- [ ] Read slices directly from GGUF/mmap/staging.
- [ ] Avoid allocating the original persistent packed routed tensors.
- [ ] Prove no complete RAM expert bank remains.
- [ ] Add all-CPU, all-GPU, and mixed loading tests.
- [ ] Add exact RAM/VRAM accounting.
- [!] Run U2 on the primary machine.

**V1.2 exit gate:** compact pools load and unload correctly, permanent CPU+VRAM expert bytes approximately equal the original routed bank, and no full RAM duplicate remains.

## V1.3 — Correct mixed execution

- [ ] Partition original top-k by location table.
- [ ] Remap global IDs to local CPU/GPU pool IDs.
- [ ] Preserve token indices and router weights.
- [ ] Execute CPU pools only on CPU.
- [ ] Execute CUDA pools only on CUDA.
- [ ] Prevent operation offload of CPU-pool `MUL_MAT_ID`.
- [ ] Implement split gate/up activation.
- [ ] Implement split down projection.
- [ ] Join contributions exactly once.
- [ ] Preserve shared experts and optional auxiliary tensors.
- [ ] Handle 0 through top-k GPU assignments.
- [ ] Handle different splits across tokens in one ubatch.
- [ ] Add tensor-level, logits, and perplexity correctness tests.

**V1.3 exit gate:** correctness passes before optimization.

## V1.4 — Parallel CPU/CUDA execution

- [ ] Add synthetic CPU/GPU fork-and-join test.
- [ ] Determine whether the standard scheduler overlaps branches.
- [ ] Select scheduler split or dedicated hybrid operation.
- [ ] Use pinned activation/result buffers.
- [ ] Use asynchronous copies and events.
- [ ] Avoid per-layer global device synchronization.
- [ ] Bypass empty CPU or CUDA branches.
- [ ] Add CPU/GPU/join timing.
- [ ] Confirm time approaches `max(CPU, GPU)`, not their sum.

## V1.5 — Integration and metrics

- [ ] Integrate core behavior with server, CLI, and bench.
- [ ] Preserve TurboQuant KV-cache behavior.
- [ ] Add routing coverage metrics.
- [ ] Add activation-transfer counters.
- [ ] Add routed-expert weight-transfer counter.
- [ ] Confirm weight-transfer counter remains zero after load during decode.
- [ ] Benchmark PP and TG separately.
- [ ] Compare current layer placement with static per-expert placement.
- [ ] Measure RAM, VRAM, and PCIe traffic.
- [ ] Produce reproducible benchmark report.

## V1.6 — Production acceptance

- [ ] Validate startup, shutdown, unload, and reload.
- [ ] Validate long-running server operation.
- [ ] Validate `--parallel 1` end to end.
- [ ] Explicitly reject unsupported multi-GPU/RPC/backend combinations.
- [ ] Validate no-plan stock behavior.
- [ ] Validate strict failure and non-strict fallback.
- [ ] Document supported architectures, quants, and backends.
- [ ] Document profile -> plan -> launch workflow.
- [!] Run final U3 correctness/performance package.

**Version 1 acceptance gate:**

- required CI and tests are green;
- model output is correct within accepted numerical tolerance;
- no permanent routed-expert duplication;
- no routed-expert weight transfers during ordinary decode;
- CPU and CUDA experts execute on their assigned backend;
- actual coverage matches the plan within profiling error;
- load, unload, and shutdown are stable;
- TurboQuant KV remains functional;
- TG regression is not greater than 5%;
- target TG improvement is at least 15%, or a measured bottleneck analysis explains why not;
- the full workflow is reproducible.

Version 1 is not complete when only the planner or loader works.

---

# Explicitly deferred

Do not implement on the current branch:

- periodic or runtime expert rebalancing;
- EWMA migration scores;
- LRU/LFU cache;
- cache-miss weight loading;
- RAM/VRAM swaps during server operation;
- NVMe tier;
- migration generations or rollback APIs.

# Update policy

Every implementation commit that starts, completes, or blocks a tracked item updates this file in the same commit or immediately afterward. An item is complete only after its test or gate passes.

# Change log

## 2026-07-27

- Added CPU-versus-accelerator `MUL_MAT_ID` execution counters for MoE verification; awaiting CI.
- Exposed selective MoE weight-copy counters through the public context API and performance report; awaiting CI.
- Added cumulative scheduler counters for selective MoE weight-copy bytes, payload bytes, expert slices, copy calls, and packed weight inputs; awaiting CI.
- Created the Version 1 tracker and removed adaptive Version 2 from active scope.
- Completed profiling, execution-path analysis, offline planner, and initial C++ plan contract.
- Added schema-v2 exact tensor manifest and sampled model fingerprint.
- Added strict server-side comparison of names, types, shapes, strides, sizes, and model fingerprint.
- Added canonical GGML quant type naming.
- Confirmed schema-v2 server code compiles on Windows and Ubuntu.
- Prepared one-command U1 validation and CUDA 13.3 artifact packaging.
- Kept split-pool allocation blocked until automated V1.1 checks and U1 pass.
