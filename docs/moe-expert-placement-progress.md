# MoE expert placement implementation progress

This file is the source of truth for **Version 1: complete static per-expert MoE placement**.

Version 2, adaptive rebalancing, LRU/LFU caching, miss loading and NVMe tiers are out of scope until Version 1 is complete, benchmarked and accepted.

Detailed design and results:

- `docs/moe-expert-static-placement-spec.md`
- `docs/moe-expert-static-placement-design-v1.md`
- `docs/moe-expert-placement-user-checks.md`
- `docs/moe-u1-results.md`
- `docs/moe-u2-results.md`

Active work:

- profiling branch: `agent/moe-expert-stats`
- profiling PR: https://github.com/erukolya/llama-cpp-turboquant/pull/1
- implementation branch: `agent/static-moe-expert-placement`
- implementation draft PR: https://github.com/erukolya/llama-cpp-turboquant/pull/2

## Status legend

| Marker | Meaning |
|---|---|
| `[x]` | Implemented and its automated gate passed |
| `[~]` | Implemented or active, awaiting the next acceptance gate |
| `[!]` | Blocked by a required hardware gate |
| `[ ]` | Not started |

## User involvement

**Current status: USER NOT NEEDED.**

The V1.3 mixed execution graph, server integration, backend route remap and guarded missing-slot execution are implemented. Windows and Ubuntu tests execute the real I32 route-map operation and guarded `MUL_MAT_ID(-1)` path. The dedicated Windows CUDA 13.3 U3 artifact is currently being built.

The next user action is one U3 command for Q4_K_M and one for Q5_K_M after the artifact is ready. Each command produces a diagnostic ZIP automatically.

Hardware checkpoints:

1. **U1 — complete:** exact model/plan dry run on Q5_K_M and Q4_K_M.
2. **U2 — accepted for Q5_K_M:** exact exclusive split-loader accounting. Q4_K_M structurally confirmed compact pools and zero packed tensors; its uploaded run used the Q5 plan.
3. **U3 — artifact building:** end-to-end baseline/static output, backend execution, zero weight-transfer and memory checks on both Q4_K_M and Q5_K_M.

## Mandatory invariants

- hot routed expert weights exist only in VRAM;
- cold routed expert weights exist only in RAM;
- the complete routed-expert bank is not duplicated in RAM;
- ordinary decode performs no routed-expert weight transfer over PCIe;
- GPU experts execute on CUDA;
- CPU experts execute on CPU;
- mixed top-k assignments preserve slot positions, router weights and model semantics;
- CPU and CUDA branches are joined exactly once;
- stock behavior without a placement plan remains unchanged.

## Current project state

**Current phase:** V1.3 implementation complete; U3 integration and hardware acceptance active.

Qwen3.5/3.6 MoE now claims packed routed tensors only as GGUF slice sources, allocates model-owned compact CPU and CUDA pools, and loads quantized expert slices without creating persistent packed runtime tensors. The router and top-k selection run once. Backend-local IDs are obtained from immutable I32 route maps, CPU and CUDA compact pools execute separate expert branches, and their weighted contributions are joined once with the result retained on the GPU side.

The server accepts `--moe-expert-plan` for real inference. Dry-run remains a separate stock-packed validation mode. Static placement disables stock all-expert warmup and exposes these Prometheus counters:

- CPU and accelerator `MUL_MAT_ID` executions;
- selective routed-weight bytes and payload bytes;
- expert slices, grouped copy calls and packed weight inputs.

U3 requires both execution counters to be positive and every routed-weight copy counter to remain exactly zero.

## Accepted U1 results

Both uploaded packages emitted `moe_plan: validated` with schema v2, 40 layers, 256 experts per layer and 120 exact routed tensor manifest entries.

| Metric | Q5_K_M | Q4_K_M |
|---|---:|---:|
| Routed expert bank | 21.289 GiB | 18.164 GiB |
| VRAM experts under 10,338 MiB budget | 4,863 | 5,725 |
| Cold routed bytes remaining in RAM | 11.194 GiB | 8.069 GiB |
| Estimated GPU routing coverage | 71.242% | 78.047% |
| Estimated CPU routing share | 28.758% | 21.953% |

Q4 adds 862 hot experts, improves estimated GPU coverage by 6.806 percentage points and reduces estimated CPU-routed selections by 23.665% relative to Q5. These are placement estimates, not measured tokens/s or quality.

---

# Version 1 roadmap

## P0 — Requirements and architecture

- [x] Complete technical specification.
- [x] Version 1 is the only active scope.
- [x] Exclusive residency and no decode-path weight streaming recorded.
- [x] Correctness, memory, PCIe and benchmark gates defined.

## P1 — Profiler and source data

- [x] Per `(layer, expert)` routing counts.
- [x] Exact and sampled profiling.
- [x] Exact tensor type, dimensions, strides and bytes.
- [x] KAT topology confirmed: 40 layers, 256 experts, top-8.

## P2 — Existing runtime path and instrumentation

- [x] Loader, graph and `MUL_MAT_ID` paths located.
- [x] Selective RAM-to-CUDA expert-slice copy path confirmed.
- [x] Tensor views rejected as an exclusive-residency solution.
- [x] Scheduler/public API counters for selective weight copies and CPU/accelerator `MUL_MAT_ID`.
- [x] Expose the counters through the server `/metrics` endpoint.

## V1.1 — Planner, contract and exact model validation

- [x] Byte-aware offline planner from stats + GGUF.
- [x] Schema-v2 tensor manifest and sampled model fingerprint.
- [x] Strict C++ parser, writer and structural validation.
- [x] Exact runtime comparison of names, types, `ne[]`, `nb[]`, sizes and strides.
- [x] Separate and merged gate-up layouts handled.
- [x] Unsupported auxiliary expert tensors rejected.
- [x] Windows CUDA 13.3 U1 artifact.
- [x] Q5_K_M U1 accepted.
- [x] Q4_K_M U1 accepted.

**V1.1 exit gate: passed.**

## V1.2 — Exclusive split storage and selective loading

- [x] Deterministic per-layer global-to-local CPU/GPU location tables.
- [x] Immutable placement snapshot owned by the model.
- [x] Compact CPU and CUDA routed-expert pools.
- [x] Direct quantized GGUF slice loading without dequantization.
- [x] Separate and merged gate-up layouts.
- [x] No original persistent packed routed tensor allocation.
- [x] Exact planned/actual CPU/GPU logical and allocated byte accounting.
- [x] Model-only load/unload validator on Windows and Ubuntu.
- [x] Model/plan fingerprint rejection before backend initialization and allocation.
- [x] Windows CUDA 13.3 U2 artifact.
- [x] Q5_K_M exact U2 accounting, zero packed tensors and unload accepted.
- [x] Q4_K_M compact pools and zero packed tensors structurally confirmed.
- [~] Optional Q4_K_M exact matching-plan accounting rerun; final U3 covers it.

**V1.2 exit gate:** passed for Q5_K_M; storage implementation structurally passed for Q4_K_M.

## V1.3 — Correct mixed execution

- [x] Split router/top-k generation from expert execution while retaining the stock wrapper.
- [x] Partition original top-k by immutable per-layer location maps.
- [x] Remap global IDs to local CPU/GPU IDs without changing top-k slot positions.
- [x] Preserve the original router weights for both branches.
- [x] Execute compact CPU and GPU expert branches separately.
- [x] Support separate and merged gate-up projection paths.
- [x] Preserve shared experts unchanged.
- [x] Handle arbitrary per-token CPU/GPU top-k splits with `-1` missing slots.
- [x] Guard `MUL_MAT_ID` missing-slot behavior without changing stock execution.
- [x] Implement the guarded path in generic CPU, x86 repack and CUDA backends.
- [x] Join CPU and GPU weighted contributions exactly once.
- [x] Add host slot/weight/sum tests on Windows and Ubuntu.
- [x] Add real backend I32 remap and guarded `MUL_MAT_ID(-1)` tests on Windows and Ubuntu.
- [~] End-to-end baseline/static output equality and real CUDA execution: U3 pending.

**V1.3 implementation gate: passed. Hardware correctness gate: U3 pending.**

## V1.4 — Parallel CPU/CUDA execution

- [ ] Add synthetic fork-and-join timing test.
- [~] Determine scheduler overlap behavior on the real mixed graph during U3.
- [ ] Use pinned activation/result buffers and asynchronous events if measurements require them.
- [ ] Avoid unnecessary per-layer synchronization.
- [ ] Add CPU/GPU/join timing.
- [ ] Confirm mixed time approaches `max(CPU, GPU)` rather than their sum.

## V1.5 — Integration and metrics

- [x] Integrate static placement with `llama-server` and CLI parsing.
- [x] Preserve a separate stock-packed dry-run validator.
- [x] Disable all-expert warmup for static placement.
- [x] Expose CPU/accelerator execution and routed-weight copy counters through `/metrics`.
- [~] Confirm routed-expert weight-transfer counters remain zero during real decode: U3 pending.
- [~] Preserve TurboQuant KV behavior: final server test pending.
- [ ] Benchmark PP and TG separately.
- [ ] Compare current layer placement against static per-expert placement.
- [ ] Measure RAM, VRAM and PCIe traffic.

## V1.6 — Production acceptance

- [~] Validate startup, generation, shutdown and unload through the U3 runner.
- [ ] Validate long-running server operation.
- [x] Implement and test `--parallel 1` U3 configuration.
- [ ] Reject unsupported multi-GPU/RPC/backend combinations explicitly.
- [~] Validate no-plan stock behavior and strict plan mode through U3.
- [~] Document supported models, quants, backends and workflow.
- [~] Build the Windows CUDA 13.3 U3 artifact.
- [!] Run final U3 on Q4_K_M and Q5_K_M.

## Version 1 acceptance gate

- required Windows, Ubuntu and CUDA builds are green;
- stock baseline and static output match under the deterministic U3 prompt;
- no permanent routed-expert duplication;
- every routed-expert weight-transfer counter remains zero during ordinary decode;
- CPU and CUDA execution counters are both positive;
- load, generation, shutdown and unload are stable;
- TurboQuant KV remains functional;
- TG regression is not greater than 5%;
- target TG improvement is at least 15%, or a measured bottleneck analysis explains why not;
- the workflow is reproducible for both Q4_K_M and Q5_K_M.

# Explicitly deferred

Do not implement on the current branch:

- periodic/runtime expert rebalancing;
- EWMA migration scores;
- LRU/LFU cache;
- cache-miss weight loading;
- RAM/VRAM swaps during server operation;
- NVMe tier.

# Change log

## 2026-07-28

- Accepted Q5_K_M U2 with exact CPU/GPU/total bytes, 240 compact tensors, zero packed routed tensors and successful unload.
- Diagnosed the Q4_K_M U2 archive as a Q4 model paired with the Q5 plan while confirming compact pools and zero packed tensors.
- Added early model fingerprint rejection before backend initialization/allocation.
- Split MoE routing/top-k generation from expert execution.
- Added model-owned CPU/GPU route-map tensors and direct compact tensor pointers on each layer.
- Added mixed CPU/CUDA expert branches with slot-preserving global-to-local ID remap.
- Added guarded negative route slots to generic CPU, x86 repack and CUDA `MUL_MAT_ID` paths.
- Added Windows and Ubuntu host partition, backend I32 remap and guarded missing-slot tests.
- Fixed Release tests that incorrectly depended on side effects inside `assert()`.
- Enabled real `llama-server --moe-expert-plan` inference while retaining dry-run validation.
- Disabled all-expert warmup for static placement.
- Exposed MoE execution and routed-weight copy counters through `/metrics`.
- Added `run-u3.ps1` for exact U2 accounting, deterministic baseline/static generation, output equality, backend execution and zero-copy checks.
- Started the targeted Windows CUDA 13.3 SM120 U3 artifact build.

## 2026-07-27

- Accepted U1 for Q5_K_M and Q4_K_M on RTX 5070 Ti.
- Added exact Q5/Q4 placement comparison in `docs/moe-u1-results.md`.
- Started V1.2 with deterministic global-to-local CPU/GPU location tables and byte accounting.
