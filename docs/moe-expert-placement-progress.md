# MoE expert placement implementation progress

This file is the source of truth for **Version 1: complete static per-expert MoE placement**.

Version 2, adaptive rebalancing, LRU/LFU caching, miss loading and NVMe tiers are out of scope until Version 1 is complete, benchmarked and accepted.

Detailed design:

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
| `[x]` | Implemented and its gate passed |
| `[~]` | Active or awaiting automated validation |
| `[!]` | Blocked by a required hardware gate |
| `[ ]` | Not started |

## User involvement

**Current status: USER NOT NEEDED.**

Q5_K_M completed the U2 exclusive-storage gate. The uploaded Q4_K_M package used the Q5 plan, so its exact byte comparison is intentionally not accepted; it still confirmed compact CPU/CUDA pools and zero packed routed tensors. The validator now rejects model/plan fingerprint mismatches before backend initialization and allocation.

No repeat Q4 test is required before V1.3. The next required user action is U3 only after mixed CPU/CUDA execution, correctness checks and a dedicated Windows CUDA 13.3 artifact are ready.

Planned hardware checkpoints:

1. **U1 — complete:** exact model/plan dry run on Q5_K_M and Q4_K_M.
2. **U2 — accepted for Q5_K_M:** exclusive split-loader RAM/VRAM validation. Q4_K_M exact matching-plan accounting remains optional before final acceptance; its structural exclusive-storage invariants passed.
3. **U3 — pending:** final correctness, PCIe and performance validation on both Q4_K_M and Q5_K_M.

## Mandatory invariants

- hot routed expert weights exist only in VRAM;
- cold routed expert weights exist only in RAM;
- the complete routed-expert bank is not duplicated in RAM;
- ordinary decode performs no routed-expert weight transfer over PCIe;
- GPU experts execute on CUDA;
- CPU experts execute on CPU;
- mixed top-k assignments are partitioned without changing router semantics;
- CPU and CUDA branches overlap where possible;
- stock behavior without a placement plan remains unchanged.

## Current project state

**Current phase:** V1.3 correct mixed CPU/CUDA execution.

**Current decision:** Qwen3.5/3.6 MoE can claim the packed routed tensors as GGUF slice sources, allocate compact model-owned CPU and CUDA pools, and load quantized expert slices without creating the original persistent packed runtime tensors. Q5_K_M proved exact planned/actual CPU, GPU and total routed bytes, zero packed routed pointers and successful unload. The Q4_K_M archive used the Q5 plan; the loaded Q4 compact bank was correctly smaller and still had zero packed tensors. Ordinary server inference with compact pools remains blocked until the V1.3 graph executes each tier on its assigned backend.

Confirmed baseline:

```text
CUDA_Host packed expert tensor
-> read selected top-k IDs
-> copy selected expert slices over PCIe
-> CUDA MUL_MAT_ID
```

The static CPU pool must explicitly avoid this operation-offload path.

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

**Status:** implementation complete; Q5_K_M hardware gate accepted, Q4_K_M structural invariants confirmed with exact matching-plan rerun deferred.

- [x] Design ownership and lifecycle of CPU/GPU expert pools.
- [x] Locate Qwen3.5 MoE packed tensor creation in `src/models/qwen35moe.cpp`.
- [x] Confirm one `build_moe_ffn` graph path.
- [x] Identify pre-allocation model-loading integration point.
- [x] Identify runtime tensors without source weights as the compact-pool extension point.
- [x] Add deterministic per-layer global-to-local CPU/GPU location tables.
- [x] Move validated placement access to the model-loading boundary.
- [x] Deep-copy immutable placement before the server load scope ends.
- [x] Plan compact CPU/GPU tensor shapes, exact bytes and coalesced copy spans.
- [x] Convert real GGUF `ggml_tensor` metadata into exact compact source layouts.
- [x] Claim packed GGUF tensors as slice sources without standard runtime allocation.
- [x] Validate all-CPU, all-GPU and mixed compact-pool plans.
- [x] Create compact CPU routed-expert pools.
- [x] Create compact CUDA routed-expert pools.
- [x] Support separate and merged gate-up tensors in model loading.
- [x] Copy quantized slices without dequantization.
- [x] Validate quant-block and backend alignment for Q5_K_M on real hardware; Q4_K_M produced the expected smaller compact bank under the mismatched Q5 plan.
- [x] Read slices directly from GGUF with `--no-mmap`; mmap and direct-I/O compact loading remain explicitly blocked.
- [x] Avoid allocating the original persistent packed routed tensors in Qwen3.5 MoE.
- [x] Prove no complete RAM expert bank remains for Q5_K_M; Q4_K_M also reported zero packed routed tensors.
- [x] Add all-CPU, all-GPU and mixed loading tests; Q5_K_M real mixed load passed.
- [x] Add exact planned/actual CPU/GPU logical and allocated byte accounting.
- [x] Add model-only load/unload validator and Windows PowerShell result package.
- [x] Build the model-only validator on Windows and Ubuntu.
- [x] Reject model/plan fingerprint mismatches before backend initialization and compact allocation.
- [x] Block ordinary compact-pool server inference until V1.3.
- [x] Build the final Windows CUDA 13.3 U2 artifact.
- [x] Run U2 on Q5_K_M and accept exact accounting, zero packed tensors and unload.
- [~] Optional Q4_K_M exact matching-plan accounting rerun; not required to continue V1.3.

**V1.2 exit gate:** passed for Q5_K_M. The same storage implementation structurally passed on Q4_K_M; exact Q4 plan accounting remains recorded as a deferred confirmation, not a V1.3 blocker.

## V1.3 — Correct mixed execution

- [~] Partition original top-k by location table; host reference and all-CPU/all-GPU/mixed tests are implemented, graph integration pending.
- [~] Remap global IDs to local CPU/GPU pool IDs while preserving top-k slot positions; host reference is implemented, graph integration pending.
- [ ] Execute CPU pools only on CPU and GPU pools only on CUDA.
- [ ] Prevent operation offload of CPU-pool `MUL_MAT_ID`.
- [ ] Implement split gate/up and down projection.
- [ ] Join contributions exactly once.
- [ ] Preserve shared experts, token indices and router weights.
- [ ] Handle arbitrary per-token CPU/GPU top-k splits.
- [ ] Add tensor-level, logits and perplexity correctness tests.

## V1.4 — Parallel CPU/CUDA execution

- [ ] Add synthetic fork-and-join test.
- [ ] Determine standard scheduler overlap behavior.
- [ ] Use pinned activation/result buffers and asynchronous events.
- [ ] Avoid per-layer global synchronization.
- [ ] Add CPU/GPU/join timing.
- [ ] Confirm mixed time approaches `max(CPU, GPU)` rather than their sum.

## V1.5 — Integration and metrics

- [ ] Integrate server, CLI and bench.
- [ ] Preserve TurboQuant KV behavior.
- [ ] Add routing coverage and activation-transfer metrics.
- [ ] Confirm routed-expert weight-transfer counter remains zero during decode.
- [ ] Benchmark PP and TG separately.
- [ ] Compare current layer placement against static per-expert placement.
- [ ] Measure RAM, VRAM and PCIe traffic.

## V1.6 — Production acceptance

- [ ] Validate startup, shutdown, unload and reload.
- [ ] Validate long-running server operation.
- [ ] Validate `--parallel 1`.
- [ ] Reject unsupported multi-GPU/RPC/backend combinations.
- [ ] Validate no-plan stock behavior and strict/non-strict failure modes.
- [ ] Document supported models, quants, backends and workflow.
- [!] Run final U3 package on Q4_K_M and Q5_K_M.

## Version 1 acceptance gate

- required CI and tests are green;
- output is correct within accepted numerical tolerance;
- no permanent routed-expert duplication;
- no routed-expert weight transfers during ordinary decode;
- CPU and CUDA experts execute on their assigned backend;
- load, unload and shutdown are stable;
- TurboQuant KV remains functional;
- TG regression is not greater than 5%;
- target TG improvement is at least 15%, or a measured bottleneck analysis explains why not;
- the full workflow is reproducible for both Q4_K_M and Q5_K_M.

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

- Fixed the Windows shared-library smoke test without exporting the internal placement dimension validator; both Windows and Ubuntu MoE CI jobs passed.
- Added immutable deep-copy placement snapshots at the model-loading boundary.
- Added compact tensor pool planning with exact CPU/GPU shapes, bytes and coalesced source-copy spans.
- Added exact adapters from real `ggml_tensor` metadata into compact source layouts.
- Added a loader contract that claims packed tensors as selective slice sources without adding them to standard model buffers.
- Added model-owned compact CPU and CUDA storage with correct buffer-before-context teardown.
- Added direct quantized GGUF slice loading for Qwen3.5/3.6 MoE separate and merged gate-up layouts.
- Removed persistent packed routed tensor allocation from the compact model-load path.
- Added `llama-moe-load-check` with exact byte accounting, zero-packed-pointer verification and load/unload markers.
- Added `run-u2.ps1` with RAM, VRAM, stdout/stderr and machine-readable result collection.
- Built the U2 validator successfully on Windows and Ubuntu.
- Blocked ordinary server inference with compact pools until the mixed execution graph exists in V1.3.
- Added mixed, all-GPU, invalid-size, invalid-axis and duplicate-destination storage tests.
- Accepted Q5_K_M U2 with exact CPU/GPU/total bytes, 240 compact tensors, zero packed routed tensors and successful unload.
- Diagnosed the Q4_K_M U2 archive as a Q4 model paired with the Q5 plan; compact pools and zero packed tensors were still confirmed.
- Added early model fingerprint rejection before backend initialization/allocation to prevent future Q4/Q5 plan mix-ups.
- Kept Q4_K_M and Q5_K_M as equal supported targets for final U3.
- Added and connected the V1.3 route-partition host reference with all-CPU, all-GPU and mixed slot-preservation tests on Windows and Ubuntu.

## 2026-07-27

- Accepted U1 for Q5_K_M and Q4_K_M on RTX 5070 Ti.
- Added exact Q5/Q4 placement comparison in `docs/moe-u1-results.md`.
- Started V1.2 with deterministic global-to-local CPU/GPU location tables and byte accounting.
- Kept user involvement deferred until U2.
