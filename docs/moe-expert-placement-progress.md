# MoE expert placement implementation progress

This file is the source of truth for **Version 1: complete static per-expert MoE placement**.

Version 2, adaptive rebalancing, LRU/LFU caching, miss loading and NVMe tiers are out of scope until Version 1 is complete, benchmarked and accepted.

Detailed design:

- `docs/moe-expert-static-placement-spec.md`
- `docs/moe-expert-static-placement-design-v1.md`
- `docs/moe-expert-placement-user-checks.md`
- `docs/moe-u1-results.md`

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

The next user action is **U2**, only after compact CPU/CUDA pools are implemented and a ready Windows CUDA artifact exists.

Planned hardware checkpoints:

1. **U1 — complete:** exact model/plan dry run on Q5_K_M and Q4_K_M.
2. **U2 — pending:** exclusive split-loader RAM/VRAM validation.
3. **U3 — pending:** final correctness, PCIe and performance validation.

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

**Current phase:** V1.2 exclusive split storage and selective loading.

**Current decision:** the exact model-layout gate is closed; implementation may proceed to loader integration and compact pools.

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
- [~] Scheduler/public API counters for selective weight copies and CPU/accelerator `MUL_MAT_ID`; implemented, latest CI pending.

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

**Status:** active.

- [~] Design ownership and lifecycle of CPU/GPU expert pools.
- [x] Locate Qwen3.5 MoE packed tensor creation in `src/models/qwen35moe.cpp`.
- [x] Confirm one `build_moe_ffn` graph path.
- [x] Identify pre-allocation model-loading integration point.
- [x] Identify runtime tensors without source weights as the compact-pool extension point.
- [~] Add deterministic per-layer global-to-local CPU/GPU location tables; committed, CI pending.
- [ ] Move validated placement access to the model-loading boundary.
- [ ] Create compact CPU routed-expert pools.
- [ ] Create compact CUDA routed-expert pools.
- [ ] Support separate and merged gate-up tensors.
- [ ] Copy quantized slices without dequantization.
- [ ] Validate quant-block and backend alignment.
- [ ] Read slices directly from GGUF/mmap/staging.
- [ ] Avoid allocating the original persistent packed routed tensors.
- [ ] Prove no complete RAM expert bank remains.
- [ ] Add all-CPU, all-GPU and mixed loading tests.
- [ ] Add exact RAM/VRAM accounting.
- [!] Run U2 on the primary machine.

**V1.2 exit gate:** compact pools load and unload correctly, permanent CPU+VRAM routed-expert bytes approximately equal the original routed bank, and no full RAM duplicate remains.

## V1.3 — Correct mixed execution

- [ ] Partition original top-k by location table.
- [ ] Remap global IDs to local CPU/GPU pool IDs.
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
- [!] Run final U3 package.

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
- the full workflow is reproducible.

# Explicitly deferred

Do not implement on the current branch:

- periodic/runtime expert rebalancing;
- EWMA migration scores;
- LRU/LFU cache;
- cache-miss weight loading;
- RAM/VRAM swaps during server operation;
- NVMe tier.

# Change log

## 2026-07-27

- Accepted U1 for Q5_K_M and Q4_K_M on RTX 5070 Ti.
- Added exact Q5/Q4 placement comparison in `docs/moe-u1-results.md`.
- Started V1.2 with deterministic global-to-local CPU/GPU location tables and byte accounting.
- Kept user involvement deferred until U2.
