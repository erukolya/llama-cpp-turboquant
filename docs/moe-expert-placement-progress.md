# MoE expert placement implementation progress

This file is the implementation tracker and source of truth for the static/adaptive per-expert MoE placement work.

Detailed requirements:

- `docs/moe-expert-static-placement-spec.md`
- `docs/moe-expert-static-placement-design-v1.md`

Active branches and pull requests:

- profiling branch: `agent/moe-expert-stats`
- profiling PR: https://github.com/erukolya/llama-cpp-turboquant/pull/1
- implementation branch: `agent/static-moe-expert-placement`
- implementation draft PR: https://github.com/erukolya/llama-cpp-turboquant/pull/2

## Status legend

| Marker | Meaning |
|---|---|
| `[x]` | Completed and committed |
| `[~]` | In progress |
| `[!]` | Blocked or waiting on a required gate |
| `[ ]` | Not started |

## Project invariants

These requirements are mandatory for the final implementation:

- hot routed expert weights are stored only in VRAM;
- cold routed expert weights are stored only in RAM;
- the complete routed-expert bank must not remain duplicated in RAM;
- ordinary decode must not transfer routed-expert weights over PCIe;
- GPU experts execute on CUDA;
- CPU experts execute on CPU;
- a mixed top-k is split into CPU and CUDA assignments;
- CPU and CUDA branches must overlap where possible;
- router scores, top-k selection, weights, quantization and model semantics are unchanged;
- the feature is disabled by default until correctness and performance gates pass;
- stock behavior without the feature flag must remain unchanged.

## Current project state

**Current phase:** V1.1 / model-aware dry-run validation.

**Current decision:** continue work that does not modify model allocation or inference. Do not start the split loader until the current C++/Python CI is green and model-aware validation is complete.

**Confirmed baseline behavior:** current host-resident `MUL_MAT_ID` offload can inspect top-k IDs and copy selected expert weight slices from host memory to CUDA during inference. The new CPU pool must explicitly avoid this operation-offload path.

**Measured planner result for the KAT profile:**

| Metric | Value |
|---|---:|
| Logical routed experts | 10,240 |
| Existing VRAM expert count | 4,864 |
| Selected by byte-aware plan | 4,863 |
| Routed-expert VRAM budget | 10,338 MiB |
| Used by generated plan | 10,337.664 MiB |
| Current estimated GPU selection coverage | about 47.5% |
| Planned estimated GPU selection coverage | about 71.242% |

These percentages are routing-selection coverage, not expected tokens/s improvement.

---

# Version 1 — static placement

## P0 — Requirements and architecture

- [x] Write complete technical specification.
- [x] Separate Version 1 and Version 2 requirements.
- [x] Record exclusive residency requirement.
- [x] Record prohibition on decode-path weight streaming.
- [x] Record correctness, memory, PCIe and benchmark acceptance criteria.
- [x] Collect implementation references from llama.cpp RFCs/forks, ik_llama.cpp, KTransformers and Colibrì.

**Artifacts:**

- `docs/moe-expert-static-placement-spec.md`
- `docs/moe-expert-static-placement-design-v1.md`

## P1 — Existing profiler and placement data

- [x] Collect per `(layer, expert)` routing statistics.
- [x] Support exact and sampled profiling.
- [x] Export placement report.
- [x] Confirm KAT topology: 40 layers, 256 routed experts, top-8.
- [x] Correct interpretation of `layer_share`.
- [x] Extend placement report with exact `size_bytes`.
- [x] Extend placement report with `n_experts`.
- [x] Extend placement report with `expert_stride_bytes = nb[2]`.

**Gate:** profiling PR must remain independently reviewable.

## P2 — Confirm current execution path

- [x] Locate model loader tensor-allocation path.
- [x] Locate `build_moe_ffn` and `build_lora_mm_id`.
- [x] Locate scheduler backend assignment.
- [x] Confirm weight-buffer backend preference.
- [x] Confirm operation offload can override CPU residency.
- [x] Confirm active-expert host-to-CUDA copies in `ggml_backend_sched_compute_splits`.
- [x] Document why tensor views cannot provide independent residency.
- [ ] Add optional runtime instrumentation for copied weight bytes and execution backend.

## V1.1 — Offline placement planning

- [x] Add profile CSV parser.
- [x] Add placement CSV parser.
- [x] Compute logical expert bytes from gate/up/down strides.
- [x] Rank experts by `estimated_hits / size_bytes`.
- [x] Select experts under a byte-accurate VRAM budget.
- [x] Treat missing sampled profile rows as zero-hit with low confidence.
- [x] Write versioned JSON plan.
- [x] Write complete per-expert CSV report.
- [x] Add deterministic tie-breaking.
- [x] Add Python unit tests.
- [x] Validate planner with real KAT CSV files.

**Artifacts:**

- `tools/moe-plan/moe_plan.py`
- `tools/moe-plan/test_moe_plan.py`

## V1.1 — C++ plan contract

- [x] Add reusable C++ plan types.
- [x] Add strict JSON parser.
- [x] Add structural validation.
- [x] Add deterministic JSON writer/round trip.
- [x] Validate budgets, counts, sorted IDs and hit totals.
- [x] Add `llama-moe-plan-check`.
- [x] Add C++ unit tests.
- [~] Validate all current CI configurations.

**Current CI gate:**

- [x] Server workflow passed at the last check.
- [~] CPU CI running/queued.
- [~] Python type-check running/queued.
- [~] flake8 running/queued.
- [~] self-hosted CI running/queued.

Do not manually mark these complete without checking the workflow result for the current branch head.

## V1.1 — Model-aware dry run

- [~] Define routed tensor manifest in the placement plan.
- [ ] Add stable model fingerprint based on GGUF metadata and tensor directory.
- [ ] Enumerate actual routed expert tensors from the loaded model.
- [ ] Validate tensor names.
- [ ] Validate `ggml_type`.
- [ ] Validate `ne[]`.
- [ ] Validate `nb[]`.
- [ ] Validate routed expert axis.
- [ ] Validate `nb[2]` expert stride.
- [ ] Validate all gate/up/down components of a logical expert share the same tier.
- [ ] Detect merged `gate_up_exps` layouts.
- [ ] Detect per-expert scale and bias tensors that must follow placement.
- [ ] Add `--moe-expert-plan FNAME`.
- [ ] Add `--moe-expert-plan-strict`.
- [ ] Add `--moe-expert-plan-dry-run`.
- [ ] Print per-layer intended CPU/VRAM expert counts and bytes.
- [ ] Reject unsupported model/layout without changing allocation.
- [ ] Add tests for fingerprint mismatch and unsupported layout.

**Exit gate for V1.1:**

- current CI green;
- plan generated by Python is accepted by C++;
- loaded model manifest matches the plan;
- dry run changes no allocation and no inference result;
- unsupported layouts fail explicitly in strict mode.

## V1.2 — Split expert storage and selective loading

**Status:** blocked until V1.1 exit gate.

- [!] Design ownership and lifecycle of CPU/GPU expert pools.
- [ ] Add per-layer global-to-local expert location tables.
- [ ] Create compact CPU gate/up/down pools.
- [ ] Create compact CUDA gate/up/down pools.
- [ ] Support merged gate/up tensors.
- [ ] Copy quantized expert slices without dequantization.
- [ ] Validate quant-block and backend alignment.
- [ ] Load slices directly from GGUF/mmap/staging where possible.
- [ ] Release original persistent packed routed-expert representation.
- [ ] Prove no complete RAM expert bank remains.
- [ ] Add all-CPU, all-GPU and mixed placement load tests.
- [ ] Add memory accounting and placement report.

**Exit gate for V1.2:**

- split pools load successfully;
- combined permanent CPU+VRAM expert bytes approximately equal original routed-expert bytes;
- no persistent full-bank duplication;
- inference path may still use a correctness/reference implementation, but model unload/reload is correct.

## V1.3 — Correct mixed CPU/GPU execution

- [ ] Partition original top-k assignments by location table.
- [ ] Remap global expert IDs to local CPU/GPU pool IDs.
- [ ] Preserve token index and router weight for each assignment.
- [ ] Execute CPU pool experts only on CPU.
- [ ] Execute GPU pool experts only on CUDA.
- [ ] Prevent scheduler offload of CPU-pool `MUL_MAT_ID`.
- [ ] Implement gate/up activation path for split assignments.
- [ ] Implement down projection path for split assignments.
- [ ] Join partial results exactly once.
- [ ] Preserve shared expert behavior.
- [ ] Preserve optional expert scales/biases.
- [ ] Handle 0–8 GPU experts in a top-8.
- [ ] Handle different splits for different tokens in one ubatch.
- [ ] Add tensor-level correctness tests.
- [ ] Compare logits and perplexity with baseline.

**Exit gate for V1.3:** correctness passes before optimization.

## V1.4 — Parallel CPU/CUDA execution

- [ ] Build synthetic CPU/GPU fork-and-join graph test.
- [ ] Determine whether standard scheduler overlaps branches.
- [ ] Choose scheduler-level split or dedicated hybrid operation.
- [ ] Use pinned activation/result buffers.
- [ ] Use asynchronous copies and CUDA events.
- [ ] Avoid global device synchronization per MoE layer.
- [ ] Bypass CPU branch when all selected experts are in VRAM.
- [ ] Bypass CUDA branch when all selected experts are in RAM.
- [ ] Add CPU compute, GPU compute and join-wait timers.
- [ ] Confirm elapsed time approaches `max(CPU, GPU)` rather than their sum.

## V1.5 — Integration, metrics and benchmarks

- [ ] Integrate with `llama-server`.
- [ ] Integrate with `llama-cli`.
- [ ] Integrate with `llama-bench`.
- [ ] Preserve TurboQuant KV-cache functionality.
- [ ] Add routing hit-rate metrics.
- [ ] Add activation transfer counters.
- [ ] Add expert-weight migration counter.
- [ ] Confirm weight migration counter remains zero after model load.
- [ ] Benchmark prompt processing separately from token generation.
- [ ] Benchmark current layer placement versus static per-expert placement.
- [ ] Measure RAM and VRAM.
- [ ] Measure PCIe traffic with internal counters and Nsight where possible.
- [ ] Produce reproducible benchmark report.
- [ ] Keep the feature disabled by default until acceptance gates pass.

**Version 1 acceptance gate:**

- correctness accepted;
- no persistent expert duplication;
- no routed-expert weight transfer during normal decode;
- actual GPU selection coverage matches the plan within expected profiling error;
- no token-generation regression greater than 5%;
- target token-generation improvement is at least 15% on the primary KAT workload, or a documented bottleneck analysis explains why not.

---

# Version 2 — adaptive rare rebalancing

Version 2 begins only after Version 1 is stable and benchmarked.

## V2.1 — Low-overhead live heat tracking

- [ ] Collect routing counts without callback-induced GPU synchronization.
- [ ] Track lifetime hits.
- [ ] Track recent-window hits.
- [ ] Track EWMA score.
- [ ] Track prompt and decode statistics separately.
- [ ] Track residency age and migration count.

## V2.2 — Rebalance planner

- [ ] Compute candidate placement from live EWMA heat.
- [ ] Use score per byte.
- [ ] Add minimum relative gain threshold.
- [ ] Add minimum residency period.
- [ ] Add cooldown.
- [ ] Add maximum swaps per cycle.
- [ ] Prevent placement thrashing.

Default target settings:

```text
rebalance interval:       60 minutes
minimum residency:       120 minutes
minimum relative gain:    20%
maximum swaps per cycle:  64
```

## V2.3 — Idle-time transactional migration

- [ ] Rebalance only between requests/all slots idle initially.
- [ ] Add temporary staging buffer.
- [ ] Swap GPU and CPU experts transactionally.
- [ ] Use backend events for completion.
- [ ] Atomically publish a new location-table generation.
- [ ] Roll back on migration failure.
- [ ] Remove transient duplicates after commit.
- [ ] Persist new plan atomically.

## V2.4 — Operational integration

- [ ] Add manual rebalance trigger.
- [ ] Add placement and statistics endpoint or command.
- [ ] Add rebalance metrics.
- [ ] Load latest valid persisted plan at startup.
- [ ] Detect corrupt/incompatible persisted plan.
- [ ] Benchmark adaptation across different workloads.

**Version 2 acceptance gate:**

- no migration during active requests in idle-only mode;
- failed migration rolls back safely;
- permanent placement remains exclusive;
- weight transfers occur only during explicit rebalance;
- hysteresis prevents repeated swaps;
- restart restores the last valid placement generation.

---

# Update policy

Every implementation commit that completes, starts or blocks a tracked item must update this file in the same commit or the immediately following progress-only commit.

When updating:

1. change the checklist marker;
2. update `Current project state`;
3. record any new blocking decision;
4. update the relevant exit gate;
5. add a dated entry to the change log below.

Do not mark an item complete solely because code was written. It is complete only when its stated tests or gate pass.

# Change log

## 2026-07-27

- Created the progress tracker.
- Marked specification, execution-path analysis, offline planner and C++ plan contract as completed.
- Marked CI validation and model-aware dry run as active.
- Blocked split-pool allocation until CI and model-aware validation gates pass.
