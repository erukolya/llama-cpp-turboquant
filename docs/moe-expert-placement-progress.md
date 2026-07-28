# MoE expert placement implementation progress

This file is the source of truth for **Version 1: complete static per-expert MoE placement**.

Version 2, adaptive rebalancing, LRU/LFU caching, miss loading and NVMe tiers are out of scope until Version 1 is complete, benchmarked and accepted.

Detailed design and results:

- `docs/moe-expert-static-placement-spec.md`
- `docs/moe-expert-static-placement-design-v1.md`
- `docs/moe-expert-placement-user-checks.md`
- `docs/moe-u1-results.md`
- `docs/moe-u2-results.md`
- `docs/moe-u3-package-manifest.json`
- `docs/moe-u3-final-package.sha256`

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

The first U3 hardware run exposed a graph-construction bug in static routing: `ggml_argsort_top_k` can return a non-contiguous expert-ID tensor, while the static graph attempted to flatten it directly through `ggml_reshape_1d`. The production path now materializes that small I32 route tensor once with `ggml_cont`, then shares the flattened result between the CPU and CUDA route-map branches.

A focused regression test using real `ggml_argsort_top_k -> ggml_cont -> reshape -> GET_ROWS` is green. The verified production fix is committed as `c929e579fbe5d459a53a9a00af2611aff954da6c`.

The corrected Windows CUDA 13.3 SM120 U3 artifact is being built in workflow run `30369033475`. Do not run the previous U3 package again. The next user action is one Q4_K_M and one Q5_K_M run only after the corrected artifact has completed and been verified.

Hardware checkpoints:

1. **U1 — complete:** exact model/plan dry run on Q5_K_M and Q4_K_M.
2. **U2 — complete for both quants:** Q5_K_M previously accepted; the uploaded Q4_K_M U3 archive proved exact CPU/GPU/total accounting, 240 compact tensors, zero packed tensors and successful model unload.
3. **U3 — corrected artifact building:** end-to-end baseline/static output, TurboQuant KV, backend execution, zero weight-transfer and memory checks on Q4_K_M and Q5_K_M.

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

**Current phase:** V1.3 implementation corrected; final U3 hardware acceptance pending the rebuilt artifact.

Qwen3.5/3.6 MoE claims packed routed tensors only as GGUF slice sources, allocates model-owned compact CPU and CUDA pools, and loads quantized expert slices without creating persistent packed runtime tensors. The router and top-k selection run once. The selected global IDs are materialized as one contiguous I32 tensor, backend-local IDs are obtained from immutable route maps, CPU and CUDA compact pools execute separate expert branches, and their weighted contributions are joined once with the result retained on the GPU side.

The server accepts `--moe-expert-plan` for real inference. Dry-run remains a separate stock-packed validation mode. Static placement disables stock all-expert warmup and exposes these Prometheus counters:

- CPU and accelerator `MUL_MAT_ID` executions;
- selective routed-weight bytes and payload bytes;
- expert slices, grouped copy calls and packed weight inputs.

U3 uses `K=turbo4`, `V=turbo3`, flash attention, `--parallel 1`, deterministic sampling and exact baseline/static output comparison. It requires both execution counters to be positive and every routed-weight copy counter to remain exactly zero.

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

## Accepted U2 results

| Metric | Q5_K_M | Q4_K_M |
|---|---:|---:|
| Exact compact accounting | accepted | accepted |
| Packed runtime tensors | 0 | 0 |
| Compact CPU tensors | 120 | 120 |
| Compact GPU tensors | 120 | 120 |
| Model unload marker | present | present |

Q4_K_M exact values from the U3 archive:

- CPU experts: 4,515;
- GPU experts: 5,725;
- CPU logical bytes: 8,663,654,400;
- GPU logical bytes: 10,839,859,200;
- total routed logical bytes: 19,503,513,600.

The Q4 checker emitted all acceptance markers and then faulted in global Windows CUDA backend cleanup. The one-shot checker no longer calls that global teardown after model-owned buffers have already been released. The U3 runner records the numeric exit code and requires all semantic U2 markers.

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
- [x] Q4_K_M exact U2 accounting, zero packed tensors and unload accepted from the uploaded U3 archive.
- [x] Remove the one-shot Windows CUDA global-cleanup false failure.

**V1.2 exit gate: passed for Q5_K_M and Q4_K_M.**

## V1.3 — Correct mixed execution

- [x] Split router/top-k generation from expert execution while retaining the stock wrapper.
- [x] Partition original top-k by immutable per-layer location maps.
- [x] Remap global IDs to local CPU/GPU IDs without changing top-k slot positions.
- [x] Materialize non-contiguous `argsort_top_k` IDs once before flattening and route-map lookup.
- [x] Preserve the original router weights for both branches.
- [x] Execute compact CPU and GPU expert branches separately.
- [x] Support separate and merged gate-up projection paths.
- [x] Preserve shared experts unchanged.
- [x] Handle arbitrary per-token CPU/GPU top-k splits with `-1` missing slots.
- [x] Guard `MUL_MAT_ID` missing-slot behavior without changing stock execution.
- [x] Implement the guarded path in generic CPU, x86 repack and CUDA backends.
- [x] Join CPU and GPU weighted contributions exactly once.
- [x] Add host slot/weight/sum tests on Windows and Ubuntu.
- [x] Add real backend I32 remap and guarded `MUL_MAT_ID(-1)` tests.
- [x] Add focused non-contiguous `argsort_top_k` route-remap regression test.
- [!] End-to-end baseline/static output equality and real CUDA execution: corrected U3 artifact required.

**V1.3 implementation gate: passed after U3 graph fix. Hardware correctness gate: pending.**

## V1.4 — Parallel CPU/CUDA execution

- [ ] Add synthetic fork-and-join timing test if U3 measurements require it.
- [!] Determine scheduler overlap behavior on the real mixed graph during U3.
- [ ] Use pinned activation/result buffers and asynchronous events if measurements require them.
- [ ] Avoid unnecessary per-layer synchronization if measurements expose it.
- [ ] Add CPU/GPU/join timing after functional U3 acceptance.
- [ ] Confirm mixed time approaches `max(CPU, GPU)` rather than their sum.

## V1.5 — Integration and metrics

- [x] Integrate static placement with `llama-server` and CLI parsing.
- [x] Preserve a separate stock-packed dry-run validator.
- [x] Disable all-expert warmup for static placement.
- [x] Expose CPU/accelerator execution and routed-weight copy counters through `/metrics`.
- [!] Confirm routed-expert weight-transfer counters remain zero during real decode: U3 required.
- [!] Validate TurboQuant KV `turbo4/turbo3` with flash attention: U3 required.
- [!] Benchmark PP and TG through U3.
- [!] Measure RAM and VRAM through U3.

## V1.6 — Production acceptance

- [!] Validate startup, generation, shutdown and unload through the corrected U3 runner.
- [ ] Validate long-running server operation after functional U3 acceptance.
- [x] Implement and test `--parallel 1` U3 configuration.
- [ ] Reject additional unsupported multi-GPU/RPC/backend combinations explicitly where needed.
- [!] Validate no-plan stock behavior and strict plan mode through U3.
- [x] Document supported models, quants, backends and workflow.
- [~] Build the corrected Windows CUDA 13.3 SM120 U3 artifact in run `30369033475`.
- [!] Run corrected U3 on Q4_K_M and Q5_K_M.

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
- Accepted Q4_K_M U2 from the uploaded U3 archive with exact CPU/GPU/total bytes, 240 compact tensors, zero packed routed tensors and successful model unload.
- Diagnosed and removed the one-shot checker global CUDA cleanup false failure after accepted model unload.
- Split MoE routing/top-k generation from expert execution.
- Added model-owned CPU/GPU route-map tensors and direct compact tensor pointers on each layer.
- Added mixed CPU/CUDA expert branches with slot-preserving global-to-local ID remap.
- Added guarded negative route slots to generic CPU, x86 repack and CUDA `MUL_MAT_ID` paths.
- Added Windows and Ubuntu host partition, backend I32 remap and guarded missing-slot tests.
- Fixed Release tests that incorrectly depended on side effects inside `assert()`.
- Enabled real static-plan server inference and disabled stock all-expert warmup for static placement.
- Exposed MoE execution and routed-weight transfer counters through `/metrics`.
- Built the first Windows CUDA 13.3 SM120 U3 package and used it to reach real static graph construction.
- Diagnosed `GGML_ASSERT(ggml_is_contiguous(a))` as a direct reshape of non-contiguous `argsort_top_k` IDs.
- Materialized one shared contiguous I32 route tensor before CPU/GPU route-map remap.
- Added and passed a focused `argsort_top_k -> ggml_cont -> reshape -> GET_ROWS` regression test.
- Started corrected Windows CUDA 13.3 SM120 artifact build in run `30369033475` from verified commit `c929e579fbe5d459a53a9a00af2611aff954da6c`.


## U3 Q5 MMVQ missing-ID diagnosis (2026-07-28)

- Q5 U2 compact accounting and the stock baseline passed.
- Static inference loaded successfully and failed on the first CUDA graph with `illegal memory access`.
- Root cause: the optimized quantized MMVQ kernels read the split-route `-1` sentinel as an unsigned expert index (`UINT_MAX`).
- Fix: keep missing slots mapped to a safe zero index for reads and suppress their writes; the destination is already zeroed by guarded `MUL_MAT_ID`.
- Both single-token decode and multi-token prompt MMVQ paths are covered.
- U3 PowerShell now captures the actual U2 process exit code instead of serializing `$null` as zero.
- Corrected CUDA 13.3 / SM120 package is being built by this workflow.
- **USER NOT NEEDED** until the corrected artifact is published.


## U3 Q5 checker startup diagnosis (2026-07-28)

- The MMVQ-corrected package failed before opening the model.
- `llama-moe-load-check.exe` aborted in `ggml.cpp` because the static `std::terminate` hook was registered twice.
- The registration is now idempotent: an already-installed identical handler is accepted without replacing its predecessor.
- U3 child processes also set `GGML_NO_BACKTRACE=1` as a defensive workaround.
- Windows CI now executes `llama-moe-load-check.exe --help` before packaging and rejects any checker that cannot start.
- **USER NOT NEEDED** until the replacement package passes the startup smoke test and is published.


## U3 Q4/Q5 hardware acceptance complete (2026-07-28)

The corrected Windows CUDA 13.3 / SM120 runtime passed end-to-end static mixed CPU/CUDA execution on both target quantizations.

### Q5_K_M

- model fingerprint: `sampled-fnv1a64:66d21554683500ba`;
- experts: 5,377 CPU + 4,863 GPU = 10,240;
- compact tensors: 120 CPU + 120 GPU;
- exact routed bytes: 12,019,130,368 CPU + 10,839,826,432 GPU = 22,858,956,800;
- packed routed runtime tensors: `0`;
- CPU `MUL_MAT_ID` operations: 1,200;
- accelerator `MUL_MAT_ID` operations: 1,200;
- routed-weight copy bytes/payload/slices/calls/inputs: all `0`;
- baseline decode: 28.5714 tok/s;
- static decode: 39.2157 tok/s (`+37.26%` for this short deterministic probe).

### Q4_K_M

- model fingerprint: `sampled-fnv1a64:d1859e626beb3c0e`;
- experts: 4,515 CPU + 5,725 GPU = 10,240;
- compact tensors: 120 CPU + 120 GPU;
- exact routed bytes: 8,663,654,400 CPU + 10,839,859,200 GPU = 19,503,513,600;
- packed routed runtime tensors: `0`;
- CPU `MUL_MAT_ID` operations: 1,200;
- accelerator `MUL_MAT_ID` operations: 1,200;
- routed-weight copy bytes/payload/slices/calls/inputs: all `0`;
- baseline decode: 34.3348 tok/s;
- static decode: 43.0108 tok/s (`+25.27%` for this short deterministic probe).

### Validator correction

- `llama-server` may suppress detailed compact-loader INFO lines; strict plan scope plus the exact U2 semantic gate is accepted as proof of compact loading and zero packed runtime tensors.
- Full greedy text equality across CPU-only and mixed CPU/CUDA backends is retained as a diagnostic only. Quantized CPU and CUDA matmul can diverge after several tokens because accumulation order differs.
- The deterministic acceptance probe requires the first generated lexical unit to match; both hardware runs matched `Paris.` with a seven-character common prefix.

**V1.3 hardware gate is complete. USER NOT NEEDED.**
