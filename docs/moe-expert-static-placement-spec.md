# Static MoE expert placement between CPU RAM and CUDA VRAM

Status: implementation specification

Target repository: `erukolya/llama-cpp-turboquant`

Base branch for this work: `agent/moe-expert-stats`

Primary target for V1: Windows, one CUDA GPU, CPU backend, `llama-server`, KAT-Coder-V2.5-Dev Q5_K_M.

## 1. Goal

Replace coarse routed-MoE placement by whole layers with placement of individual logical experts:

```text
(layer_index, expert_index) -> CPU RAM or CUDA VRAM
```

The placement is generated from an offline routing profile collected by the existing MoE profiler.

The first implementation is static for the whole process lifetime. A later implementation may rebalance a limited number of experts between requests every 30-60 minutes.

## 2. Non-negotiable requirements

### 2.1 Exclusive residency

After model loading completes:

```text
hot expert  -> VRAM only
cold expert -> RAM only
```

The accepted implementation must not keep the full routed-expert bank in RAM while also keeping hot copies in VRAM.

Temporary duplication is allowed only during model loading or an atomic V2 migration transaction. Temporary staging buffers must be released after commit.

### 2.2 No expert-weight transfer in the normal decode path

The normal V1 execution path must be:

```text
VRAM expert -> CUDA compute
RAM expert  -> CPU compute
```

It must not be:

```text
RAM expert weights -> PCIe -> GPU -> compute
```

Activation and partial-output transfers are allowed. Expert-weight transfers after startup are not allowed in V1.

### 2.3 Parallel CPU and GPU execution

A mixed top-k must be partitioned into CPU and GPU assignments and computed concurrently.

Example:

```text
router top-8: [254, 87, 100, 42, 15, 203, 9, 71]
GPU:          [254, 87, 42, 203, 71]
CPU:          [100, 15, 9]
```

The target layer latency is approximately:

```text
max(T_gpu, T_cpu + activation_transfer) + T_join
```

not:

```text
T_gpu + T_cpu
```

### 2.4 Preserve model semantics

Do not change:

- router logits;
- top-k selection;
- router weights;
- routed-expert activation functions;
- shared experts;
- quantization type;
- expert outputs or reduction order unless required by backend scheduling;
- quality-related sampling parameters.

Placement is a memory and execution optimization only.

## 3. Model structure

The current test model is `Kwaipilot/KAT-Coder-V2.5-Dev-Q5_K_M.gguf`.

Relevant architecture:

```text
40 transformer layers
256 routed experts per layer
top-8 routed experts per token per layer
2048 hidden size
512 routed-expert intermediate size
512 shared-expert intermediate size
```

This means there are:

```text
40 * 256 = 10240 distinct logical routed experts
```

The same numeric expert ID in different layers refers to different weights.

One logical expert consists of three slices:

```text
blk.N.ffn_gate_exps.weight[E]
blk.N.ffn_up_exps.weight[E]
blk.N.ffn_down_exps.weight[E]
```

The router ID is global only inside one layer. Runtime placement must therefore use `(layer, expert)`.

The shared expert is not part of routed IDs `0..255` and is out of scope for initial splitting.

## 4. Existing profiling work

The branch `agent/moe-expert-stats` already adds:

```text
--moe-stats
--moe-stats-file FNAME
--moe-stats-sample N
--moe-stats-exact
--moe-placement
--moe-placement-file FNAME
```

The existing profiler observes `ffn_moe_topk`, supports sampled counting and produces records such as:

```csv
layer,expert,hits,sampled_hits,layer_share,coverage,sample_rate
38,254,18408,435,0.033277234,0.023630474,32
```

Keep this functionality working. Core placement and execution must not remain a `tools/server`-only hack. The final implementation must be usable by `llama-server`, `llama-cli` and `llama-bench`.

## 5. Current measured baseline

Current routed-expert placement:

```text
layers 0-18  -> VRAM
layers 19-39 -> RAM
```

Current count:

```text
VRAM: 4864 logical experts
RAM:  5376 logical experts
```

Current routed-expert storage observed from the GGUF/runtime report:

```text
VRAM routed experts: about 10338 MiB
RAM routed experts:  about 11462 MiB
Total:               about 21800 MiB
```

Current layer-based placement serves approximately `47.5%` of measured routed-expert selections from VRAM.

A byte-aware offline selection of the hottest `(layer, expert)` pairs under the same routed-expert VRAM budget is estimated to serve about `71.2%` of measured selections from VRAM.

This estimate is not a guaranteed tokens/s improvement. It predicts a reduction of CPU expert selections, not total inference time.

## 6. Version 1: static profile-based placement

### 6.1 V1 workflow

```text
run A:
    collect routing profile

plan step:
    load model metadata and profile
    compute expert byte sizes
    choose hot experts under a VRAM budget
    write a deterministic placement plan

run B:
    load placement plan
    construct exclusive GPU and CPU expert pools
    run mixed CPU/GPU MoE without weight migration
```

### 6.2 Offline planner

Add a separate tool, preferred name:

```text
llama-moe-plan
```

Example:

```powershell
.\llama-moe-plan.exe `
  --model model.gguf `
  --profile kat-moe.csv `
  --vram-budget-mib 10338 `
  --output kat-static-plan.json `
  --report kat-static-plan.csv
```

Default ranking score:

```text
estimated_hits / expert_size_bytes
```

Do not assume every expert has the same byte size. Use actual tensor layout and quantization metadata.

The planner must verify that the profile matches the model. Create a model fingerprint from normalized tensor metadata and relevant GGUF architecture metadata. Hashing all weight bytes is not required.

### 6.3 Placement plan format

Use a versioned JSON document. Minimum fields:

```json
{
  "schema_version": 1,
  "model_fingerprint": "sha256:...",
  "strategy": "hits_per_byte",
  "vram_budget_bytes": 0,
  "estimated_total_hits": 0,
  "estimated_gpu_hits": 0,
  "estimated_gpu_hit_rate": 0.0,
  "layers": [
    {
      "layer": 0,
      "gpu_experts": [1, 4, 7],
      "gpu_bytes": 0,
      "estimated_gpu_hit_rate": 0.0
    }
  ]
}
```

The CPU list is the complement of `gpu_experts`.

The plan generation must be deterministic for identical inputs.

### 6.4 Runtime CLI

Minimum V1 runtime options:

```text
--moe-expert-plan FNAME
--moe-expert-plan-strict
--moe-expert-placement-report FNAME
```

Expected behavior:

- no new option: preserve stock behavior;
- invalid plan with strict mode: fail startup with a clear error;
- invalid plan without strict mode: warn and use stock placement;
- unsupported tensor layout or quant type: explicit error or explicit fallback;
- never silently run a partially applied plan.

### 6.5 Runtime data structures

Recommended conceptual structures:

```cpp
enum class moe_expert_backend : uint8_t {
    cpu,
    cuda,
};

struct moe_expert_location {
    moe_expert_backend backend;
    uint16_t local_index;
};

struct moe_layer_location_table {
    std::vector<moe_expert_location> global_to_location;
    std::vector<uint16_t> gpu_to_global;
    std::vector<uint16_t> cpu_to_global;
    std::vector<uint16_t> global_to_gpu;
    std::vector<uint16_t> global_to_cpu;
};

struct moe_expert_pool {
    ggml_tensor * gate;
    ggml_tensor * up;
    ggml_tensor * down;
    ggml_backend_buffer_t buffer;
    ggml_backend_t backend;
    uint32_t expert_count;
};
```

The exact ownership must follow current llama.cpp model and backend conventions.

### 6.6 Selective loader

The source GGUF stores all experts of one layer inside packed tensors. The loader must construct two compact runtime pools per layer:

```text
GPU pool: hot experts
CPU pool: cold experts
```

Requirements:

1. Detect and validate the expert axis.
2. Use real `ne[]`, `nb[]`, tensor type, quant-block alignment and tensor offsets.
3. Never use guessed `tensor_size / num_experts` offsets without validating layout.
4. Keep gate, up and down slices of one logical expert in the same tier.
5. Preserve original quantization bytes and type.
6. Do not dequantize during placement.
7. Release the original persistent full packed routed-expert allocation after split succeeds.
8. Support Windows file mapping and lifecycle; do not rely only on Linux-specific `madvise` behavior.
9. Use a whitelist/fallback for unsupported layouts.

A tensor view is not sufficient because it keeps the original buffer alive.

### 6.7 Mixed routing execution

The router continues to produce original expert IDs. For every selected assignment:

```text
location = table[global_expert_id]
```

Create compact CPU and GPU assignment lists containing at least:

```text
token/row index
local expert index
router weight
```

Run only real assignments. A final implementation must not compute all CPU and GPU experts and mask unused outputs.

The two branches must produce weighted contributions and join exactly once.

Fast paths are required:

```text
all selected experts on GPU -> no CPU transfer or CPU work
all selected experts on CPU -> no CUDA dispatch
mixed                       -> parallel CPU/CUDA execution and join
```

### 6.8 Activation transfers

When dense state is on GPU and CPU experts are selected, transfer only:

```text
activation rows
selected IDs/weights if needed
partial result
```

Use reusable pinned buffers, asynchronous copies and CUDA events. Avoid device-wide synchronization.

After model load, V1 expert-weight migration counters must remain zero.

### 6.9 Interaction with fit and memory budgeting

Do not let expert placement evict critical dense/shared tensors or consume KV-cache reserve.

Priority must account for:

```text
dense/attention/router/shared tensors
KV cache and fit target
backend workspaces
graph buffers
TurboQuant buffers
safety reserve
then routed-expert hot pool
```

The first implementation may require an explicit routed-expert VRAM budget before integrating with automatic fit.

### 6.10 Metrics

Add counters for:

```text
moe_gpu_selections_total
moe_cpu_selections_total
moe_gpu_hit_rate
moe_cpu_experts_per_token_avg
moe_gpu_experts_per_token_avg
moe_cpu_compute_us
moe_gpu_compute_us
moe_join_wait_cpu_us
moe_join_wait_gpu_us
moe_activation_h2d_bytes
moe_activation_d2h_bytes
moe_weight_migration_bytes
```

Separate prompt-processing and token-generation metrics where practical.

### 6.11 Placement report

Add a per-expert runtime report:

```csv
layer,expert,tier,local_index,gate_bytes,up_bytes,down_bytes,total_bytes,estimated_hits,estimated_share
38,254,VRAM,17,...
38,100,RAM,91,...
```

Keep the existing packed-tensor placement report for diagnostics.

## 7. V1 implementation stages

### V1.0 Research and baseline

Before core code changes:

- locate current model loader, MoE graph builder, `MUL_MAT_ID` CPU/CUDA paths and scheduler;
- trace the current `CUDA_Host` expert path;
- determine whether CPU-resident expert weights are computed on CPU or copied to GPU under each relevant mode;
- capture PP/TG, CPU, GPU, RAM, VRAM and PCIe baseline;
- write a short design note with confirmed symbols and ownership.

Search starting points:

```bash
rg -n "ffn_moe_topk|build_moe_ffn|ggml_mul_mat_id|GGML_OP_MUL_MAT_ID" .
rg -n "cpu_moe|n_cpu_moe|override_tensor|tensor_buft" .
rg -n "graph_compute_async|backend_sched|sched.*graph" ggml src .
```

### V1.1 Planner only

Implement profile parsing, model fingerprinting, expert-size calculation, optimizer, JSON plan and CSV report. No runtime execution changes.

### V1.2 Split loader

Construct exclusive CPU/GPU pools and location tables. Validate memory ownership and eliminate the persistent full routed-expert bank.

### V1.3 Correctness execution

Partition assignments and produce correct outputs for every split from 0 GPU experts to all top-k experts on GPU.

### V1.4 Parallel execution

Add actual overlap, reusable transfer buffers, events, metrics and remove avoidable global synchronization.

### V1.5 Integration

Support server, CLI and bench; document fallback behavior; add Windows CUDA CI/build coverage; benchmark KAT workload.

## 8. V1 acceptance criteria

### Correctness

Test at least:

- all CPU;
- all GPU on a small test model;
- 1 GPU + 7 CPU;
- 4 GPU + 4 CPU;
- 7 GPU + 1 CPU;
- different splits for different rows in one ubatch;
- expert 0 and expert 255;
- missing profile records;
- invalid fingerprint;
- unsupported layout;
- shared expert unaffected.

Compare MoE block output, logits, perplexity and deterministic generation using backend-appropriate tolerances.

### Memory

After load:

```text
routed_expert_RAM_bytes + routed_expert_VRAM_bytes
approximately equals routed_expert_total_bytes
```

The implementation must not report a full RAM routed-expert bank plus an additional VRAM hot set.

### PCIe

After startup, normal token generation must show:

```text
expert weight transfer bytes == 0
```

Activation and result transfers are expected.

### Performance

Compare:

```text
current layer placement
static expert placement with the same routed-expert VRAM byte budget
all CPU routed experts
```

Measure PP and TG separately and include a real `llama-server` coding workload.

Target on the primary machine: at least 15% TG improvement. This is a target, not permission to hide a lower result.

If the target is missed, report CPU compute, GPU compute, join wait, activation transfer and scheduler serialization.

## 9. Version 2: rare adaptive rebalancing

V2 builds on the static V1 layout. It is not a per-request cache.

Runtime continues to collect low-overhead routing heat. Every 30-60 minutes, or on manual request, the server may replace a limited number of cold GPU experts with hotter CPU experts.

Normal inference still uses resident weights. PCIe weight transfer occurs only during a rebalance transaction.

### 9.1 Initial V2 safe point

Rebalance only:

```text
between requests
when all server slots are idle
```

Mid-request or mid-batch migration is out of the first V2 scope.

### 9.2 Statistics

Track:

```text
lifetime hits
recent-window hits
EWMA hits
last selected timestamp
current tier
residency start time
migration count
```

Recommended ranking:

```text
EWMA_hits / size_bytes
```

### 9.3 Anti-thrashing defaults

```text
rebalance interval:       60 minutes
minimum residency:        120 minutes
minimum relative gain:    20%
maximum swaps per cycle:  64
EWMA half-life:            configurable
```

A CPU candidate may replace a GPU expert only when:

```text
candidate_score > coldest_gpu_score * (1 + minimum_gain)
```

### 9.4 Swap transaction

With fixed-size CPU and GPU pool slots:

1. wait for idle/safe generation;
2. copy the cold GPU expert to temporary staging;
3. copy the hot CPU expert into the GPU slot;
4. copy staging into the released CPU slot;
5. wait for backend completion events;
6. atomically publish a new location-table generation;
7. persist the new plan;
8. release staging.

Transient duplication is allowed only inside the transaction.

On failure, keep the old location table and valid weights.

### 9.5 V2 options

Suggested CLI:

```text
--moe-expert-rebalance-minutes 60
--moe-expert-ewma-half-life-minutes 120
--moe-expert-min-residency-minutes 120
--moe-expert-min-gain 0.20
--moe-expert-max-swaps 64
--moe-expert-plan-save FNAME
--moe-expert-rebalance-idle-only
```

Add a manual trigger for testing, either a command or internal endpoint.

### 9.6 V2 persistence

Persist plans with temporary-file write, flush and atomic rename. Include model fingerprint, placement generation and last rebalance timestamp.

## 10. Out of scope

V1 does not require:

- NVMe streaming;
- LRU cache misses;
- runtime expert-weight prefetch;
- multi-GPU;
- Vulkan, Metal, ROCm or RPC;
- router changes;
- shared-expert migration;
- dynamic rebalance.

Initial V2 does not require migration during an active request or an NVMe tier.

## 11. References

Study these implementations and discussions, but do not copy their RAM-backed cache semantics into the final design.

### Current fork and profiler

- https://github.com/erukolya/llama-cpp-turboquant
- https://github.com/TheTom/llama-cpp-turboquant
- https://github.com/erukolya/llama-cpp-turboquant/pull/1

### llama.cpp expert-cache and scheduling work

- https://github.com/ggml-org/llama.cpp/discussions/24528
- https://github.com/leloch/llama.cpp/tree/moe-cache-pr
- https://github.com/ggml-org/llama.cpp/issues/20757
- https://github.com/martinalderson/llama.cpp/tree/moe-profile
- https://github.com/ggml-org/llama.cpp/discussions/22584
- https://github.com/ggml-org/llama.cpp/pull/21067
- https://github.com/Lidenburg/llama.cpp
- https://github.com/ggml-org/llama.cpp/discussions/17621

Key difference:

```text
most cache prototypes:
    full RAM backing + VRAM copies

this project:
    exclusive CPU/GPU pools
```

### Related runtimes

- https://github.com/ikawrakow/ik_llama.cpp
- https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md
- https://github.com/kvcache-ai/ktransformers
- https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md
- https://github.com/JustVugg/colibri

Use them to study per-expert scheduling, dual CPU/GPU execution, expert masks and independent expert storage.

## 12. Development rules

- Work in small reviewable commits.
- Do not rewrite unrelated TurboQuant code.
- Preserve TurboQuant KV cache types and fit behavior unless explicitly changed.
- Keep the default path unchanged when the feature flag is absent.
- Add tests before performance optimization.
- Never silently use a different execution model.
- Never claim performance without reproducible commands and measurements.
- Do not merge a permanent full-RAM plus VRAM-copy cache as the final implementation.
- Consider multiple server slots even if the first benchmark uses `--parallel 1`.

## 13. First engineering task

Before implementing pools, create a design note that answers with confirmed source locations:

1. Where are packed routed-expert tensors created and loaded?
2. What is their exact `ne[]`, `nb[]` and quant-block layout for KAT Q5_K_M?
3. Where is `ffn_moe_topk` produced?
4. Where is `GGML_OP_MUL_MAT_ID` assigned to CPU or CUDA?
5. What does the current `CUDA_Host` routed-expert path actually execute?
6. Can current backend scheduling overlap a CPU and CUDA branch?
7. Which buffers own model tensors and what must change to release the original packed bank?
8. Which existing tests cover quantized `MUL_MAT_ID`?
9. What is the smallest correctness prototype that does not lock the project into a RAM-backed cache architecture?

No core runtime implementation should begin until this note identifies the ownership and synchronization model.
