# V1 design note: confirmed current MoE loading and execution path

This note records facts confirmed against branch `agent/static-moe-expert-placement` before implementing static per-expert placement.

It complements `docs/moe-expert-static-placement-spec.md`.

## 1. Confirmed source locations

### Model loader

Primary file:

```text
src/llama-model-loader.cpp
```

Relevant functions:

```text
llama_model_loader::create_tensor
llama_model_loader::create_tensor_as_view
llama_model_loader::init_mappings
llama_model_loader::load_data_for
llama_model_loader::load_all_data
```

### MoE graph construction

Primary file:

```text
src/llama-graph.cpp
```

Relevant functions:

```text
llm_graph_context::build_moe_ffn
llm_graph_context::build_lora_mm_id
```

### Backend assignment and host-weight offload

Primary file:

```text
ggml/src/ggml-backend.cpp
```

Relevant functions:

```text
ggml_backend_sched_backend_id_from_cur
ggml_backend_sched_split_graph
ggml_backend_sched_compute_splits
```

### CUDA `MUL_MAT_ID`

Primary area:

```text
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-cuda/*mul-mat-id / mmq / mmvq / topk-moe related headers
```

The exact CUDA dispatch and supported quant types still require a focused trace before V1.2/V1.3.

## 2. Confirmed loader behavior

### 2.1 Buffer type is selected for the whole tensor

`llama_model_loader::create_tensor` obtains `llm_tensor_info`, determines the operation type and selects one backend buffer type for the complete tensor.

Tensor buffer overrides are also evaluated at the tensor-name level. When a tensor is overridden to CPU, the loader selects a compatible CPU/host buffer for the complete packed tensor.

Implication:

```text
blk.N.ffn_up_exps.weight
```

is currently assigned as one unit. There is no built-in per-expert buffer selection inside this tensor.

### 2.2 A tensor view does not provide independent residency

`llama_model_loader::create_tensor_as_view` uses `ggml_view_4d` over a base tensor.

A view therefore retains the base tensor and its buffer. Splitting experts into views cannot satisfy exclusive residency or release the original packed bank.

### 2.3 Current model loading is whole-tensor oriented

`llama_model_loader::load_data_for` either:

- points tensor data into an mmap region; or
- copies `ggml_nbytes(cur)` from the GGUF file.

`llama_model_loader::load_all_data` iterates runtime tensors and uploads/copies complete tensor allocations. It already contains pinned staging buffers and backend events for asynchronous model upload when mmap is disabled.

Implication:

Selective expert loading should reuse the loader's file-offset and async-upload infrastructure, but it needs a new slice-aware path rather than calling the existing whole-tensor load unchanged.

## 3. Confirmed MoE graph behavior

`llm_graph_context::build_moe_ffn` currently performs:

```text
router logits
-> gating function
-> optional bias/group masking
-> ggml_argsort_top_k
-> selected_experts tensor named ffn_moe_topk
-> selected router weights
-> one or more MUL_MAT_ID operations over packed expert tensors
-> expert activation/gating
-> down MUL_MAT_ID
-> weight and sum expert outputs
```

The selected expert IDs are created by:

```cpp
ggml_argsort_top_k(ctx0, selection_probs, n_expert_used)
```

and exposed through the graph callback as:

```text
ffn_moe_topk
```

The current packed path calls `build_lora_mm_id`, which in turn calls:

```cpp
ggml_mul_mat_id(ctx0, packed_weights, activation, selected_experts)
```

for merged or separate gate/up and for down projection.

Implication:

The correct split point is after top-k and router-weight creation but before the packed `MUL_MAT_ID` calls.

A future mixed path must:

1. partition original IDs into CPU and CUDA assignments;
2. remap global IDs to local pool IDs;
3. execute compact CPU and CUDA assignments;
4. restore the original token/top-k contribution positions or directly reduce weighted partial outputs;
5. preserve router weights exactly.

## 4. Confirmed scheduler behavior

### 4.1 Operations prefer the backend that owns their weight buffer

`ggml_backend_sched_backend_id_from_cur` scans operation sources marked as weights and chooses a backend compatible with the weight buffer.

Without operation offload, CPU-resident weights therefore cause their operation to be assigned to CPU.

This is compatible with the intended final rule:

```text
CPU pool -> CPU compute
CUDA pool -> CUDA compute
```

### 4.2 Current op-offload path can override weight residency

When all of the following are true:

```text
scheduler op_offload enabled
weight is on the last backend, assumed CPU
weight buffer is host-accessible
a higher-priority backend supports and wants to offload the operation
```

the scheduler assigns the operation to the higher-priority backend instead of the weight backend.

This is important for `MUL_MAT_ID` and explains why `CUDA_Host` residency does not necessarily mean CPU compute.

### 4.3 Current implementation selectively copies active experts to GPU

`ggml_backend_sched_compute_splits` has a dedicated path for host-resident `GGML_OP_MUL_MAT_ID` weights.

It:

1. synchronizes and reads the `ids` tensor;
2. builds a bitset of used expert IDs;
3. uses:

```text
n_expert   = input->ne[2]
expert_size = input->nb[2]
```

4. groups consecutive selected IDs;
5. copies only those expert byte ranges to the destination backend;
6. runs the GPU split against the copied tensor.

The relevant copy is performed through asynchronous backend tensor set operations.

Therefore the current offloaded host-expert path is effectively:

```text
packed expert bank in CUDA_Host/RAM
-> inspect top-k IDs
-> copy selected expert slices over PCIe
-> CUDA MUL_MAT_ID
```

This confirms the user's concern: the current offload path can transfer expert weights during inference.

## 5. Consequences for static placement design

### 5.1 CPU pool must not be operation-offloaded

For the new CPU expert pool, `MUL_MAT_ID` must remain on CPU. Otherwise static exclusive placement would still copy cold weights to GPU during every use.

The implementation needs an explicit mechanism, not an accidental scheduler outcome.

Possible designs to evaluate:

1. mark the split CPU operations as non-offloadable;
2. add an operation/tensor hint disabling `offload_op` for the CPU expert pool;
3. use a dedicated hybrid MoE op whose CPU sub-operation is internally dispatched and is not visible as a generic offloadable `MUL_MAT_ID`;
4. temporarily disable scheduler op-offload only for identified static CPU-pool nodes.

A global disabling of op offload is not preferred because it can regress unrelated operations.

### 5.2 Existing selective-copy code is a useful reference, not the final path

The current scheduler code proves:

- expert slices are represented by `nb[2]` in the existing packed layout;
- selected IDs can be collected and grouped;
- active-expert byte ranges can be copied independently;
- CUDA supports executing a copied subset representation.

However, it must not remain in the normal V1 decode path because it transfers weights per inference step and relies on a full RAM backing bank.

It can be reused for:

- validating expert-axis assumptions;
- initial model-load copy into compact pools;
- V2 idle-time migration;
- diagnostics.

### 5.3 Do not implement the final feature only in `tools/server`

The profiler integration is server-specific, but static expert storage and execution must be model/backend functionality.

Recommended ownership:

```text
profile and plan CLI parsing        -> common/tool layer
plan model and validation           -> reusable common or src component
split expert storage                -> llama model/loader layer
location tables                     -> llama model/runtime layer
mixed execution                     -> graph/backend layer
server metrics and reports          -> server integration
```

## 6. Parallelism status

The scheduler splits a graph into backend-specific subgraphs and dispatches split graphs with asynchronous backend compute calls.

It also inserts cross-backend input copies and backend events.

This means CPU/CUDA overlap may be possible, but it is not yet proven for a forked MoE graph with a join. The current scheduler iterates splits in order, so the actual overlap depends on:

- graph topological ordering;
- whether both independent branch splits are dispatched before either result is required;
- inserted copy dependencies;
- backend event semantics;
- where the join split is placed;
- whether CPU graph compute is truly asynchronous relative to the calling thread.

Before selecting scheduler-level branching as the final approach, add a minimal synthetic graph test:

```text
input
├─ CPU delayed branch
├─ CUDA delayed branch
└─ join
```

Measure whether elapsed time approaches `max(CPU, GPU)` or `CPU + GPU`.

If the scheduler serializes the branches, prefer a dedicated hybrid execution path similar to the CPU-owned `MUL_MAT_ID` approach discussed in llama.cpp MoE-cache RFC work.

## 7. Immediate implementation sequence

### Commit A: specification

Completed in:

```text
docs/moe-expert-static-placement-spec.md
```

### Commit B: current execution-path design note

This file.

### Commit C: V1.1 planner data model

Add reusable structures without runtime placement:

```text
moe profile record
model fingerprint
expert-size record
placement candidate
versioned placement plan
plan validation result
```

Do not add JSON parsing through an unrelated new dependency if the repository already has a supported JSON facility.

### Commit D: V1.1 offline plan tool

Add `llama-moe-plan` with:

```text
--model
--profile
--vram-budget-mib
--output
--report
```

It must read actual packed tensor metadata and compute per-expert bytes from validated layout.

### Commit E: tests

Test:

- CSV parsing;
- missing expert rows;
- deterministic ranking;
- unequal expert sizes;
- budget boundary;
- invalid model/profile pairing;
- tie-breaking by layer then expert;
- JSON round trip.

Only after these commits should V1.2 modify model loading.

## 8. Open questions before V1.2

The next design pass must answer with code references and tests:

1. Which exact model structure owns `gate/up/down` packed tensors for KAT/Qwen3.5 MoE?
2. Can a single layer hold alternative CPU/GPU tensors without changing all architecture-specific model builders?
3. What is the exact KAT Q5_K_M `ne[]`, `nb[]`, type and byte range for all three packed tensors?
4. Does each expert slice start on a valid quant-block and backend alignment boundary?
5. How should LoRA interact with split expert pools?
6. Are per-expert scale/bias tensors also packed, and must they be split?
7. Which CUDA `MUL_MAT_ID` kernels accept a compact number of experts that differs from model `n_expert`?
8. How should local expert IDs be represented for the down projection after activation?
9. Can graph reuse tolerate a fixed placement table and variable per-token CPU/GPU assignment counts?
10. What is the correct lifecycle for releasing the original mmap-backed packed expert representation while retaining unrelated mapped tensors?

## 9. Baseline instrumentation required before V1.2

Add or enable a diagnostic mode that reports for every routed-expert `MUL_MAT_ID`:

```text
layer
weight tensor name
weight buffer
assigned execution backend
whether selected slices were copied
copied weight bytes
ids readback bytes and synchronization time
operation compute time
```

This diagnostic must be optional and may be expensive.

The baseline report should distinguish:

```text
CPU-resident + CPU-computed
CPU-resident + CUDA-offloaded with selected weight copy
CUDA-resident + CUDA-computed
```

No performance conclusion should be made from placement CSV alone.
