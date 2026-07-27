#pragma once

#include "llama-moe-placement.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct llama_moe_packed_tensor_layout {
    int32_t layer = -1;
    std::string name;
    std::array<int64_t, 4> ne = {1, 1, 1, 1};
    std::array<uint64_t, 4> nb = {0, 0, 0, 0};
    uint64_t size_bytes = 0;
};

struct llama_moe_pool_copy_span {
    uint32_t global_expert_first = 0;
    uint32_t local_expert_first = 0;
    uint32_t expert_count = 0;
    uint64_t source_offset_bytes = 0;
    uint64_t destination_offset_bytes = 0;
    uint64_t size_bytes = 0;
};

// Describes two exclusive compact destinations for one packed routed-expert
// source tensor. A backend with zero experts has no destination tensor and its
// ne[2] is zero; callers must skip allocation for that backend.
// Validates a set of source-to-destination byte ranges. Destination ranges
// must cover the compact tensor exactly once; source ranges must stay within
// the packed tensor and may not overlap.
bool llama_moe_pool_copy_spans_validate(
    uint64_t source_size_bytes,
    uint64_t destination_size_bytes,
    const std::vector<llama_moe_pool_copy_span> & spans,
    std::string & error);

struct llama_moe_compact_tensor_pool_plan {
    int32_t layer = -1;
    std::string source_name;
    uint64_t expert_stride_bytes = 0;
    uint32_t cpu_expert_count = 0;
    uint32_t gpu_expert_count = 0;
    std::array<int64_t, 4> cpu_ne = {1, 1, 0, 1};
    std::array<int64_t, 4> gpu_ne = {1, 1, 0, 1};
    uint64_t cpu_bytes = 0;
    uint64_t gpu_bytes = 0;
    std::vector<llama_moe_pool_copy_span> cpu_spans;
    std::vector<llama_moe_pool_copy_span> gpu_spans;
};

bool llama_moe_compact_tensor_pool_plan_build(
    const llama_moe_packed_tensor_layout & source,
    const llama_moe_load_layer_placement & placement,
    llama_moe_compact_tensor_pool_plan & plan,
    std::string & error);
