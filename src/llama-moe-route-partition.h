#pragma once

#include "llama-moe-placement.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr int32_t LLAMA_MOE_ROUTE_MISSING = -1;

// Dense per-layer lookup tables used by the mixed CPU/CUDA execution graph.
// For every global expert exactly one table contains a non-negative local ID;
// the other contains LLAMA_MOE_ROUTE_MISSING.
struct llama_moe_route_maps {
    uint32_t expert_count = 0;
    uint32_t cpu_expert_count = 0;
    uint32_t gpu_expert_count = 0;
    std::vector<int32_t> cpu_local_by_global;
    std::vector<int32_t> gpu_local_by_global;
};

bool llama_moe_route_maps_build(
    const llama_moe_load_layer_placement & placement,
    llama_moe_route_maps & maps,
    std::string & error);

// Host-side reference partitioner used by tests and diagnostics. It preserves
// the original top-k slot order. Router weights do not need to be changed:
// each slot is active in exactly one backend-specific ID vector.
bool llama_moe_route_partition_selected(
    const llama_moe_route_maps & maps,
    const int32_t * global_ids,
    size_t count,
    std::vector<int32_t> & cpu_local_ids,
    std::vector<int32_t> & gpu_local_ids,
    std::string & error);
