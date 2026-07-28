#include "llama-moe-route-partition.h"

#include <algorithm>
#include <sstream>

namespace {

bool validate_complete_local_ids(
        const std::vector<uint8_t> & seen,
        const char * backend,
        std::string & error) {
    const auto missing = std::find(seen.begin(), seen.end(), uint8_t{0});
    if (missing == seen.end()) {
        return true;
    }

    std::ostringstream message;
    message << backend << " local expert ID "
            << std::distance(seen.begin(), missing)
            << " is not assigned";
    error = message.str();
    return false;
}

} // namespace

bool llama_moe_route_maps_build(
        const llama_moe_load_layer_placement & placement,
        llama_moe_route_maps & maps,
        std::string & error) {
    llama_moe_route_maps candidate;

    if (placement.expert_count == 0) {
        error = "route placement has no experts";
        return false;
    }
    if (placement.global_to_local.size() != placement.expert_count) {
        error = "route placement global-to-local size differs from expert_count";
        return false;
    }
    if (placement.cpu_expert_count + placement.gpu_expert_count != placement.expert_count) {
        error = "route placement CPU/GPU counts do not cover all experts";
        return false;
    }

    candidate.expert_count = placement.expert_count;
    candidate.cpu_expert_count = placement.cpu_expert_count;
    candidate.gpu_expert_count = placement.gpu_expert_count;
    candidate.cpu_local_by_global.assign(placement.expert_count, LLAMA_MOE_ROUTE_MISSING);
    candidate.gpu_local_by_global.assign(placement.expert_count, LLAMA_MOE_ROUTE_MISSING);

    std::vector<uint8_t> cpu_seen(placement.cpu_expert_count, 0);
    std::vector<uint8_t> gpu_seen(placement.gpu_expert_count, 0);

    for (uint32_t global = 0; global < placement.expert_count; ++global) {
        const auto & location = placement.global_to_local[global];
        if (location.backend == llama_moe_load_backend::cpu) {
            if (location.local_index >= placement.cpu_expert_count) {
                error = "CPU local expert ID is out of range";
                return false;
            }
            if (cpu_seen[location.local_index] != 0) {
                error = "duplicate CPU local expert ID";
                return false;
            }
            cpu_seen[location.local_index] = 1;
            candidate.cpu_local_by_global[global] = static_cast<int32_t>(location.local_index);
        } else if (location.backend == llama_moe_load_backend::gpu) {
            if (location.local_index >= placement.gpu_expert_count) {
                error = "GPU local expert ID is out of range";
                return false;
            }
            if (gpu_seen[location.local_index] != 0) {
                error = "duplicate GPU local expert ID";
                return false;
            }
            gpu_seen[location.local_index] = 1;
            candidate.gpu_local_by_global[global] = static_cast<int32_t>(location.local_index);
        } else {
            error = "unknown route placement backend";
            return false;
        }
    }

    if (!validate_complete_local_ids(cpu_seen, "CPU", error) ||
        !validate_complete_local_ids(gpu_seen, "GPU", error)) {
        return false;
    }

    maps = std::move(candidate);
    error.clear();
    return true;
}

bool llama_moe_route_partition_selected(
        const llama_moe_route_maps & maps,
        const int32_t * global_ids,
        size_t count,
        std::vector<int32_t> & cpu_local_ids,
        std::vector<int32_t> & gpu_local_ids,
        std::string & error) {
    if (maps.expert_count == 0 ||
        maps.cpu_local_by_global.size() != maps.expert_count ||
        maps.gpu_local_by_global.size() != maps.expert_count) {
        error = "route maps are incomplete";
        return false;
    }
    if (count != 0 && global_ids == nullptr) {
        error = "selected global expert IDs are null";
        return false;
    }

    std::vector<int32_t> candidate_cpu(count, LLAMA_MOE_ROUTE_MISSING);
    std::vector<int32_t> candidate_gpu(count, LLAMA_MOE_ROUTE_MISSING);

    for (size_t index = 0; index < count; ++index) {
        const int32_t global = global_ids[index];
        if (global < 0 || static_cast<uint32_t>(global) >= maps.expert_count) {
            error = "selected global expert ID is out of range";
            return false;
        }

        const int32_t cpu = maps.cpu_local_by_global[global];
        const int32_t gpu = maps.gpu_local_by_global[global];
        if ((cpu >= 0) == (gpu >= 0)) {
            error = "selected expert must map to exactly one backend";
            return false;
        }
        candidate_cpu[index] = cpu;
        candidate_gpu[index] = gpu;
    }

    cpu_local_ids = std::move(candidate_cpu);
    gpu_local_ids = std::move(candidate_gpu);
    error.clear();
    return true;
}
