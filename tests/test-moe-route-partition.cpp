#include "llama-moe-route-partition.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

llama_moe_load_location cpu(uint32_t local) {
    return {llama_moe_load_backend::cpu, local};
}

llama_moe_load_location gpu(uint32_t local) {
    return {llama_moe_load_backend::gpu, local};
}

void test_mixed_partition_preserves_slots() {
    llama_moe_load_layer_placement placement;
    placement.layer = 7;
    placement.expert_count = 4;
    placement.cpu_expert_count = 2;
    placement.gpu_expert_count = 2;
    placement.global_to_local = {cpu(0), gpu(0), cpu(1), gpu(1)};

    llama_moe_route_maps maps;
    std::string error;
    assert(llama_moe_route_maps_build(placement, maps, error));
    assert(error.empty());
    assert((maps.cpu_local_by_global == std::vector<int32_t>{0, -1, 1, -1}));
    assert((maps.gpu_local_by_global == std::vector<int32_t>{-1, 0, -1, 1}));

    const int32_t selected[] = {3, 0, 2, 1, 3, 2};
    std::vector<int32_t> cpu_ids;
    std::vector<int32_t> gpu_ids;
    assert(llama_moe_route_partition_selected(
        maps, selected, sizeof(selected) / sizeof(selected[0]), cpu_ids, gpu_ids, error));
    assert(error.empty());
    assert((cpu_ids == std::vector<int32_t>{-1, 0, 1, -1, -1, 1}));
    assert((gpu_ids == std::vector<int32_t>{1, -1, -1, 0, 1, -1}));

    for (size_t index = 0; index < cpu_ids.size(); ++index) {
        assert((cpu_ids[index] >= 0) != (gpu_ids[index] >= 0));
    }
}

void test_partition_preserves_router_weights_and_sum() {
    llama_moe_load_layer_placement placement;
    placement.layer = 9;
    placement.expert_count = 4;
    placement.cpu_expert_count = 2;
    placement.gpu_expert_count = 2;
    placement.global_to_local = {cpu(0), gpu(0), cpu(1), gpu(1)};

    llama_moe_route_maps maps;
    std::string error;
    assert(llama_moe_route_maps_build(placement, maps, error));

    constexpr size_t top_k = 3;
    constexpr size_t token_count = 2;
    const int32_t selected[top_k * token_count] = {
        3, 0, 2,
        1, 3, 0,
    };
    const float weights[top_k * token_count] = {
        0.50f, 0.30f, 0.20f,
        0.60f, 0.25f, 0.15f,
    };

    // Synthetic scalar output for each global expert. Local pool ordering follows
    // the placement above: CPU [global 0, global 2], GPU [global 1, global 3].
    const float global_output[] = {10.0f, 20.0f, 30.0f, 40.0f};
    const float cpu_output[] = {10.0f, 30.0f};
    const float gpu_output[] = {20.0f, 40.0f};

    std::vector<int32_t> cpu_ids;
    std::vector<int32_t> gpu_ids;
    assert(llama_moe_route_partition_selected(
        maps, selected, top_k * token_count, cpu_ids, gpu_ids, error));

    for (size_t token = 0; token < token_count; ++token) {
        float original = 0.0f;
        float cpu_branch = 0.0f;
        float gpu_branch = 0.0f;

        for (size_t slot = 0; slot < top_k; ++slot) {
            const size_t index = token * top_k + slot;
            const int32_t global_id = selected[index];
            const float weight = weights[index];

            original += weight * global_output[global_id];

            if (cpu_ids[index] >= 0) {
                cpu_branch += weight * cpu_output[cpu_ids[index]];
            }
            if (gpu_ids[index] >= 0) {
                gpu_branch += weight * gpu_output[gpu_ids[index]];
            }
        }

        assert(std::fabs(original - (cpu_branch + gpu_branch)) < 1e-6f);
    }
}

void test_all_cpu_and_all_gpu() {
    std::string error;

    llama_moe_load_layer_placement all_cpu;
    all_cpu.expert_count = 3;
    all_cpu.cpu_expert_count = 3;
    all_cpu.gpu_expert_count = 0;
    all_cpu.global_to_local = {cpu(0), cpu(1), cpu(2)};

    llama_moe_route_maps cpu_maps;
    assert(llama_moe_route_maps_build(all_cpu, cpu_maps, error));
    const int32_t selected_cpu[] = {2, 0, 1};
    std::vector<int32_t> cpu_ids;
    std::vector<int32_t> gpu_ids;
    assert(llama_moe_route_partition_selected(cpu_maps, selected_cpu, 3, cpu_ids, gpu_ids, error));
    assert((cpu_ids == std::vector<int32_t>{2, 0, 1}));
    assert((gpu_ids == std::vector<int32_t>{-1, -1, -1}));

    llama_moe_load_layer_placement all_gpu;
    all_gpu.expert_count = 3;
    all_gpu.cpu_expert_count = 0;
    all_gpu.gpu_expert_count = 3;
    all_gpu.global_to_local = {gpu(0), gpu(1), gpu(2)};

    llama_moe_route_maps gpu_maps;
    assert(llama_moe_route_maps_build(all_gpu, gpu_maps, error));
    const int32_t selected_gpu[] = {1, 2, 0};
    assert(llama_moe_route_partition_selected(gpu_maps, selected_gpu, 3, cpu_ids, gpu_ids, error));
    assert((cpu_ids == std::vector<int32_t>{-1, -1, -1}));
    assert((gpu_ids == std::vector<int32_t>{1, 2, 0}));
}

void test_invalid_maps_are_rejected() {
    std::string error;
    llama_moe_route_maps maps;

    llama_moe_load_layer_placement duplicate;
    duplicate.expert_count = 3;
    duplicate.cpu_expert_count = 2;
    duplicate.gpu_expert_count = 1;
    duplicate.global_to_local = {cpu(0), cpu(0), gpu(0)};
    assert(!llama_moe_route_maps_build(duplicate, maps, error));
    assert(error == "duplicate CPU local expert ID");

    llama_moe_load_layer_placement incomplete;
    incomplete.expert_count = 3;
    incomplete.cpu_expert_count = 1;
    incomplete.gpu_expert_count = 1;
    incomplete.global_to_local = {cpu(0), gpu(0), gpu(0)};
    assert(!llama_moe_route_maps_build(incomplete, maps, error));
    assert(error == "route placement CPU/GPU counts do not cover all experts");
}

void test_invalid_selected_id_is_rejected_without_partial_output() {
    llama_moe_load_layer_placement placement;
    placement.expert_count = 2;
    placement.cpu_expert_count = 1;
    placement.gpu_expert_count = 1;
    placement.global_to_local = {cpu(0), gpu(0)};

    llama_moe_route_maps maps;
    std::string error;
    assert(llama_moe_route_maps_build(placement, maps, error));

    const int32_t selected[] = {0, 2};
    std::vector<int32_t> cpu_ids = {99};
    std::vector<int32_t> gpu_ids = {98};
    assert(!llama_moe_route_partition_selected(maps, selected, 2, cpu_ids, gpu_ids, error));
    assert(error == "selected global expert ID is out of range");
    assert((cpu_ids == std::vector<int32_t>{99}));
    assert((gpu_ids == std::vector<int32_t>{98}));
}

} // namespace

int main() {
    test_mixed_partition_preserves_slots();
    test_partition_preserves_router_weights_and_sum();
    test_all_cpu_and_all_gpu();
    test_invalid_maps_are_rejected();
    test_invalid_selected_id_is_rejected_without_partial_output();
    std::cout << "test-moe-route-partition: ok\n";
    return 0;
}
