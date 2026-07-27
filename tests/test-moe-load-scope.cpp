#include "llama-moe-placement.h"

#include <cstdint>
#include <stdexcept>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static bool query(
        const void * userdata,
        int32_t layer,
        uint32_t global_expert,
        llama_moe_load_location * location) {
    if (userdata == nullptr || location == nullptr || layer < 0 || global_expert >= 4) {
        return false;
    }
    const uint32_t gpu_cutoff = *static_cast<const uint32_t *>(userdata);
    location->backend = global_expert < gpu_cutoff ?
        llama_moe_load_backend::gpu : llama_moe_load_backend::cpu;
    location->local_index = global_expert < gpu_cutoff ? global_expert : global_expert - gpu_cutoff;
    return true;
}

int main() {
    require(llama_moe_load_placement_current() == nullptr, "placement leaked before scope");

    uint32_t outer_cutoff = 2;
    llama_moe_load_placement_view outer;
    outer.layer_count = 2;
    outer.logical_expert_count = 8;
    outer.query = query;
    outer.userdata = &outer_cutoff;

    {
        llama_moe_load_placement_scope outer_scope(&outer);
        require(llama_moe_load_placement_current() == &outer, "outer scope not installed");

        llama_moe_load_location location;
        require(outer.query(outer.userdata, 0, 1, &location), "outer query failed");
        require(location.backend == llama_moe_load_backend::gpu, "GPU mapping mismatch");
        require(location.local_index == 1, "GPU local index mismatch");
        require(outer.query(outer.userdata, 0, 3, &location), "outer CPU query failed");
        require(location.backend == llama_moe_load_backend::cpu, "CPU mapping mismatch");
        require(location.local_index == 1, "CPU local index mismatch");

        uint32_t inner_cutoff = 1;
        llama_moe_load_placement_view inner = outer;
        inner.userdata = &inner_cutoff;
        {
            llama_moe_load_placement_scope inner_scope(&inner);
            require(llama_moe_load_placement_current() == &inner, "inner scope not installed");
        }
        require(llama_moe_load_placement_current() == &outer, "outer scope not restored");
    }

    require(llama_moe_load_placement_current() == nullptr, "placement leaked after scope");

    {
        llama_moe_load_placement_scope disabled(nullptr);
        require(llama_moe_load_placement_current() == nullptr, "null scope should stay disabled");
    }

    return 0;
}
