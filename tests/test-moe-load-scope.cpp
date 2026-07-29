#include "llama-moe-placement.h"

#include <cstdint>
#include <stdexcept>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct query_state {
    uint32_t gpu_cutoff = 0;
    bool duplicate_gpu_local = false;
};

static bool query(
        const void * userdata,
        int32_t layer,
        uint32_t global_expert,
        llama_moe_load_location * location) {
    if (userdata == nullptr || location == nullptr || layer < 0 || layer >= 2 || global_expert >= 4) {
        return false;
    }
    const auto & state = *static_cast<const query_state *>(userdata);
    location->backend = global_expert < state.gpu_cutoff ?
        llama_moe_load_backend::gpu : llama_moe_load_backend::cpu;
    location->local_index = global_expert < state.gpu_cutoff ?
        (state.duplicate_gpu_local ? 0 : global_expert) : global_expert - state.gpu_cutoff;
    return true;
}

static llama_moe_load_placement_snapshot build_snapshot(query_state & state) {
    llama_moe_load_placement_view view;
    view.layer_count = 2;
    view.experts_per_layer = 4;
    view.logical_expert_count = 8;
    view.query = query;
    view.userdata = &state;

    llama_moe_load_placement_snapshot snapshot;
    std::string error;
    require(llama_moe_load_placement_snapshot_build(view, snapshot, error), error.c_str());
    return snapshot;
}

int main() {
    require(llama_moe_load_placement_current() == nullptr, "placement leaked before scope");

    query_state outer_state{2, false};
    llama_moe_load_placement_snapshot outer = build_snapshot(outer_state);
    require(outer.logical_expert_count == 8, "logical expert count mismatch");
    require(outer.gpu_expert_count == 4, "GPU expert count mismatch");
    require(outer.cpu_expert_count == 4, "CPU expert count mismatch");

#ifndef LLAMA_MOE_SHARED_API_TEST
    // This validator is intentionally internal to the llama core. Keep its
    // coverage in the direct source-linked unit test, but do not require it to
    // be exported by the Windows shared-library API smoke test.
    std::string dimension_error;
    require(llama_moe_load_placement_snapshot_validate_dimensions(outer, 2, 4, dimension_error),
        dimension_error.c_str());
    require(!llama_moe_load_placement_snapshot_validate_dimensions(outer, 3, 4, dimension_error),
        "wrong layer count must be rejected");
    require(!llama_moe_load_placement_snapshot_validate_dimensions(outer, 2, 5, dimension_error),
        "wrong expert count must be rejected");
#endif

    // Prove the snapshot is independent from the source view and userdata.
    outer_state.gpu_cutoff = 0;
    llama_moe_load_location location;
    require(outer.query(0, 1, &location), "snapshot GPU query failed");
    require(location.backend == llama_moe_load_backend::gpu, "snapshot did not deep-copy GPU mapping");
    require(location.local_index == 1, "snapshot GPU local index mismatch");
    require(outer.query(0, 3, &location), "snapshot CPU query failed");
    require(location.backend == llama_moe_load_backend::cpu, "snapshot CPU mapping mismatch");
    require(location.local_index == 1, "snapshot CPU local index mismatch");

    {
        llama_moe_load_placement_scope outer_scope(&outer);
        require(llama_moe_load_placement_current() == &outer, "outer scope not installed");

        query_state inner_state{1, false};
        llama_moe_load_placement_snapshot inner = build_snapshot(inner_state);
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

    query_state duplicate_state{2, true};
    llama_moe_load_placement_view duplicate_view{2, 4, 8, query, &duplicate_state};
    llama_moe_load_placement_snapshot invalid;
    std::string error;
    require(!llama_moe_load_placement_snapshot_build(duplicate_view, invalid, error),
        "duplicate local IDs must be rejected");
    require(error.find("duplicate") != std::string::npos, "duplicate error message missing");

    llama_moe_load_placement_view bad_total_view{2, 4, 7, query, &outer_state};
    require(!llama_moe_load_placement_snapshot_build(bad_total_view, invalid, error),
        "invalid logical total must be rejected");

    return 0;
}
