#pragma once

#include "../../src/llama-moe-placement.h"

#include "moe-expert-plan.h"

#include <cstdint>
#include <string>

class server_moe_load_placement {
public:
    bool prepare(const std::string & plan_path, std::string & error) {
        common_moe_expert_plan plan;
        if (!common_moe_expert_plan_load(plan_path, plan, error)) {
            return false;
        }
        if (!common_moe_expert_placement_build(plan, placement_, error)) {
            return false;
        }

        view_.layer_count = static_cast<uint32_t>(placement_.layers.size());
        view_.logical_expert_count = placement_.logical_expert_count;
        view_.query = query;
        view_.userdata = &placement_;
        return true;
    }

    const llama_moe_load_placement_view * view() const noexcept {
        return view_.query == nullptr ? nullptr : &view_;
    }

private:
    static bool query(
            const void * userdata,
            int32_t layer,
            uint32_t global_expert,
            llama_moe_load_location * location) {
        if (userdata == nullptr || location == nullptr) {
            return false;
        }
        const auto & placement = *static_cast<const common_moe_expert_placement *>(userdata);
        const auto * layer_placement = placement.find_layer(layer);
        if (layer_placement == nullptr || global_expert >= layer_placement->global_to_local.size()) {
            return false;
        }

        const auto & source = layer_placement->global_to_local[global_expert];
        location->backend = source.backend == common_moe_expert_backend::gpu ?
            llama_moe_load_backend::gpu : llama_moe_load_backend::cpu;
        location->local_index = source.local_index;
        return true;
    }

    common_moe_expert_placement placement_;
    llama_moe_load_placement_view view_;
};
