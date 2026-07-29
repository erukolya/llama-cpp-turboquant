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

        common_moe_expert_placement placement;
        if (!common_moe_expert_placement_build(plan, placement, error)) {
            return false;
        }
        if (placement.layers.empty()) {
            error = "placement contains no layers";
            return false;
        }

        const uint32_t experts_per_layer = placement.layers.front().expert_count;
        for (const auto & layer : placement.layers) {
            if (layer.expert_count != experts_per_layer) {
                error = "model-load placement requires a uniform expert count per layer";
                return false;
            }
        }

        llama_moe_load_placement_view view;
        view.layer_count = static_cast<uint32_t>(placement.layers.size());
        view.experts_per_layer = experts_per_layer;
        view.logical_expert_count = placement.logical_expert_count;
        view.query = query;
        view.userdata = &placement;

        return llama_moe_load_placement_snapshot_build(view, snapshot_, error);
    }

    const llama_moe_load_placement_snapshot * view() const noexcept {
        return snapshot_.empty() ? nullptr : &snapshot_;
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

    llama_moe_load_placement_snapshot snapshot_;
};
