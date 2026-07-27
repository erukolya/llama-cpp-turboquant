#include "llama-moe-placement.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace {

thread_local const llama_moe_load_placement_snapshot * g_current_moe_load_placement = nullptr;

bool validate_local_indices(
        const std::vector<llama_moe_load_location> & locations,
        llama_moe_load_backend backend,
        uint32_t expected_count,
        int32_t layer,
        std::string & error) {
    std::vector<bool> seen(expected_count, false);
    for (const auto & location : locations) {
        if (location.backend != backend) {
            continue;
        }
        if (location.local_index >= expected_count) {
            error = "layer " + std::to_string(layer) + ": local expert index is outside its backend pool";
            return false;
        }
        if (seen[location.local_index]) {
            error = "layer " + std::to_string(layer) + ": duplicate local expert index";
            return false;
        }
        seen[location.local_index] = true;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        error = "layer " + std::to_string(layer) + ": local expert indices are not contiguous";
        return false;
    }
    return true;
}

} // namespace

const llama_moe_load_location * llama_moe_load_layer_placement::find(uint32_t global_expert) const noexcept {
    return global_expert < global_to_local.size() ? &global_to_local[global_expert] : nullptr;
}

bool llama_moe_load_placement_snapshot::empty() const noexcept {
    return layers.empty();
}

const llama_moe_load_layer_placement * llama_moe_load_placement_snapshot::find_layer(int32_t layer) const noexcept {
    if (layer < 0 || static_cast<size_t>(layer) >= layers.size()) {
        return nullptr;
    }
    const auto & result = layers[static_cast<size_t>(layer)];
    return result.layer == layer ? &result : nullptr;
}

bool llama_moe_load_placement_snapshot::query(
        int32_t layer,
        uint32_t global_expert,
        llama_moe_load_location * location) const noexcept {
    if (location == nullptr) {
        return false;
    }
    const auto * layer_placement = find_layer(layer);
    const auto * source = layer_placement == nullptr ? nullptr : layer_placement->find(global_expert);
    if (source == nullptr) {
        return false;
    }
    *location = *source;
    return true;
}

bool llama_moe_load_placement_snapshot_build(
        const llama_moe_load_placement_view & view,
        llama_moe_load_placement_snapshot & snapshot,
        std::string & error) {
    if (view.query == nullptr) {
        error = "placement query callback is null";
        return false;
    }
    if (view.layer_count == 0 || view.experts_per_layer == 0) {
        error = "placement must contain at least one layer and one expert";
        return false;
    }
    if (view.layer_count > std::numeric_limits<uint32_t>::max() / view.experts_per_layer ||
        view.layer_count * view.experts_per_layer != view.logical_expert_count) {
        error = "placement logical expert count does not match layer dimensions";
        return false;
    }

    llama_moe_load_placement_snapshot built;
    built.logical_expert_count = view.logical_expert_count;
    built.layers.resize(view.layer_count);

    for (uint32_t layer_index = 0; layer_index < view.layer_count; ++layer_index) {
        auto & layer = built.layers[layer_index];
        layer.layer = static_cast<int32_t>(layer_index);
        layer.expert_count = view.experts_per_layer;
        layer.global_to_local.resize(view.experts_per_layer);

        for (uint32_t global_expert = 0; global_expert < view.experts_per_layer; ++global_expert) {
            llama_moe_load_location location;
            if (!view.query(view.userdata, layer.layer, global_expert, &location)) {
                error = "placement query failed for layer " + std::to_string(layer.layer) +
                    ", expert " + std::to_string(global_expert);
                return false;
            }
            if (location.backend == llama_moe_load_backend::gpu) {
                ++layer.gpu_expert_count;
            } else if (location.backend == llama_moe_load_backend::cpu) {
                ++layer.cpu_expert_count;
            } else {
                error = "placement query returned an unknown backend";
                return false;
            }
            layer.global_to_local[global_expert] = location;
        }

        if (layer.cpu_expert_count + layer.gpu_expert_count != layer.expert_count) {
            error = "layer " + std::to_string(layer.layer) + ": backend expert counts are inconsistent";
            return false;
        }
        if (!validate_local_indices(
                layer.global_to_local,
                llama_moe_load_backend::cpu,
                layer.cpu_expert_count,
                layer.layer,
                error) ||
            !validate_local_indices(
                layer.global_to_local,
                llama_moe_load_backend::gpu,
                layer.gpu_expert_count,
                layer.layer,
                error)) {
            return false;
        }

        built.cpu_expert_count += layer.cpu_expert_count;
        built.gpu_expert_count += layer.gpu_expert_count;
    }

    if (built.cpu_expert_count + built.gpu_expert_count != built.logical_expert_count) {
        error = "placement snapshot totals are inconsistent";
        return false;
    }

    snapshot = std::move(built);
    error.clear();
    return true;
}

llama_moe_load_placement_scope::llama_moe_load_placement_scope(
        const llama_moe_load_placement_snapshot * snapshot) noexcept :
    previous_(g_current_moe_load_placement) {
    g_current_moe_load_placement = snapshot;
}

llama_moe_load_placement_scope::~llama_moe_load_placement_scope() {
    g_current_moe_load_placement = previous_;
}

const llama_moe_load_placement_snapshot * llama_moe_load_placement_current() noexcept {
    return g_current_moe_load_placement;
}
