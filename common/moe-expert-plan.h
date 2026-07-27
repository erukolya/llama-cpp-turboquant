#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct common_moe_expert_plan_tensor {
    int32_t layer = -1;
    std::string name;
    std::string type;
    uint32_t expert_first = 0;
    uint32_t expert_last = 0;
    uint64_t size_bytes = 0;
    uint64_t expert_stride_bytes = 0;
    std::array<int64_t, 4> ne = {1, 1, 1, 1};
    std::array<uint64_t, 4> nb = {0, 0, 0, 0};
};

struct common_moe_expert_plan_layer {
    int32_t layer = -1;
    uint32_t expert_count = 0;
    std::vector<uint32_t> gpu_experts;
    uint64_t gpu_bytes = 0;
    uint64_t estimated_hits = 0;
    uint64_t estimated_gpu_hits = 0;
    double estimated_gpu_hit_rate = 0.0;
};

struct common_moe_expert_plan {
    uint32_t schema_version = 0;
    std::string strategy;
    std::string model_fingerprint;
    std::string placement_fingerprint;
    std::vector<std::string> source_profiles;
    std::string source_placement;
    uint64_t vram_budget_bytes = 0;
    uint64_t selected_bytes = 0;
    uint64_t unused_budget_bytes = 0;
    uint32_t logical_expert_count = 0;
    uint32_t selected_expert_count = 0;
    uint64_t estimated_total_hits = 0;
    uint64_t estimated_gpu_hits = 0;
    double estimated_gpu_hit_rate = 0.0;
    std::vector<common_moe_expert_plan_tensor> tensor_manifest;
    std::vector<common_moe_expert_plan_layer> layers;
};

struct common_moe_expert_plan_expectation {
    int32_t layer_count = -1;
    int32_t experts_per_layer = -1;
    std::string model_fingerprint;
    bool require_tensor_manifest = false;
};

struct common_moe_expert_plan_validation {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    bool ok() const {
        return errors.empty();
    }
};

enum class common_moe_expert_backend : uint8_t {
    cpu = 0,
    gpu = 1,
};

struct common_moe_expert_location {
    common_moe_expert_backend backend = common_moe_expert_backend::cpu;
    uint32_t local_index = 0;
};

struct common_moe_expert_layer_placement {
    int32_t layer = -1;
    uint32_t expert_count = 0;
    uint64_t expert_bytes = 0;
    uint64_t cpu_bytes = 0;
    uint64_t gpu_bytes = 0;
    std::vector<common_moe_expert_location> global_to_local;
    std::vector<uint32_t> cpu_experts;
    std::vector<uint32_t> gpu_experts;
};

struct common_moe_expert_placement {
    uint32_t logical_expert_count = 0;
    uint32_t cpu_expert_count = 0;
    uint32_t gpu_expert_count = 0;
    uint64_t cpu_bytes = 0;
    uint64_t gpu_bytes = 0;
    std::vector<common_moe_expert_layer_placement> layers;

    const common_moe_expert_layer_placement * find_layer(int32_t layer) const {
        if (layer < 0 || static_cast<size_t>(layer) >= layers.size()) {
            return nullptr;
        }
        const auto & result = layers[static_cast<size_t>(layer)];
        return result.layer == layer ? &result : nullptr;
    }
};

bool common_moe_expert_plan_load(
        const std::string & path,
        common_moe_expert_plan & plan,
        std::string & error);

bool common_moe_expert_plan_save(
        const std::string & path,
        const common_moe_expert_plan & plan,
        std::string & error);

common_moe_expert_plan_validation common_moe_expert_plan_validate(
        const common_moe_expert_plan & plan,
        const common_moe_expert_plan_expectation & expectation = {});

inline bool common_moe_expert_placement_build(
        const common_moe_expert_plan & plan,
        common_moe_expert_placement & placement,
        std::string & error) {
    const auto validation = common_moe_expert_plan_validate(
        plan,
        common_moe_expert_plan_expectation{-1, -1, {}, true});
    if (!validation.ok()) {
        error = validation.errors.front();
        return false;
    }
    if (plan.schema_version < 2) {
        error = "exclusive expert placement requires a schema-v2 tensor manifest";
        return false;
    }

    std::map<int32_t, uint64_t> expert_bytes_by_layer;
    for (const auto & tensor : plan.tensor_manifest) {
        const auto layer_it = std::find_if(
            plan.layers.begin(),
            plan.layers.end(),
            [&](const common_moe_expert_plan_layer & layer) {
                return layer.layer == tensor.layer;
            });
        if (layer_it == plan.layers.end()) {
            error = "tensor manifest references a missing layer";
            return false;
        }
        if (tensor.expert_first != 0 || tensor.expert_last + 1 != layer_it->expert_count) {
            error = "tensor manifest must cover the complete expert range for layer " +
                std::to_string(tensor.layer);
            return false;
        }
        expert_bytes_by_layer[tensor.layer] += tensor.expert_stride_bytes;
    }

    common_moe_expert_placement built;
    built.layers.resize(plan.layers.size());

    for (const auto & source : plan.layers) {
        if (source.layer < 0 || static_cast<size_t>(source.layer) >= built.layers.size()) {
            error = "placement layers must be contiguous and zero-based";
            return false;
        }

        auto & target = built.layers[static_cast<size_t>(source.layer)];
        if (target.layer != -1) {
            error = "duplicate placement layer";
            return false;
        }

        const auto bytes_it = expert_bytes_by_layer.find(source.layer);
        if (bytes_it == expert_bytes_by_layer.end() || bytes_it->second == 0) {
            error = "missing routed tensor strides for layer " + std::to_string(source.layer);
            return false;
        }

        target.layer = source.layer;
        target.expert_count = source.expert_count;
        target.expert_bytes = bytes_it->second;
        target.gpu_experts = source.gpu_experts;
        target.global_to_local.resize(source.expert_count);
        target.cpu_experts.reserve(source.expert_count - source.gpu_experts.size());

        size_t gpu_cursor = 0;
        uint32_t cpu_local = 0;
        uint32_t gpu_local = 0;
        for (uint32_t global = 0; global < source.expert_count; ++global) {
            const bool is_gpu =
                gpu_cursor < source.gpu_experts.size() &&
                source.gpu_experts[gpu_cursor] == global;
            auto & location = target.global_to_local[global];
            if (is_gpu) {
                location.backend = common_moe_expert_backend::gpu;
                location.local_index = gpu_local++;
                ++gpu_cursor;
            } else {
                location.backend = common_moe_expert_backend::cpu;
                location.local_index = cpu_local++;
                target.cpu_experts.push_back(global);
            }
        }

        if (gpu_cursor != source.gpu_experts.size()) {
            error = "GPU expert list contains an unmapped expert";
            return false;
        }

        target.gpu_bytes = static_cast<uint64_t>(target.gpu_experts.size()) * target.expert_bytes;
        target.cpu_bytes = static_cast<uint64_t>(target.cpu_experts.size()) * target.expert_bytes;
        if (target.gpu_bytes != source.gpu_bytes) {
            error = "layer " + std::to_string(source.layer) +
                " GPU bytes differ from tensor-manifest strides";
            return false;
        }

        built.logical_expert_count += source.expert_count;
        built.gpu_expert_count += static_cast<uint32_t>(target.gpu_experts.size());
        built.cpu_expert_count += static_cast<uint32_t>(target.cpu_experts.size());
        built.gpu_bytes += target.gpu_bytes;
        built.cpu_bytes += target.cpu_bytes;
    }

    for (size_t layer = 0; layer < built.layers.size(); ++layer) {
        if (built.layers[layer].layer != static_cast<int32_t>(layer)) {
            error = "placement layers must be contiguous and zero-based";
            return false;
        }
    }
    if (built.logical_expert_count != plan.logical_expert_count ||
        built.gpu_expert_count != plan.selected_expert_count ||
        built.gpu_bytes != plan.selected_bytes) {
        error = "placement totals differ from plan totals";
        return false;
    }

    placement = std::move(built);
    error.clear();
    return true;
}
