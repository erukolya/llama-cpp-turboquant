#pragma once

#include <array>
#include <cstdint>
#include <string>
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
