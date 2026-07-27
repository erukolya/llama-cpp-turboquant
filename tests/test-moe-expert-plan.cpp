#include "moe-expert-plan.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static common_moe_expert_plan make_valid_plan() {
    common_moe_expert_plan plan;
    plan.schema_version = 1;
    plan.strategy = "greedy_hits_per_byte";
    plan.model_fingerprint = "sha256:model";
    plan.placement_fingerprint = "sha256:placement";
    plan.source_profiles = {"stats.csv"};
    plan.source_placement = "placement.csv";
    plan.vram_budget_bytes = 400;
    plan.selected_bytes = 300;
    plan.unused_budget_bytes = 100;
    plan.logical_expert_count = 4;
    plan.selected_expert_count = 2;
    plan.estimated_total_hits = 270;
    plan.estimated_gpu_hits = 180;
    plan.estimated_gpu_hit_rate = 180.0 / 270.0;

    common_moe_expert_plan_layer layer0;
    layer0.layer = 0;
    layer0.expert_count = 2;
    layer0.gpu_experts = {0};
    layer0.gpu_bytes = 100;
    layer0.estimated_hits = 120;
    layer0.estimated_gpu_hits = 100;
    layer0.estimated_gpu_hit_rate = 100.0 / 120.0;

    common_moe_expert_plan_layer layer1;
    layer1.layer = 1;
    layer1.expert_count = 2;
    layer1.gpu_experts = {0};
    layer1.gpu_bytes = 200;
    layer1.estimated_hits = 150;
    layer1.estimated_gpu_hits = 80;
    layer1.estimated_gpu_hit_rate = 80.0 / 150.0;

    plan.layers = {layer0, layer1};
    return plan;
}

static void test_validation() {
    auto plan = make_valid_plan();
    common_moe_expert_plan_expectation expectation;
    expectation.layer_count = 2;
    expectation.experts_per_layer = 2;
    expectation.model_fingerprint = "sha256:model";

    const auto valid = common_moe_expert_plan_validate(plan, expectation);
    require(valid.ok(), "valid plan rejected");

    plan.layers[0].gpu_experts = {1, 0};
    const auto invalid = common_moe_expert_plan_validate(plan, expectation);
    require(!invalid.ok(), "unsorted GPU experts accepted");
}

static void test_round_trip() {
    const std::string path = "test-moe-expert-plan.tmp.json";
    const auto plan = make_valid_plan();
    std::string error;
    require(common_moe_expert_plan_save(path, plan, error), error.c_str());

    common_moe_expert_plan loaded;
    require(common_moe_expert_plan_load(path, loaded, error), error.c_str());
    std::remove(path.c_str());

    require(loaded.schema_version == plan.schema_version, "schema version changed");
    require(loaded.model_fingerprint == plan.model_fingerprint, "fingerprint changed");
    require(loaded.layers.size() == 2, "layer count changed");
    require(loaded.layers[1].gpu_experts == std::vector<uint32_t>({0}), "GPU expert list changed");
    require(common_moe_expert_plan_validate(loaded).ok(), "round-tripped plan is invalid");
}

static void test_malformed_json() {
    const std::string path = "test-moe-expert-plan-invalid.tmp.json";
    {
        std::ofstream output(path);
        output << R"({"schema_version":1})";
    }
    common_moe_expert_plan plan;
    std::string error;
    require(!common_moe_expert_plan_load(path, plan, error), "malformed plan accepted");
    require(!error.empty(), "malformed plan returned no error");
    std::remove(path.c_str());
}

int main() {
    test_validation();
    test_round_trip();
    test_malformed_json();
    return 0;
}
