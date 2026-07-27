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

static common_moe_expert_plan_tensor make_tensor(
        int32_t layer,
        const std::string & name,
        uint64_t stride) {
    common_moe_expert_plan_tensor tensor;
    tensor.layer = layer;
    tensor.name = name;
    tensor.type = "Q5_K";
    tensor.expert_first = 0;
    tensor.expert_last = 1;
    tensor.size_bytes = stride * 2;
    tensor.expert_stride_bytes = stride;
    tensor.ne = {64, 2, 2, 1};
    tensor.nb = {176, 176, stride, stride * 2};
    return tensor;
}

static common_moe_expert_plan make_valid_plan(uint32_t schema_version = 2) {
    common_moe_expert_plan plan;
    plan.schema_version = schema_version;
    plan.strategy = "greedy_hits_per_byte";
    plan.model_fingerprint = "sampled-fnv1a64:0123456789abcdef";
    plan.placement_fingerprint = "sha256:placement";
    plan.source_profiles = {"stats.csv"};
    plan.source_placement = "model.gguf";
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
    if (schema_version >= 2) {
        plan.tensor_manifest = {
            make_tensor(0, "blk.0.ffn_gate_exps.weight", 100),
            make_tensor(1, "blk.1.ffn_gate_exps.weight", 200),
        };
    }
    return plan;
}

static void test_validation() {
    auto plan = make_valid_plan();
    common_moe_expert_plan_expectation expectation;
    expectation.layer_count = 2;
    expectation.experts_per_layer = 2;
    expectation.model_fingerprint = "sampled-fnv1a64:0123456789abcdef";
    expectation.require_tensor_manifest = true;

    const auto valid = common_moe_expert_plan_validate(plan, expectation);
    require(valid.ok(), "valid plan rejected");

    plan.layers[0].gpu_experts = {1, 0};
    const auto invalid = common_moe_expert_plan_validate(plan, expectation);
    require(!invalid.ok(), "unsorted GPU experts accepted");
}

static void test_manifest_validation() {
    auto plan = make_valid_plan();
    plan.tensor_manifest[0].nb[2] += 1;
    require(!common_moe_expert_plan_validate(plan).ok(), "invalid manifest stride accepted");

    plan = make_valid_plan();
    plan.tensor_manifest.clear();
    require(!common_moe_expert_plan_validate(plan).ok(), "schema v2 without manifest accepted");

    auto legacy = make_valid_plan(1);
    common_moe_expert_plan_expectation expectation;
    expectation.require_tensor_manifest = true;
    require(!common_moe_expert_plan_validate(legacy, expectation).ok(),
        "legacy plan accepted when manifest required");
}

static void test_location_table() {
    const auto plan = make_valid_plan();
    common_moe_expert_placement placement;
    std::string error;
    require(common_moe_expert_placement_build(plan, placement, error), error.c_str());

    require(placement.layers.size() == 2, "placement layer count mismatch");
    require(placement.logical_expert_count == 4, "placement logical expert count mismatch");
    require(placement.gpu_expert_count == 2, "placement GPU expert count mismatch");
    require(placement.cpu_expert_count == 2, "placement CPU expert count mismatch");
    require(placement.gpu_bytes == 300, "placement GPU bytes mismatch");
    require(placement.cpu_bytes == 300, "placement CPU bytes mismatch");

    const auto * layer0 = placement.find_layer(0);
    require(layer0 != nullptr, "layer 0 lookup failed");
    require(layer0->expert_bytes == 100, "layer 0 expert bytes mismatch");
    require(layer0->gpu_experts == std::vector<uint32_t>({0}), "layer 0 GPU list mismatch");
    require(layer0->cpu_experts == std::vector<uint32_t>({1}), "layer 0 CPU list mismatch");
    require(layer0->global_to_local[0].backend == common_moe_expert_backend::gpu,
        "global expert 0 should map to GPU");
    require(layer0->global_to_local[0].local_index == 0, "GPU local index mismatch");
    require(layer0->global_to_local[1].backend == common_moe_expert_backend::cpu,
        "global expert 1 should map to CPU");
    require(layer0->global_to_local[1].local_index == 0, "CPU local index mismatch");
    require(placement.find_layer(2) == nullptr, "out-of-range layer lookup succeeded");

    auto invalid = plan;
    invalid.layers[0].gpu_bytes += 1;
    require(!common_moe_expert_placement_build(invalid, placement, error),
        "inconsistent per-layer GPU bytes accepted");

    auto legacy = make_valid_plan(1);
    require(!common_moe_expert_placement_build(legacy, placement, error),
        "schema v1 accepted for exclusive placement");
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
    require(loaded.tensor_manifest.size() == 2, "tensor manifest count changed");
    require(loaded.tensor_manifest[1].nb[2] == 200, "tensor manifest stride changed");
    require(loaded.layers[1].gpu_experts == std::vector<uint32_t>({0}), "GPU expert list changed");
    require(common_moe_expert_plan_validate(loaded).ok(), "round-tripped plan is invalid");
}

static void test_legacy_round_trip() {
    const std::string path = "test-moe-expert-plan-v1.tmp.json";
    const auto plan = make_valid_plan(1);
    std::string error;
    require(common_moe_expert_plan_save(path, plan, error), error.c_str());

    common_moe_expert_plan loaded;
    require(common_moe_expert_plan_load(path, loaded, error), error.c_str());
    std::remove(path.c_str());

    require(loaded.schema_version == 1, "legacy schema version changed");
    require(loaded.tensor_manifest.empty(), "legacy plan acquired a manifest");
    require(common_moe_expert_plan_validate(loaded).ok(), "legacy plan is invalid");
}

static void test_malformed_json() {
    const std::string path = "test-moe-expert-plan-invalid.tmp.json";
    {
        std::ofstream output(path);
        output << R"({"schema_version":2})";
    }
    common_moe_expert_plan plan;
    std::string error;
    require(!common_moe_expert_plan_load(path, plan, error), "malformed plan accepted");
    require(!error.empty(), "malformed plan returned no error");
    std::remove(path.c_str());
}

int main() {
    test_validation();
    test_manifest_validation();
    test_location_table();
    test_round_trip();
    test_legacy_round_trip();
    test_malformed_json();
    return 0;
}
