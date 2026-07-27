#include "moe-expert-plan.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --plan FNAME [--layers N] [--experts-per-layer N] [--model-fingerprint VALUE] [--verbose]\n",
        argv0);
}

bool parse_non_negative_i32(const char * value, int32_t & result) {
    if (value == nullptr || *value == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
        return false;
    }
    result = static_cast<int32_t>(parsed);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string plan_path;
    common_moe_expert_plan_expectation expectation;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--verbose") {
            verbose = true;
            continue;
        }
        if (arg == "--plan" || arg == "--layers" || arg == "--experts-per-layer" || arg == "--model-fingerprint") {
            if (++i >= argc) {
                std::fprintf(stderr, "error: %s requires a value\n", arg.c_str());
                return 2;
            }
            const char * value = argv[i];
            if (arg == "--plan") {
                plan_path = value;
            } else if (arg == "--layers") {
                if (!parse_non_negative_i32(value, expectation.layer_count)) {
                    std::fprintf(stderr, "error: invalid --layers value: %s\n", value);
                    return 2;
                }
            } else if (arg == "--experts-per-layer") {
                if (!parse_non_negative_i32(value, expectation.experts_per_layer)) {
                    std::fprintf(stderr, "error: invalid --experts-per-layer value: %s\n", value);
                    return 2;
                }
            } else {
                expectation.model_fingerprint = value;
            }
            continue;
        }
        std::fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
        print_usage(argv[0]);
        return 2;
    }

    if (plan_path.empty()) {
        std::fprintf(stderr, "error: --plan is required\n");
        print_usage(argv[0]);
        return 2;
    }

    common_moe_expert_plan plan;
    std::string error;
    if (!common_moe_expert_plan_load(plan_path, plan, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    const auto validation = common_moe_expert_plan_validate(plan, expectation);
    for (const auto & warning : validation.warnings) {
        std::fprintf(stderr, "warning: %s\n", warning.c_str());
    }
    for (const auto & validation_error : validation.errors) {
        std::fprintf(stderr, "error: %s\n", validation_error.c_str());
    }
    if (!validation.ok()) {
        return 1;
    }

    std::printf("MoE placement plan is valid\n");
    std::printf("  schema:              %u\n", plan.schema_version);
    std::printf("  strategy:            %s\n", plan.strategy.c_str());
    std::printf("  layers:              %zu\n", plan.layers.size());
    std::printf("  logical experts:     %u\n", plan.logical_expert_count);
    std::printf("  selected experts:    %u\n", plan.selected_expert_count);
    std::printf("  selected MiB:        %.3f\n", static_cast<double>(plan.selected_bytes) / (1024.0 * 1024.0));
    std::printf("  budget MiB:          %.3f\n", static_cast<double>(plan.vram_budget_bytes) / (1024.0 * 1024.0));
    std::printf("  estimated GPU hits:  %.3f%%\n", plan.estimated_gpu_hit_rate * 100.0);
    std::printf("  placement hash:      %s\n", plan.placement_fingerprint.c_str());
    std::printf("  model hash:          %s\n", plan.model_fingerprint.empty() ? "<missing>" : plan.model_fingerprint.c_str());

    if (verbose) {
        for (const auto & layer : plan.layers) {
            std::printf(
                "  layer %d: GPU %zu/%u experts, %.3f MiB, estimated hits %.3f%%\n",
                layer.layer,
                layer.gpu_experts.size(),
                layer.expert_count,
                static_cast<double>(layer.gpu_bytes) / (1024.0 * 1024.0),
                layer.estimated_gpu_hit_rate * 100.0);
        }
    }

    return 0;
}
