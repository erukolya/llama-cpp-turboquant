#include "moe-expert-plan.h"

#include "ggml-backend.h"
#include "llama-model.h"
#include "llama-moe-placement.h"
#include "llama.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct options {
    std::string model_path;
    std::string plan_path;
    int32_t n_gpu_layers = -1;
};

struct placement_query_state {
    const common_moe_expert_placement * placement = nullptr;
};

struct model_deleter {
    void operator()(llama_model * model) const noexcept {
        llama_model_free(model);
    }
};

using model_ptr = std::unique_ptr<llama_model, model_deleter>;

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model MODEL.gguf --plan PLAN.json [--n-gpu-layers N]\n",
        argv0);
}

bool parse_i32(const char * value, int32_t & result) {
    if (value == nullptr || *value == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < -1 || parsed > INT32_MAX) {
        return false;
    }
    result = static_cast<int32_t>(parsed);
    return true;
}

bool parse_args(int argc, char ** argv, options & result) {
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--model" || arg == "--plan" || arg == "--n-gpu-layers") {
            if (++index >= argc) {
                std::fprintf(stderr, "error: %s requires a value\n", arg.c_str());
                return false;
            }
            if (arg == "--model") {
                result.model_path = argv[index];
            } else if (arg == "--plan") {
                result.plan_path = argv[index];
            } else if (!parse_i32(argv[index], result.n_gpu_layers)) {
                std::fprintf(stderr, "error: invalid --n-gpu-layers value: %s\n", argv[index]);
                return false;
            }
            continue;
        }
        std::fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
        return false;
    }

    if (result.model_path.empty() || result.plan_path.empty()) {
        std::fprintf(stderr, "error: --model and --plan are required\n");
        return false;
    }
    return true;
}

bool placement_query(
        const void * userdata,
        int32_t layer,
        uint32_t global_expert,
        llama_moe_load_location * location) {
    if (userdata == nullptr || location == nullptr) {
        return false;
    }
    const auto & state = *static_cast<const placement_query_state *>(userdata);
    if (state.placement == nullptr) {
        return false;
    }
    const auto * layer_placement = state.placement->find_layer(layer);
    if (layer_placement == nullptr || global_expert >= layer_placement->global_to_local.size()) {
        return false;
    }

    const auto & source = layer_placement->global_to_local[global_expert];
    location->backend = source.backend == common_moe_expert_backend::gpu ?
        llama_moe_load_backend::gpu : llama_moe_load_backend::cpu;
    location->local_index = source.local_index;
    return true;
}

llama_moe_load_placement_snapshot build_snapshot(
        const common_moe_expert_placement & placement) {
    if (placement.layers.empty()) {
        throw std::runtime_error("placement contains no layers");
    }

    const uint32_t experts_per_layer = placement.layers.front().expert_count;
    for (const auto & layer : placement.layers) {
        if (layer.expert_count != experts_per_layer) {
            throw std::runtime_error("U2 loader requires a uniform expert count per layer");
        }
    }

    placement_query_state state{&placement};
    llama_moe_load_placement_view view;
    view.layer_count = static_cast<uint32_t>(placement.layers.size());
    view.experts_per_layer = experts_per_layer;
    view.logical_expert_count = placement.logical_expert_count;
    view.query = placement_query;
    view.userdata = &state;

    llama_moe_load_placement_snapshot snapshot;
    std::string error;
    if (!llama_moe_load_placement_snapshot_build(view, snapshot, error)) {
        throw std::runtime_error("cannot build immutable placement snapshot: " + error);
    }
    return snapshot;
}

size_t count_packed_runtime_tensors(
        const llama_model & model,
        size_t placement_layer_count) {
    if (model.layers.size() < placement_layer_count) {
        throw std::runtime_error("loaded model has fewer layers than the placement");
    }

    size_t count = 0;
    for (size_t layer_index = 0; layer_index < placement_layer_count; ++layer_index) {
        const auto & layer = model.layers[layer_index];
        count += layer.ffn_gate_exps != nullptr ? 1 : 0;
        count += layer.ffn_up_exps != nullptr ? 1 : 0;
        count += layer.ffn_gate_up_exps != nullptr ? 1 : 0;
        count += layer.ffn_down_exps != nullptr ? 1 : 0;
    }
    return count;
}

void print_bytes(const char * key, uint64_t value) {
    std::printf("moe_u2: %s=%llu\n", key, static_cast<unsigned long long>(value));
}

} // namespace

int main(int argc, char ** argv) {
    options args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 2;
    }

    common_moe_expert_plan plan;
    std::string error;
    if (!common_moe_expert_plan_load(args.plan_path, plan, error)) {
        std::fprintf(stderr, "error: cannot load placement plan: %s\n", error.c_str());
        return 1;
    }

    common_moe_expert_placement placement;
    if (!common_moe_expert_placement_build(plan, placement, error)) {
        std::fprintf(stderr, "error: cannot build placement: %s\n", error.c_str());
        return 1;
    }

    llama_moe_load_placement_snapshot snapshot;
    try {
        snapshot = build_snapshot(placement);
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "error: %s\n", exception.what());
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();

    int result = 1;
    try {
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = args.n_gpu_layers;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        params.use_mmap = false;
        params.use_direct_io = false;
        params.no_alloc = false;

        llama_moe_load_placement_scope placement_scope(&snapshot);
        model_ptr model(llama_model_load_from_file(args.model_path.c_str(), params));
        if (!model) {
            throw std::runtime_error("llama_model_load_from_file returned null");
        }

        auto * model_base = dynamic_cast<llama_model_base *>(model.get());
        if (model_base == nullptr) {
            throw std::runtime_error("loaded model does not use llama_model_base");
        }
        if (model->arch != LLM_ARCH_QWEN35MOE) {
            throw std::runtime_error("U2 compact loader currently supports only Qwen3.5/3.6 MoE");
        }

        const auto & cpu_storage = model_base->moe_cpu_storage();
        const auto & gpu_storage = model_base->moe_gpu_storage();
        const uint64_t actual_cpu_logical = cpu_storage.logical_tensor_bytes();
        const uint64_t actual_gpu_logical = gpu_storage.logical_tensor_bytes();
        const uint64_t actual_total_logical = actual_cpu_logical + actual_gpu_logical;
        const uint64_t expected_total = placement.cpu_bytes + placement.gpu_bytes;
        const size_t packed_runtime_tensors =
            count_packed_runtime_tensors(*model, placement.layers.size());

        std::printf("moe_u2: loaded\n");
        std::printf("moe_u2: model_arch=%s\n", llm_arch_name(model->arch));
        std::printf("moe_u2: layers=%zu\n", placement.layers.size());
        std::printf("moe_u2: experts_total=%u\n", placement.logical_expert_count);
        std::printf("moe_u2: experts_cpu=%u\n", placement.cpu_expert_count);
        std::printf("moe_u2: experts_gpu=%u\n", placement.gpu_expert_count);
        std::printf("moe_u2: compact_tensors_cpu=%zu\n", cpu_storage.tensor_count());
        std::printf("moe_u2: compact_tensors_gpu=%zu\n", gpu_storage.tensor_count());
        print_bytes("expected_cpu_logical_bytes", placement.cpu_bytes);
        print_bytes("actual_cpu_logical_bytes", actual_cpu_logical);
        print_bytes("actual_cpu_allocated_bytes", cpu_storage.allocated_buffer_bytes());
        print_bytes("expected_gpu_logical_bytes", placement.gpu_bytes);
        print_bytes("actual_gpu_logical_bytes", actual_gpu_logical);
        print_bytes("actual_gpu_allocated_bytes", gpu_storage.allocated_buffer_bytes());
        print_bytes("expected_total_logical_bytes", expected_total);
        print_bytes("actual_total_logical_bytes", actual_total_logical);
        std::printf("moe_u2: packed_runtime_tensors=%zu\n", packed_runtime_tensors);

        if (cpu_storage.empty() || gpu_storage.empty()) {
            throw std::runtime_error("one of the compact expert tiers is empty");
        }
        if (actual_cpu_logical != placement.cpu_bytes) {
            throw std::runtime_error("compact CPU logical bytes differ from the plan");
        }
        if (actual_gpu_logical != placement.gpu_bytes) {
            throw std::runtime_error("compact GPU logical bytes differ from the plan");
        }
        if (actual_total_logical != expected_total) {
            throw std::runtime_error("compact total logical bytes differ from the routed expert bank");
        }
        if (packed_runtime_tensors != 0) {
            throw std::runtime_error("persistent packed routed-expert tensors remain allocated");
        }

        std::printf("moe_u2: accounting=ok\n");
        model.reset();
        std::printf("moe_u2: unloaded\n");
        result = 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "moe_u2: error=%s\n", exception.what());
    }

    llama_backend_free();
    return result;
}
