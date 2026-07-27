#include "llama-moe-registry.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <stdexcept>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static llama_moe_load_layer_placement make_placement() {
    llama_moe_load_layer_placement layer;
    layer.layer = 0;
    layer.expert_count = 8;
    layer.cpu_expert_count = 4;
    layer.gpu_expert_count = 4;
    layer.global_to_local = {
        {llama_moe_load_backend::gpu, 0},
        {llama_moe_load_backend::gpu, 1},
        {llama_moe_load_backend::cpu, 0},
        {llama_moe_load_backend::cpu, 1},
        {llama_moe_load_backend::gpu, 2},
        {llama_moe_load_backend::cpu, 2},
        {llama_moe_load_backend::cpu, 3},
        {llama_moe_load_backend::gpu, 3},
    };
    return layer;
}

static llama_moe_packed_tensor_layout make_source(const std::string & name) {
    llama_moe_packed_tensor_layout source;
    source.layer = 0;
    source.name = name;
    source.ne = {8, 4, 8, 1};
    source.nb = {4, 32, 128, 1024};
    source.size_bytes = 1024;
    return source;
}

static ggml_tensor make_tensor_source(const char * name) {
    ggml_tensor source = {};
    source.type = GGML_TYPE_F32;
    source.ne[0] = 8;
    source.ne[1] = 4;
    source.ne[2] = 8;
    source.ne[3] = 1;
    source.nb[0] = 4;
    source.nb[1] = 32;
    source.nb[2] = 128;
    source.nb[3] = 1024;
    ggml_set_name(&source, name);
    return source;
}

int main() {
    const auto placement = make_placement();

    ggml_tensor * gate_cpu = nullptr;
    ggml_tensor * gate_gpu = nullptr;
    ggml_tensor * down_cpu = nullptr;
    ggml_tensor * down_gpu = nullptr;

    llama_moe_compact_registry registry;
    std::string error;
    const ggml_tensor gate_source = make_tensor_source("blk.0.ffn_gate_exps.weight");
    require(registry.add(&gate_source, 0, placement, &gate_cpu, &gate_gpu, error), error.c_str());
    require(registry.add(make_source("blk.0.ffn_down_exps.weight"), GGML_TYPE_F32,
        placement, &down_cpu, &down_gpu, error), error.c_str());
    require(registry.binding_count() == 2, "registry binding count mismatch");
    require(registry.cpu_bytes() == 1024 && registry.gpu_bytes() == 1024,
        "registry byte totals mismatch");

    llama_moe_compact_storage cpu_storage;
    llama_moe_compact_storage gpu_storage;
    require(registry.create_storages(
        ggml_backend_cpu_buffer_type(), ggml_backend_cpu_buffer_type(),
        cpu_storage, gpu_storage, error), error.c_str());

    require(gate_cpu != nullptr && gate_gpu != nullptr && down_cpu != nullptr && down_gpu != nullptr,
        "registry did not resolve destination tensor slots");
    require(gate_cpu->ne[2] == 4 && gate_gpu->ne[2] == 4,
        "registry compact expert dimensions mismatch");
    require(gate_cpu->buffer == cpu_storage.buffer() && down_cpu->buffer == cpu_storage.buffer(),
        "CPU bindings are not in one storage buffer");
    require(gate_gpu->buffer == gpu_storage.buffer() && down_gpu->buffer == gpu_storage.buffer(),
        "GPU bindings are not in one storage buffer");

    require(!registry.add(&gate_source, 0, placement, &gate_cpu, &gate_gpu, error),
        "duplicate metadata source must be rejected");

    registry.clear();
    require(registry.empty(), "registry clear failed");
    require(gate_cpu == nullptr && gate_gpu == nullptr && down_cpu == nullptr && down_gpu == nullptr,
        "registry clear did not invalidate destination slots");

    return 0;
}
