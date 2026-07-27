#pragma once

#include "llama-moe-pool-plan.h"
#include "llama-moe-storage.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_moe_compact_tensor_binding {
    std::string source_name;
    std::string cpu_name;
    std::string gpu_name;
    llama_moe_compact_tensor_pool_plan pool_plan;
    ggml_tensor ** cpu_slot = nullptr;
    ggml_tensor ** gpu_slot = nullptr;
};

// Collects all compact routed-expert tensors before allocation, then creates
// exactly one CPU storage buffer and one GPU storage buffer for the model.
class llama_moe_compact_registry {
public:
    bool add(
        const llama_moe_packed_tensor_layout & source,
        ggml_type type,
        const llama_moe_load_layer_placement & placement,
        ggml_tensor ** cpu_slot,
        ggml_tensor ** gpu_slot,
        std::string & error);

    bool create_storages(
        ggml_backend_buffer_type_t cpu_buft,
        ggml_backend_buffer_type_t gpu_buft,
        llama_moe_compact_storage & cpu_storage,
        llama_moe_compact_storage & gpu_storage,
        std::string & error);

    void clear() noexcept;

    bool empty() const noexcept;
    size_t binding_count() const noexcept;
    uint64_t cpu_bytes() const noexcept;
    uint64_t gpu_bytes() const noexcept;
    const std::vector<llama_moe_compact_tensor_binding> & bindings() const noexcept;

private:
    std::vector<llama_moe_compact_tensor_spec> cpu_specs_;
    std::vector<llama_moe_compact_tensor_spec> gpu_specs_;
    std::vector<llama_moe_compact_tensor_binding> bindings_;
    uint64_t cpu_bytes_ = 0;
    uint64_t gpu_bytes_ = 0;
};
