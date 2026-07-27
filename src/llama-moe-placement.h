#pragma once

#include <cstdint>

// Internal model-load-only placement contract. It deliberately contains no
// JSON/common/server types so the model core remains independent from tools.
enum class llama_moe_load_backend : uint8_t {
    cpu = 0,
    gpu = 1,
};

struct llama_moe_load_location {
    llama_moe_load_backend backend = llama_moe_load_backend::cpu;
    uint32_t local_index = 0;
};

using llama_moe_load_query = bool (*)(
    const void * userdata,
    int32_t layer,
    uint32_t global_expert,
    llama_moe_load_location * location);

struct llama_moe_load_placement_view {
    uint32_t layer_count = 0;
    uint32_t logical_expert_count = 0;
    llama_moe_load_query query = nullptr;
    const void * userdata = nullptr;
};

// Makes a placement view visible only on the current model-loading thread.
// Nested scopes are supported and restore the previous view on destruction.
class llama_moe_load_placement_scope {
public:
    explicit llama_moe_load_placement_scope(const llama_moe_load_placement_view * view) noexcept;
    ~llama_moe_load_placement_scope();

    llama_moe_load_placement_scope(const llama_moe_load_placement_scope &) = delete;
    llama_moe_load_placement_scope & operator=(const llama_moe_load_placement_scope &) = delete;

private:
    const llama_moe_load_placement_view * previous_ = nullptr;
};

const llama_moe_load_placement_view * llama_moe_load_placement_current() noexcept;
