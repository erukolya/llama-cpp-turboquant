#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

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

// Temporary adapter used only while constructing an immutable core snapshot.
struct llama_moe_load_placement_view {
    uint32_t layer_count = 0;
    uint32_t experts_per_layer = 0;
    uint32_t logical_expert_count = 0;
    llama_moe_load_query query = nullptr;
    const void * userdata = nullptr;
};

struct llama_moe_load_layer_placement {
    int32_t layer = -1;
    uint32_t expert_count = 0;
    uint32_t cpu_expert_count = 0;
    uint32_t gpu_expert_count = 0;
    std::vector<llama_moe_load_location> global_to_local;

    LLAMA_API const llama_moe_load_location * find(uint32_t global_expert) const noexcept;
};

// Deep-copied immutable placement owned by the model-loading core. The source
// view and its userdata may be destroyed immediately after construction.
struct llama_moe_load_placement_snapshot {
    uint32_t logical_expert_count = 0;
    uint32_t cpu_expert_count = 0;
    uint32_t gpu_expert_count = 0;
    std::vector<llama_moe_load_layer_placement> layers;

    LLAMA_API bool empty() const noexcept;
    LLAMA_API const llama_moe_load_layer_placement * find_layer(int32_t layer) const noexcept;
    LLAMA_API bool query(int32_t layer, uint32_t global_expert, llama_moe_load_location * location) const noexcept;
};

LLAMA_API bool llama_moe_load_placement_snapshot_build(
    const llama_moe_load_placement_view & view,
    llama_moe_load_placement_snapshot & snapshot,
    std::string & error);

// Makes an immutable placement snapshot visible only on the current
// model-loading thread. Nested scopes restore the previous snapshot.
class llama_moe_load_placement_scope {
public:
    LLAMA_API explicit llama_moe_load_placement_scope(const llama_moe_load_placement_snapshot * snapshot) noexcept;
    LLAMA_API ~llama_moe_load_placement_scope();

    llama_moe_load_placement_scope(const llama_moe_load_placement_scope &) = delete;
    llama_moe_load_placement_scope & operator=(const llama_moe_load_placement_scope &) = delete;

private:
    const llama_moe_load_placement_snapshot * previous_ = nullptr;
};

LLAMA_API const llama_moe_load_placement_snapshot * llama_moe_load_placement_current() noexcept;
