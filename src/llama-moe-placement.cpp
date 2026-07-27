#include "llama-moe-placement.h"

namespace {

thread_local const llama_moe_load_placement_view * g_current_moe_load_placement = nullptr;

} // namespace

llama_moe_load_placement_scope::llama_moe_load_placement_scope(
        const llama_moe_load_placement_view * view) noexcept :
    previous_(g_current_moe_load_placement) {
    g_current_moe_load_placement = view;
}

llama_moe_load_placement_scope::~llama_moe_load_placement_scope() {
    g_current_moe_load_placement = previous_;
}

const llama_moe_load_placement_view * llama_moe_load_placement_current() noexcept {
    return g_current_moe_load_placement;
}
