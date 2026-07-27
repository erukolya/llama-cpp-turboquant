#include "llama-moe-registry.h"

bool llama_moe_compact_registry::add(
        const ggml_tensor * source,
        int32_t layer,
        const llama_moe_load_layer_placement & placement,
        ggml_tensor ** cpu_slot,
        ggml_tensor ** gpu_slot,
        std::string & error) {
    llama_moe_packed_tensor_layout layout;
    if (!llama_moe_packed_tensor_layout_from_tensor(layer, source, layout, error)) {
        return false;
    }
    return add(layout, source->type, placement, cpu_slot, gpu_slot, error);
}
