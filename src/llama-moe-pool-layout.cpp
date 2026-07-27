#include "llama-moe-pool-plan.h"

#include "ggml.h"

bool llama_moe_packed_tensor_layout_from_tensor(
        int32_t layer,
        const ggml_tensor * tensor,
        llama_moe_packed_tensor_layout & layout,
        std::string & error) {
    if (layer < 0) {
        error = "packed tensor layer is negative";
        return false;
    }
    if (tensor == nullptr) {
        error = "packed tensor metadata is null";
        return false;
    }

    const char * name = ggml_get_name(tensor);
    if (name == nullptr || name[0] == '\0') {
        error = "packed tensor metadata has no name";
        return false;
    }

    llama_moe_packed_tensor_layout built;
    built.layer = layer;
    built.name = name;
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor->ne[dim] <= 0) {
            error = "packed tensor metadata has a non-positive dimension";
            return false;
        }
        if (tensor->nb[dim] == 0) {
            error = "packed tensor metadata has a zero stride";
            return false;
        }
        built.ne[dim] = tensor->ne[dim];
        built.nb[dim] = tensor->nb[dim];
    }
    built.size_bytes = ggml_nbytes(tensor);
    if (built.size_bytes == 0) {
        error = "packed tensor metadata has zero bytes";
        return false;
    }

    layout = std::move(built);
    error.clear();
    return true;
}
