#pragma once

#include "ggml-cpp.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct llama_moe_compact_tensor_spec {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, GGML_MAX_DIMS> ne = {1, 1, 1, 1};
};

// Owns one backend-specific compact routed-expert tensor pool. The context
// stores tensor metadata and the buffer stores the quantized expert weights.
// A separate instance is used for CPU and GPU residency.
class llama_moe_compact_storage {
public:
    llama_moe_compact_storage() = default;

    bool create(
        ggml_backend_buffer_type_t buft,
        const std::vector<llama_moe_compact_tensor_spec> & specs,
        std::string & error);

    void clear() noexcept;

    bool empty() const noexcept;
    ggml_context * context() const noexcept;
    ggml_backend_buffer_t buffer() const noexcept;
    ggml_tensor * find_tensor(const std::string & name) const noexcept;

    uint64_t logical_tensor_bytes() const noexcept;
    size_t allocated_buffer_bytes() const noexcept;
    size_t tensor_count() const noexcept;

private:
    ggml_context_ptr context_;
    ggml_backend_buffer_ptr buffer_;
    std::vector<ggml_tensor *> tensors_;
    uint64_t logical_tensor_bytes_ = 0;
};
