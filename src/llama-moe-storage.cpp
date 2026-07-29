#include "llama-moe-storage.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

bool llama_moe_compact_storage::create(
        ggml_backend_buffer_type_t buft,
        const std::vector<llama_moe_compact_tensor_spec> & specs,
        std::string & error) {
    clear();

    if (buft == nullptr) {
        error = "compact storage buffer type is null";
        return false;
    }
    if (specs.empty()) {
        error = "compact storage requires at least one tensor";
        return false;
    }

    std::unordered_set<std::string> names;
    names.reserve(specs.size());
    for (const auto & spec : specs) {
        if (spec.name.empty()) {
            error = "compact tensor name is empty";
            return false;
        }
        if (!names.insert(spec.name).second) {
            error = "duplicate compact tensor name: " + spec.name;
            return false;
        }
        if (spec.type < 0 || spec.type >= GGML_TYPE_COUNT) {
            error = "compact tensor has an invalid ggml type: " + spec.name;
            return false;
        }
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (spec.ne[dim] <= 0) {
                error = "compact tensor has a non-positive dimension: " + spec.name;
                return false;
            }
        }
    }

    const size_t tensor_slots = specs.size() + 1;
    if (tensor_slots > std::numeric_limits<size_t>::max() / ggml_tensor_overhead()) {
        error = "compact tensor metadata size overflow";
        return false;
    }

    const ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * tensor_slots,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr context(ggml_init(params));
    if (!context) {
        error = "failed to create compact tensor metadata context";
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    tensors.reserve(specs.size());
    uint64_t logical_bytes = 0;
    for (const auto & spec : specs) {
        ggml_tensor * tensor = ggml_new_tensor_4d(
            context.get(), spec.type, spec.ne[0], spec.ne[1], spec.ne[2], spec.ne[3]);
        if (tensor == nullptr) {
            error = "failed to create compact tensor: " + spec.name;
            return false;
        }
        ggml_set_name(tensor, spec.name.c_str());

        const uint64_t tensor_bytes = ggml_nbytes(tensor);
        if (tensor_bytes > std::numeric_limits<uint64_t>::max() - logical_bytes) {
            error = "compact tensor byte count overflow";
            return false;
        }
        logical_bytes += tensor_bytes;
        tensors.push_back(tensor);
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(context.get(), buft));
    if (!buffer) {
        error = "failed to allocate compact tensor backend buffer";
        return false;
    }
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (auto * tensor : tensors) {
        if (tensor->buffer != buffer.get() || tensor->data == nullptr) {
            error = "compact tensor was not allocated in the requested backend buffer";
            return false;
        }
    }

    context_ = std::move(context);
    buffer_ = std::move(buffer);
    tensors_ = std::move(tensors);
    logical_tensor_bytes_ = logical_bytes;
    error.clear();
    return true;
}

void llama_moe_compact_storage::clear() noexcept {
    tensors_.clear();
    // Backends may retain tensor-allocation bookkeeping in the buffer. Release
    // the buffer while the metadata context and its tensor objects still exist.
    buffer_.reset();
    context_.reset();
    logical_tensor_bytes_ = 0;
}

bool llama_moe_compact_storage::empty() const noexcept {
    return tensors_.empty();
}

ggml_context * llama_moe_compact_storage::context() const noexcept {
    return context_.get();
}

ggml_backend_buffer_t llama_moe_compact_storage::buffer() const noexcept {
    return buffer_.get();
}

ggml_tensor * llama_moe_compact_storage::find_tensor(const std::string & name) const noexcept {
    const auto it = std::find_if(tensors_.begin(), tensors_.end(), [&](const ggml_tensor * tensor) {
        return name == ggml_get_name(tensor);
    });
    return it == tensors_.end() ? nullptr : *it;
}

uint64_t llama_moe_compact_storage::logical_tensor_bytes() const noexcept {
    return logical_tensor_bytes_;
}

size_t llama_moe_compact_storage::allocated_buffer_bytes() const noexcept {
    return buffer_ ? ggml_backend_buffer_get_size(buffer_.get()) : 0;
}

size_t llama_moe_compact_storage::tensor_count() const noexcept {
    return tensors_.size();
}
