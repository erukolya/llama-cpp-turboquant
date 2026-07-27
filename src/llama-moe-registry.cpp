#include "llama-moe-registry.h"

#include <limits>

bool llama_moe_compact_registry::add(
        const llama_moe_packed_tensor_layout & source,
        ggml_type type,
        const llama_moe_load_layer_placement & placement,
        ggml_tensor ** cpu_slot,
        ggml_tensor ** gpu_slot,
        std::string & error) {
    if (cpu_slot == nullptr || gpu_slot == nullptr) {
        error = "compact tensor destination slot is null";
        return false;
    }
    *cpu_slot = nullptr;
    *gpu_slot = nullptr;

    if (type < 0 || type >= GGML_TYPE_COUNT) {
        error = "compact tensor source has an invalid ggml type";
        return false;
    }
    for (const auto & binding : bindings_) {
        if (binding.source_name == source.name) {
            error = "duplicate compact tensor source: " + source.name;
            return false;
        }
    }

    llama_moe_compact_tensor_pool_plan pool_plan;
    if (!llama_moe_compact_tensor_pool_plan_build(source, placement, pool_plan, error)) {
        return false;
    }

    llama_moe_compact_tensor_binding binding;
    binding.source_name = source.name;
    binding.pool_plan = pool_plan;
    binding.cpu_slot = cpu_slot;
    binding.gpu_slot = gpu_slot;

    if (pool_plan.cpu_expert_count > 0) {
        binding.cpu_name = source.name + ".cpu";
        cpu_specs_.push_back({binding.cpu_name, type, pool_plan.cpu_ne});
    }
    if (pool_plan.gpu_expert_count > 0) {
        binding.gpu_name = source.name + ".gpu";
        gpu_specs_.push_back({binding.gpu_name, type, pool_plan.gpu_ne});
    }

    if (pool_plan.cpu_bytes > std::numeric_limits<uint64_t>::max() - cpu_bytes_ ||
        pool_plan.gpu_bytes > std::numeric_limits<uint64_t>::max() - gpu_bytes_) {
        if (!binding.cpu_name.empty()) {
            cpu_specs_.pop_back();
        }
        if (!binding.gpu_name.empty()) {
            gpu_specs_.pop_back();
        }
        error = "compact registry byte count overflow";
        return false;
    }

    cpu_bytes_ += pool_plan.cpu_bytes;
    gpu_bytes_ += pool_plan.gpu_bytes;
    bindings_.push_back(std::move(binding));
    error.clear();
    return true;
}

bool llama_moe_compact_registry::create_storages(
        ggml_backend_buffer_type_t cpu_buft,
        ggml_backend_buffer_type_t gpu_buft,
        llama_moe_compact_storage & cpu_storage,
        llama_moe_compact_storage & gpu_storage,
        std::string & error) {
    cpu_storage.clear();
    gpu_storage.clear();

    if (bindings_.empty()) {
        error = "compact registry contains no tensors";
        return false;
    }
    if (!cpu_specs_.empty() && !cpu_storage.create(cpu_buft, cpu_specs_, error)) {
        return false;
    }
    if (!gpu_specs_.empty() && !gpu_storage.create(gpu_buft, gpu_specs_, error)) {
        cpu_storage.clear();
        return false;
    }

    if (cpu_storage.logical_tensor_bytes() != cpu_bytes_ ||
        gpu_storage.logical_tensor_bytes() != gpu_bytes_) {
        cpu_storage.clear();
        gpu_storage.clear();
        error = "compact storage bytes differ from registry plan";
        return false;
    }

    for (auto & binding : bindings_) {
        if (!binding.cpu_name.empty()) {
            *binding.cpu_slot = cpu_storage.find_tensor(binding.cpu_name);
            if (*binding.cpu_slot == nullptr) {
                error = "compact CPU tensor was not created: " + binding.cpu_name;
                cpu_storage.clear();
                gpu_storage.clear();
                return false;
            }
        }
        if (!binding.gpu_name.empty()) {
            *binding.gpu_slot = gpu_storage.find_tensor(binding.gpu_name);
            if (*binding.gpu_slot == nullptr) {
                error = "compact GPU tensor was not created: " + binding.gpu_name;
                cpu_storage.clear();
                gpu_storage.clear();
                return false;
            }
        }
    }

    error.clear();
    return true;
}

void llama_moe_compact_registry::clear() noexcept {
    for (auto & binding : bindings_) {
        if (binding.cpu_slot != nullptr) {
            *binding.cpu_slot = nullptr;
        }
        if (binding.gpu_slot != nullptr) {
            *binding.gpu_slot = nullptr;
        }
    }
    cpu_specs_.clear();
    gpu_specs_.clear();
    bindings_.clear();
    cpu_bytes_ = 0;
    gpu_bytes_ = 0;
}

bool llama_moe_compact_registry::empty() const noexcept {
    return bindings_.empty();
}

size_t llama_moe_compact_registry::binding_count() const noexcept {
    return bindings_.size();
}

uint64_t llama_moe_compact_registry::cpu_bytes() const noexcept {
    return cpu_bytes_;
}

uint64_t llama_moe_compact_registry::gpu_bytes() const noexcept {
    return gpu_bytes_;
}

const std::vector<llama_moe_compact_tensor_binding> & llama_moe_compact_registry::bindings() const noexcept {
    return bindings_;
}
