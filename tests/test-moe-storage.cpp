#include "llama-moe-storage.h"

#include "ggml-cpu.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    std::vector<llama_moe_compact_tensor_spec> specs = {
        {"blk.0.ffn_gate_exps.weight.cpu", GGML_TYPE_Q4_0, {32, 8, 3, 1}},
        {"blk.0.ffn_down_exps.weight.cpu", GGML_TYPE_Q4_0, {32, 8, 3, 1}},
    };

    llama_moe_compact_storage storage;
    std::string error;
    require(storage.create(ggml_backend_cpu_buffer_type(), specs, error), error.c_str());
    require(!storage.empty(), "storage must not be empty");
    require(storage.tensor_count() == 2, "storage tensor count mismatch");
    require(storage.buffer() != nullptr, "storage buffer is null");
    require(storage.allocated_buffer_bytes() >= storage.logical_tensor_bytes(),
        "backend allocation is smaller than logical tensor bytes");

    auto * gate = storage.find_tensor(specs[0].name);
    auto * down = storage.find_tensor(specs[1].name);
    require(gate != nullptr && down != nullptr, "compact tensor lookup failed");
    require(gate->buffer == storage.buffer() && down->buffer == storage.buffer(),
        "compact tensors do not share the storage buffer");
    require(gate->data != nullptr && down->data != nullptr, "compact tensors are not allocated");
    require(storage.find_tensor("missing") == nullptr, "missing tensor lookup must fail");

    std::vector<uint8_t> pattern(ggml_nbytes(gate), 0x5a);
    ggml_backend_tensor_set(gate, pattern.data(), 0, pattern.size());
    std::vector<uint8_t> roundtrip(pattern.size(), 0);
    ggml_backend_tensor_get(gate, roundtrip.data(), 0, roundtrip.size());
    require(roundtrip == pattern, "compact tensor backend round-trip mismatch");

    llama_moe_compact_storage duplicate;
    auto duplicate_specs = specs;
    duplicate_specs[1].name = duplicate_specs[0].name;
    require(!duplicate.create(ggml_backend_cpu_buffer_type(), duplicate_specs, error),
        "duplicate tensor names must be rejected");

    llama_moe_compact_storage invalid_shape;
    auto invalid_specs = specs;
    invalid_specs[0].ne[2] = 0;
    require(!invalid_shape.create(ggml_backend_cpu_buffer_type(), invalid_specs, error),
        "zero expert count must be rejected");

    storage.clear();
    require(storage.empty() && storage.buffer() == nullptr && storage.context() == nullptr,
        "storage clear did not release ownership");

    return 0;
}
