from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "src/llama-model.h",
        """    struct ggml_tensor * ffn_gate_up_exps  = nullptr;
    struct ggml_tensor * ffn_gate_inp_b    = nullptr;
""",
        """    struct ggml_tensor * ffn_gate_up_exps  = nullptr;

    // Static MoE compact tensors. These are metadata pointers into the
    // model-owned CPU/GPU storage buffers, not additional weight copies.
    struct ggml_tensor * ffn_gate_exps_cpu    = nullptr;
    struct ggml_tensor * ffn_gate_exps_gpu    = nullptr;
    struct ggml_tensor * ffn_down_exps_cpu    = nullptr;
    struct ggml_tensor * ffn_down_exps_gpu    = nullptr;
    struct ggml_tensor * ffn_up_exps_cpu      = nullptr;
    struct ggml_tensor * ffn_up_exps_gpu      = nullptr;
    struct ggml_tensor * ffn_gate_up_exps_cpu = nullptr;
    struct ggml_tensor * ffn_gate_up_exps_gpu = nullptr;

    struct ggml_tensor * ffn_gate_inp_b    = nullptr;
""",
    )

    replace_once(
        "src/models/qwen35moe.cpp",
        """    llama_moe_compact_registry compact_registry;
    std::vector<ggml_tensor *> compact_cpu_slots;
    std::vector<ggml_tensor *> compact_gpu_slots;
    ggml_backend_dev_t compact_gpu_device = nullptr;
""",
        """    llama_moe_compact_registry compact_registry;
    ggml_backend_dev_t compact_gpu_device = nullptr;
""",
    )

    replace_once(
        "src/models/qwen35moe.cpp",
        """
        compact_cpu_slots.reserve(static_cast<size_t>(n_layer) * 3);
        compact_gpu_slots.reserve(static_cast<size_t>(n_layer) * 3);
    }

    auto register_compact_source = [&](int il, const ggml_tensor * source) {
""",
        """
    }

    auto register_compact_source = [&] (
            int il,
            const ggml_tensor * source,
            ggml_tensor ** cpu_slot,
            ggml_tensor ** gpu_slot) {
""",
    )

    replace_once(
        "src/models/qwen35moe.cpp",
        """
        compact_cpu_slots.push_back(nullptr);
        compact_gpu_slots.push_back(nullptr);
        std::string compact_error;
        if (!compact_registry.add(
                source, il, *layer_placement,
                &compact_cpu_slots.back(), &compact_gpu_slots.back(), compact_error)) {
""",
        """
        if (cpu_slot == nullptr || gpu_slot == nullptr) {
            throw std::runtime_error("compact MoE layer destination slot is null");
        }
        std::string compact_error;
        if (!compact_registry.add(
                source, il, *layer_placement,
                cpu_slot, gpu_slot, compact_error)) {
""",
    )

    replace_once(
        "src/models/qwen35moe.cpp",
        """            register_compact_source(il, down_source);

            const auto gate_up_name""",
        """            register_compact_source(
                il, down_source,
                &layer.ffn_down_exps_cpu, &layer.ffn_down_exps_gpu);

            const auto gate_up_name""",
    )

    replace_once(
        "src/models/qwen35moe.cpp",
        """            if (gate_up_source != nullptr) {
                register_compact_source(il, gate_up_source);
            } else {
""",
        """            if (gate_up_source != nullptr) {
                register_compact_source(
                    il, gate_up_source,
                    &layer.ffn_gate_up_exps_cpu, &layer.ffn_gate_up_exps_gpu);
            } else {
""",
    )

    replace_once(
        "src/models/qwen35moe.cpp",
        """                register_compact_source(il, ml.claim_tensor_for_slices(
                    gate_name, std::vector<int64_t>{ n_embd, n_ff_exp, n_expert }));
                register_compact_source(il, ml.claim_tensor_for_slices(
                    up_name, std::vector<int64_t>{ n_embd, n_ff_exp, n_expert }));
""",
        """                register_compact_source(
                    il,
                    ml.claim_tensor_for_slices(
                        gate_name, std::vector<int64_t>{ n_embd, n_ff_exp, n_expert }),
                    &layer.ffn_gate_exps_cpu, &layer.ffn_gate_exps_gpu);
                register_compact_source(
                    il,
                    ml.claim_tensor_for_slices(
                        up_name, std::vector<int64_t>{ n_embd, n_ff_exp, n_expert }),
                    &layer.ffn_up_exps_cpu, &layer.ffn_up_exps_gpu);
""",
    )


if __name__ == "__main__":
    main()
