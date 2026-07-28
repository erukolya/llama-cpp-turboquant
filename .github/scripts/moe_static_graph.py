from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:160]!r}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "src/llama-model.h",
        """    struct ggml_tensor * ffn_gate_up_exps_cpu = nullptr;
    struct ggml_tensor * ffn_gate_up_exps_gpu = nullptr;

    struct ggml_tensor * ffn_gate_inp_b    = nullptr;
""",
        """    struct ggml_tensor * ffn_gate_up_exps_cpu = nullptr;
    struct ggml_tensor * ffn_gate_up_exps_gpu = nullptr;
    struct ggml_tensor * ffn_route_map_cpu     = nullptr;
    struct ggml_tensor * ffn_route_map_gpu     = nullptr;

    struct ggml_tensor * ffn_gate_inp_b    = nullptr;
""",
    )

    replace_once(
        "src/models/qwen35moe.cpp",
        """            if (cpu_map == nullptr || gpu_map == nullptr) {
                throw std::runtime_error("allocated MoE route map tensor is missing");
            }
            ggml_backend_tensor_set(
""",
        """            if (cpu_map == nullptr || gpu_map == nullptr) {
                throw std::runtime_error("allocated MoE route map tensor is missing");
            }
            if (layer_placement.layer < 0 ||
                static_cast<size_t>(layer_placement.layer) >= layers.size()) {
                throw std::runtime_error("MoE route map layer index is out of range");
            }
            layers[layer_placement.layer].ffn_route_map_cpu = cpu_map;
            layers[layer_placement.layer].ffn_route_map_gpu = gpu_map;

            ggml_backend_tensor_set(
""",
    )

    old_graph = """    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);
"""

    new_graph = """    const auto & layer = model.layers[il];
    const bool static_placement =
        layer.ffn_route_map_cpu != nullptr || layer.ffn_route_map_gpu != nullptr;

    ggml_tensor * moe_out = nullptr;
    if (!static_placement) {
        moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    } else {
        GGML_ASSERT(layer.ffn_route_map_cpu != nullptr);
        GGML_ASSERT(layer.ffn_route_map_gpu != nullptr);

        const auto global_routing = build_moe_routing(
            cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            n_expert,
            n_expert_used,
            true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
            il,
            nullptr);

        const auto remap_routing = [&](ggml_tensor * route_map, const char * name) {
            GGML_ASSERT(route_map != nullptr);
            GGML_ASSERT(route_map->type == GGML_TYPE_I32);
            GGML_ASSERT(route_map->ne[0] == 1);
            GGML_ASSERT(route_map->ne[1] == n_expert);

            ggml_tensor * global_ids = ggml_reshape_1d(
                ctx0,
                global_routing.selected_experts,
                ggml_nelements(global_routing.selected_experts));
            ggml_tensor * local_ids = ggml_get_rows(ctx0, route_map, global_ids);
            local_ids = ggml_reshape_2d(ctx0, local_ids, n_expert_used, n_tokens);
            cb(local_ids, name, il);
            return llm_graph_moe_routing { local_ids, global_routing.weights };
        };

        const auto build_static_branch = [&] (
                ggml_tensor * route_map,
                ggml_tensor * up,
                ggml_tensor * gate,
                ggml_tensor * down,
                ggml_tensor * gate_up,
                const char * ids_name,
                const char * out_name) -> ggml_tensor * {
            if (down == nullptr) {
                GGML_ASSERT(up == nullptr && gate == nullptr && gate_up == nullptr);
                return nullptr;
            }

            GGML_ASSERT(down->ne[2] > 0);
            if (gate_up != nullptr) {
                GGML_ASSERT(up == nullptr && gate == nullptr);
            } else {
                GGML_ASSERT(up != nullptr && gate != nullptr);
            }

            const auto local_routing = remap_routing(route_map, ids_name);
            ggml_tensor * branch = build_moe_ffn_experts(
                cur,
                up, nullptr,
                gate, nullptr,
                down, nullptr,
                down->ne[2], n_expert_used,
                LLM_FFN_SILU, il,
                local_routing,
                gate_up, nullptr,
                nullptr, nullptr, nullptr,
                true);
            cb(branch, out_name, il);
            return branch;
        };

        ggml_tensor * cpu_out = build_static_branch(
            layer.ffn_route_map_cpu,
            layer.ffn_up_exps_cpu,
            layer.ffn_gate_exps_cpu,
            layer.ffn_down_exps_cpu,
            layer.ffn_gate_up_exps_cpu,
            "ffn_moe_cpu_local_ids",
            "ffn_moe_cpu_out");
        ggml_tensor * gpu_out = build_static_branch(
            layer.ffn_route_map_gpu,
            layer.ffn_up_exps_gpu,
            layer.ffn_gate_exps_gpu,
            layer.ffn_down_exps_gpu,
            layer.ffn_gate_up_exps_gpu,
            "ffn_moe_gpu_local_ids",
            "ffn_moe_gpu_out");

        GGML_ASSERT(cpu_out != nullptr || gpu_out != nullptr);
        if (cpu_out != nullptr && gpu_out != nullptr) {
            // Keep the joined output on the GPU side. Only the CPU activation
            // contribution crosses PCIe; routed expert weights never move.
            moe_out = ggml_add(ctx0, gpu_out, cpu_out);
            cb(moe_out, "ffn_moe_static_join", il);
        } else {
            moe_out = gpu_out != nullptr ? gpu_out : cpu_out;
        }
    }
    cb(moe_out, "ffn_moe_out", il);
"""

    replace_once("src/models/qwen35moe.cpp", old_graph, new_graph)


if __name__ == "__main__":
    main()
