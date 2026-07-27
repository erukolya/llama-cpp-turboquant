from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


qwen_path = Path("src/models/qwen35moe.cpp")
qwen = qwen_path.read_text(encoding="utf-8")

qwen = replace_once(
    qwen,
    '#include "models.h"\n#include "llama-memory-recurrent.h"\n',
    '#include "models.h"\n#include "llama-memory-recurrent.h"\n#include "llama-moe-registry.h"\n\n#include "ggml-cpu.h"\n\n#include <vector>\n',
    "qwen includes",
)

qwen = replace_once(
    qwen,
    """    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight(\"blk.0.attn_norm.weight\") == nullptr);\n    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;\n\n""",
    """    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight(\"blk.0.attn_norm.weight\") == nullptr);\n    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;\n\n    const auto * static_placement = moe_placement();\n    llama_moe_compact_registry compact_registry;\n    std::vector<ggml_tensor *> compact_cpu_slots;\n    std::vector<ggml_tensor *> compact_gpu_slots;\n    ggml_backend_dev_t compact_gpu_device = nullptr;\n\n    if (static_placement != nullptr) {\n        if (mtp_only) {\n            throw std::runtime_error(\"static MoE placement is not supported for an MTP-only model\");\n        }\n        if (ml.use_mmap) {\n            throw std::runtime_error(\n                \"static MoE compact loading currently requires --no-mmap while V1.2 lifecycle integration is active\");\n        }\n        if (ml.use_direct_io) {\n            throw std::runtime_error(\n                \"static MoE compact loading currently does not support direct I/O\");\n        }\n\n        for (const auto & device : devices) {\n            if (device.is_meta || device.dev == nullptr ||\n                ggml_backend_dev_type(device.dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {\n                continue;\n            }\n            if (compact_gpu_device != nullptr) {\n                throw std::runtime_error(\n                    \"static MoE compact loading V1.2 supports exactly one GPU device\");\n            }\n            compact_gpu_device = device.dev;\n        }\n        if (compact_gpu_device == nullptr) {\n            throw std::runtime_error(\n                \"static MoE compact loading requires one dedicated GPU device\");\n        }\n\n        compact_cpu_slots.reserve(static_cast<size_t>(n_layer) * 3);\n        compact_gpu_slots.reserve(static_cast<size_t>(n_layer) * 3);\n    }\n\n    auto register_compact_source = [&](int il, const ggml_tensor * source) {\n        if (static_placement == nullptr || source == nullptr) {\n            throw std::runtime_error(\"invalid compact MoE source registration\");\n        }\n        const auto * layer_placement = static_placement->find_layer(il);\n        if (layer_placement == nullptr) {\n            throw std::runtime_error(\n                \"static MoE placement has no mapping for layer \" + std::to_string(il));\n        }\n\n        compact_cpu_slots.push_back(nullptr);\n        compact_gpu_slots.push_back(nullptr);\n        std::string compact_error;\n        if (!compact_registry.add(\n                source, il, *layer_placement,\n                &compact_cpu_slots.back(), &compact_gpu_slots.back(), compact_error)) {\n            throw std::runtime_error(\n                \"cannot register compact routed tensor '\" + std::string(source->name) + \"': \" + compact_error);\n        }\n    };\n\n""",
    "qwen compact setup",
)

qwen = replace_once(
    qwen,
    """        // Routed experts\n        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  \"weight\", il), { n_embd, n_expert }, flags);\n        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", il), { n_ff_exp, n_embd, n_expert }, flags);\n        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);\n\n""",
    """        // Routed experts\n        layer.ffn_gate_inp = create_tensor(\n            tn(LLM_TENSOR_FFN_GATE_INP, \"weight\", il), { n_embd, n_expert }, flags);\n\n        if (static_placement == nullptr) {\n            layer.ffn_down_exps = create_tensor(\n                tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", il),\n                { n_ff_exp, n_embd, n_expert }, flags);\n            create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);\n        } else {\n            const auto down_name = tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", il).str();\n            const ggml_tensor * down_source = ml.claim_tensor_for_slices(\n                down_name, std::vector<int64_t>{ n_ff_exp, n_embd, n_expert });\n            register_compact_source(il, down_source);\n\n            const auto gate_up_name = tn(LLM_TENSOR_FFN_GATE_UP_EXPS, \"weight\", il).str();\n            const ggml_tensor * gate_up_source = ml.claim_tensor_for_slices(\n                gate_up_name, std::vector<int64_t>{ n_embd, n_ff_exp * 2, n_expert }, false);\n            if (gate_up_source != nullptr) {\n                register_compact_source(il, gate_up_source);\n            } else {\n                const auto gate_name = tn(LLM_TENSOR_FFN_GATE_EXPS, \"weight\", il).str();\n                const auto up_name = tn(LLM_TENSOR_FFN_UP_EXPS, \"weight\", il).str();\n                register_compact_source(il, ml.claim_tensor_for_slices(\n                    gate_name, std::vector<int64_t>{ n_embd, n_ff_exp, n_expert }));\n                register_compact_source(il, ml.claim_tensor_for_slices(\n                    up_name, std::vector<int64_t>{ n_embd, n_ff_exp, n_expert }));\n            }\n        }\n\n""",
    "qwen trunk routed experts",
)

qwen = replace_once(
    qwen,
    """    for (int i = n_layer; i < n_layer_all; ++i) {\n        load_block_mtp(i);\n    }\n}\n\nstd::unique_ptr<llm_graph_context> llama_model_qwen35moe::build_arch_graph""",
    """    for (int i = n_layer; i < n_layer_all; ++i) {\n        load_block_mtp(i);\n    }\n\n    if (static_placement != nullptr) {\n        if (compact_registry.empty()) {\n            throw std::runtime_error(\"static MoE placement produced no compact routed tensors\");\n        }\n\n        std::string compact_error;\n        if (!compact_registry.create_storages(\n                ggml_backend_cpu_buffer_type(),\n                ggml_backend_dev_buffer_type(compact_gpu_device),\n                moe_cpu_storage(), moe_gpu_storage(), compact_error)) {\n            throw std::runtime_error(\"cannot allocate compact routed-expert pools: \" + compact_error);\n        }\n\n        const size_t previous_size_data = ml.size_data;\n        if (ml.size_data == 0) {\n            ml.size_data = ml.n_bytes;\n        }\n        try {\n            for (const auto & binding : compact_registry.bindings()) {\n                if (!binding.cpu_name.empty()) {\n                    if (binding.cpu_slot == nullptr || *binding.cpu_slot == nullptr) {\n                        throw std::runtime_error(\n                            \"compact CPU destination is missing for '\" + binding.source_name + \"'\");\n                    }\n                    ml.load_tensor_slices(\n                        *binding.cpu_slot, binding.source_name, binding.pool_plan.cpu_spans);\n                }\n                if (!binding.gpu_name.empty()) {\n                    if (binding.gpu_slot == nullptr || *binding.gpu_slot == nullptr) {\n                        throw std::runtime_error(\n                            \"compact GPU destination is missing for '\" + binding.source_name + \"'\");\n                    }\n                    ml.load_tensor_slices(\n                        *binding.gpu_slot, binding.source_name, binding.pool_plan.gpu_spans);\n                }\n            }\n        } catch (...) {\n            ml.size_data = previous_size_data;\n            throw;\n        }\n        ml.size_data = previous_size_data;\n\n        LLAMA_LOG_INFO(\n            \"%s: compact routed experts loaded: tensors=%zu, CPU=%.2f MiB logical / %.2f MiB allocated, \"\n            \"GPU=%.2f MiB logical / %.2f MiB allocated, packed runtime bytes=0\\n\",\n            __func__, compact_registry.binding_count(),\n            moe_cpu_storage().logical_tensor_bytes() / 1024.0 / 1024.0,\n            moe_cpu_storage().allocated_buffer_bytes() / 1024.0 / 1024.0,\n            moe_gpu_storage().logical_tensor_bytes() / 1024.0 / 1024.0,\n            moe_gpu_storage().allocated_buffer_bytes() / 1024.0 / 1024.0);\n    }\n}\n\nstd::unique_ptr<llm_graph_context> llama_model_qwen35moe::build_arch_graph""",
    "qwen compact allocation and load",
)

qwen_path.write_text(qwen, encoding="utf-8")

server_path = Path("tools/server/server.cpp")
server = server_path.read_text(encoding="utf-8")

server = replace_once(
    server,
    """        if (g_server_moe_plan && !g_server_moe_plan->validate(model)) {\n            return false;\n        }\n""",
    """        if (g_server_moe_plan && g_server_moe_plan->dry_run() &&\n            !g_server_moe_plan->validate(model)) {\n            return false;\n        }\n""",
    "server post-load validation",
)

server = replace_once(
    server,
    """        } else {\n            g_server_moe_load_placement = std::move(load_placement);\n        }\n\n        std::fprintf(stderr,\n            \"moe_plan: enabled, input '%s', strict=%s, dry_run=%s, load_scope=%s\\n\",\n            g_server_moe_plan->plan_path().c_str(),\n            g_server_moe_plan->strict() ? \"true\" : \"false\",\n            g_server_moe_plan->dry_run() ? \"true\" : \"false\",\n            g_server_moe_load_placement ? \"ready\" : \"disabled\");\n""",
    """        } else if (!plan_options.dry_run) {\n            g_server_moe_load_placement = std::move(load_placement);\n        }\n\n        std::fprintf(stderr,\n            \"moe_plan: enabled, input '%s', strict=%s, dry_run=%s, load_scope=%s\\n\",\n            g_server_moe_plan->plan_path().c_str(),\n            g_server_moe_plan->strict() ? \"true\" : \"false\",\n            g_server_moe_plan->dry_run() ? \"true\" : \"false\",\n            g_server_moe_load_placement ? \"ready\" :\n                (plan_options.dry_run ? \"dry-run-disabled\" : \"disabled\"));\n""",
    "server dry-run load scope",
)

server_path.write_text(server, encoding="utf-8")
