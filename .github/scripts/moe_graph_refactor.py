from __future__ import annotations

from pathlib import Path
import re


def function_range(text: str, signature: str) -> tuple[int, int]:
    start = text.find(signature)
    if start < 0:
        raise RuntimeError(f"function signature not found: {signature!r}")
    if text.find(signature, start + 1) >= 0:
        raise RuntimeError(f"function signature is not unique: {signature!r}")

    opening = text.find("{", start)
    if opening < 0:
        raise RuntimeError(f"opening brace not found for: {signature!r}")

    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1

    raise RuntimeError(f"closing brace not found for: {signature!r}")


def main() -> None:
    header_path = Path("src/llama-graph.h")
    header = header_path.read_text(encoding="utf-8")

    if "struct llm_graph_moe_routing" in header:
        raise RuntimeError("MoE graph refactor is already present in llama-graph.h")

    qkv_start = header.find("struct llm_graph_qkv {")
    qkv_end = header.find("};", qkv_start)
    if qkv_start < 0 or qkv_end < 0:
        raise RuntimeError("llm_graph_qkv bounds not found")
    qkv_end += 2

    routing_struct = """

struct llm_graph_moe_routing {
    ggml_tensor * selected_experts = nullptr; // [n_expert_used, n_tokens], global or local IDs
    ggml_tensor * weights = nullptr;          // [1, n_expert_used, n_tokens]
};

constexpr int32_t LLM_MOE_MUL_MAT_ID_MISSING_MAGIC = 0x4D4F4553;
"""
    header = header[:qkv_end] + routing_struct + header[qkv_end:]

    declaration_pattern = re.compile(
        r"(\s+ggml_tensor \* build_lora_mm_id\(\n"
        r"\s+ggml_tensor \* w,\s+// ggml_tensor \* as\n"
        r"\s+ggml_tensor \* cur,\s+// ggml_tensor \* b\n"
        r"\s+ggml_tensor \* ids)(\) const;)"
    )
    header, count = declaration_pattern.subn(
        r"\1,\n                      bool   allow_missing_ids = false\2",
        header,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"build_lora_mm_id declaration matches: {count}")

    moe_marker = "    // build MoE FFN without bias tensors\n    ggml_tensor * build_moe_ffn("
    marker_index = header.find(moe_marker)
    if marker_index < 0:
        raise RuntimeError("MoE declaration marker not found")

    declarations = """    llm_graph_moe_routing build_moe_routing(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * gate_inp_b,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
                    bool   norm_w,
                   float   w_scale,
        llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr) const;

    ggml_tensor * build_moe_ffn_experts(
             ggml_tensor * cur,
             ggml_tensor * up_exps,
             ggml_tensor * up_exps_b,
             ggml_tensor * gate_exps,
             ggml_tensor * gate_exps_b,
             ggml_tensor * down_exps,
             ggml_tensor * down_exps_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                     int   il,
        const llm_graph_moe_routing & routing,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * gate_up_exps_b = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
                    bool   allow_missing_ids = false) const;

"""
    header = header[:marker_index] + declarations + header[marker_index:]
    header_path.write_text(header, encoding="utf-8")

    cpp_path = Path("src/llama-graph.cpp")
    cpp = cpp_path.read_text(encoding="utf-8")

    lora_signature = "ggml_tensor * llm_graph_context::build_lora_mm_id("
    lora_start, lora_end = function_range(cpp, lora_signature)
    lora_function = """ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
                  bool   allow_missing_ids) const {
    const auto mul_mat_id = [&](ggml_tensor * weight, ggml_tensor * input) {
        ggml_tensor * result = ggml_mul_mat_id(ctx0, weight, input, ids);
        if (allow_missing_ids) {
            const int32_t magic = LLM_MOE_MUL_MAT_ID_MISSING_MAGIC;
            std::memcpy(result->op_params, &magic, sizeof(magic));
        }
        return result;
    };

    ggml_tensor * res = mul_mat_id(w, cur);
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = mul_mat_id(lw->b, mul_mat_id(lw->a, cur));
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}"""
    cpp = cpp[:lora_start] + lora_function + cpp[lora_end:]

    full_signature = """ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,"""
    full_start, full_end = function_range(cpp, full_signature)
    original = cpp[full_start:full_end]
    opening = original.find("{")
    signature = original[:opening]
    body = original[opening + 1 : -1]

    execution_marker = "    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);\n"
    split = body.find(execution_marker)
    if split < 0:
        raise RuntimeError("routing/execution split marker not found")

    router_body = body[:split]
    execution_body = body[split:]
    router_body = router_body.replace("    const int64_t n_embd   = cur->ne[0];\n", "", 1)
    router_body = router_body.replace(
        "    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN\n",
        "",
        1,
    )

    routing_function = """llm_graph_moe_routing llm_graph_context::build_moe_routing(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                  int   il,
         ggml_tensor * probs_in) const {
""" + router_body + """
    return { selected_experts, weights };
}

"""

    execution_function = """ggml_tensor * llm_graph_context::build_moe_ffn_experts(
         ggml_tensor * cur,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                  int   il,
    const llm_graph_moe_routing & routing,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
                bool   allow_missing_ids) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4;
    ggml_tensor * selected_experts = routing.selected_experts;
    ggml_tensor * weights = routing.weights;

    GGML_ASSERT(selected_experts != nullptr && weights != nullptr);
    GGML_ASSERT(selected_experts->ne[0] == n_expert_used);
    if (allow_missing_ids) {
        GGML_ASSERT(!weight_before_ffn);
        GGML_ASSERT(up_exps_b == nullptr && gate_exps_b == nullptr && down_exps_b == nullptr);
        GGML_ASSERT(gate_up_exps_b == nullptr);
        GGML_ASSERT(up_exps_s == nullptr && gate_exps_s == nullptr && down_exps_s == nullptr);
    }

""" + execution_body + """
}

"""

    for name in ("gate_up_exps", "up_exps", "gate_exps", "down_exps"):
        execution_function = execution_function.replace(
            f"build_lora_mm_id({name}, cur, selected_experts)",
            f"build_lora_mm_id({name}, cur, selected_experts, allow_missing_ids)",
        )

    wrapper = signature + """{
    const auto routing = build_moe_routing(
        cur, gate_inp, gate_inp_b, exp_probs_b,
        n_expert, n_expert_used, norm_w, w_scale, gating_op, il, probs_in);
    return build_moe_ffn_experts(
        cur,
        up_exps, up_exps_b,
        gate_exps, gate_exps_b,
        down_exps, down_exps_b,
        n_expert, n_expert_used, type_op, il, routing,
        gate_up_exps, gate_up_exps_b,
        up_exps_s, gate_exps_s, down_exps_s,
        false);
}"""

    cpp = cpp[:full_start] + routing_function + execution_function + wrapper + cpp[full_end:]
    cpp_path.write_text(cpp, encoding="utf-8")


if __name__ == "__main__":
    main()
