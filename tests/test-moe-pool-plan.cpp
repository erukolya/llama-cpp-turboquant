#include "llama-moe-pool-plan.h"

#include "ggml.h"

#include <stdexcept>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static llama_moe_load_layer_placement make_placement() {
    llama_moe_load_layer_placement layer;
    layer.layer = 3;
    layer.expert_count = 8;
    layer.cpu_expert_count = 4;
    layer.gpu_expert_count = 4;
    layer.global_to_local = {
        {llama_moe_load_backend::gpu, 0},
        {llama_moe_load_backend::gpu, 1},
        {llama_moe_load_backend::cpu, 0},
        {llama_moe_load_backend::cpu, 1},
        {llama_moe_load_backend::gpu, 2},
        {llama_moe_load_backend::cpu, 2},
        {llama_moe_load_backend::cpu, 3},
        {llama_moe_load_backend::gpu, 3},
    };
    return layer;
}

static llama_moe_packed_tensor_layout make_source() {
    llama_moe_packed_tensor_layout source;
    source.layer = 3;
    source.name = "blk.3.ffn_gate_exps.weight";
    source.ne = {2048, 512, 8, 1};
    source.nb = {1, 2048, 1024 * 1024, 8 * 1024 * 1024};
    source.size_bytes = 8 * 1024 * 1024;
    return source;
}

int main() {
    ggml_tensor tensor_meta = {};
    tensor_meta.type = GGML_TYPE_F32;
    tensor_meta.ne[0] = 8;
    tensor_meta.ne[1] = 4;
    tensor_meta.ne[2] = 8;
    tensor_meta.ne[3] = 1;
    tensor_meta.nb[0] = sizeof(float);
    tensor_meta.nb[1] = 8 * sizeof(float);
    tensor_meta.nb[2] = 8 * 4 * sizeof(float);
    tensor_meta.nb[3] = 8 * 4 * 8 * sizeof(float);
    ggml_set_name(&tensor_meta, "blk.3.ffn_gate_exps.weight");

    llama_moe_packed_tensor_layout metadata_layout;
    std::string error;
    require(llama_moe_packed_tensor_layout_from_tensor(3, &tensor_meta, metadata_layout, error),
        error.c_str());
    require(metadata_layout.layer == 3, "metadata layout layer mismatch");
    require(metadata_layout.name == tensor_meta.name, "metadata layout name mismatch");
    require(metadata_layout.ne[2] == 8, "metadata layout expert dimension mismatch");
    require(metadata_layout.nb[2] == tensor_meta.nb[2], "metadata layout expert stride mismatch");
    require(metadata_layout.size_bytes == ggml_nbytes(&tensor_meta), "metadata layout byte size mismatch");
    require(!llama_moe_packed_tensor_layout_from_tensor(-1, &tensor_meta, metadata_layout, error),
        "negative metadata layer must be rejected");
    require(!llama_moe_packed_tensor_layout_from_tensor(3, nullptr, metadata_layout, error),
        "null tensor metadata must be rejected");

    ggml_tensor zero_stride_meta = tensor_meta;
    zero_stride_meta.nb[2] = 0;
    require(!llama_moe_packed_tensor_layout_from_tensor(3, &zero_stride_meta, metadata_layout, error),
        "zero tensor metadata stride must be rejected");

    const auto placement = make_placement();
    const auto source = make_source();

    llama_moe_compact_tensor_pool_plan plan;
    require(llama_moe_compact_tensor_pool_plan_build(source, placement, plan, error), error.c_str());

    require(plan.cpu_expert_count == 4 && plan.gpu_expert_count == 4, "compact counts mismatch");
    require(plan.cpu_ne[2] == 4 && plan.gpu_ne[2] == 4, "compact expert dimensions mismatch");
    require(plan.cpu_bytes == 4 * 1024 * 1024, "CPU compact bytes mismatch");
    require(plan.gpu_bytes == 4 * 1024 * 1024, "GPU compact bytes mismatch");

    require(plan.gpu_spans.size() == 3, "GPU spans were not coalesced as expected");
    require(plan.gpu_spans[0].global_expert_first == 0, "first GPU span source mismatch");
    require(plan.gpu_spans[0].local_expert_first == 0, "first GPU span destination mismatch");
    require(plan.gpu_spans[0].expert_count == 2, "first GPU span count mismatch");
    require(plan.gpu_spans[1].global_expert_first == 4, "second GPU span source mismatch");
    require(plan.gpu_spans[2].global_expert_first == 7, "third GPU span source mismatch");

    require(plan.cpu_spans.size() == 2, "CPU spans were not coalesced as expected");
    require(plan.cpu_spans[0].global_expert_first == 2, "first CPU span source mismatch");
    require(plan.cpu_spans[0].expert_count == 2, "first CPU span count mismatch");
    require(plan.cpu_spans[1].global_expert_first == 5, "second CPU span source mismatch");
    require(plan.cpu_spans[1].expert_count == 2, "second CPU span count mismatch");
    require(plan.cpu_spans[1].destination_offset_bytes == 2 * 1024 * 1024,
        "second CPU span destination offset mismatch");

    llama_moe_packed_tensor_layout bad_size = source;
    --bad_size.size_bytes;
    require(!llama_moe_compact_tensor_pool_plan_build(bad_size, placement, plan, error),
        "mis-sized packed tensor must be rejected");

    llama_moe_packed_tensor_layout bad_axis = source;
    bad_axis.ne[2] = 7;
    require(!llama_moe_compact_tensor_pool_plan_build(bad_axis, placement, plan, error),
        "expert-axis mismatch must be rejected");

    auto duplicate = placement;
    duplicate.global_to_local[1].local_index = 0;
    require(!llama_moe_compact_tensor_pool_plan_build(source, duplicate, plan, error),
        "duplicate destination local index must be rejected");

    llama_moe_load_layer_placement all_gpu;
    all_gpu.layer = 3;
    all_gpu.expert_count = 8;
    all_gpu.cpu_expert_count = 0;
    all_gpu.gpu_expert_count = 8;
    for (uint32_t expert = 0; expert < 8; ++expert) {
        all_gpu.global_to_local.push_back({llama_moe_load_backend::gpu, expert});
    }
    require(llama_moe_compact_tensor_pool_plan_build(source, all_gpu, plan, error), error.c_str());
    require(plan.cpu_expert_count == 0 && plan.cpu_bytes == 0 && plan.cpu_ne[2] == 0,
        "empty CPU pool must remain unallocated");
    require(plan.cpu_spans.empty() && plan.gpu_spans.size() == 1,
        "all-GPU span coalescing mismatch");

    std::vector<llama_moe_pool_copy_span> valid_spans = {
        {0, 0, 1, 0, 0, 64},
        {2, 1, 1, 128, 64, 64},
    };
    require(llama_moe_pool_copy_spans_validate(256, 128, valid_spans, error), error.c_str());

    auto destination_gap = valid_spans;
    destination_gap[1].destination_offset_bytes = 96;
    require(!llama_moe_pool_copy_spans_validate(256, 160, destination_gap, error),
        "destination gap must be rejected");

    auto source_overlap = valid_spans;
    source_overlap[1].source_offset_bytes = 32;
    require(!llama_moe_pool_copy_spans_validate(256, 128, source_overlap, error),
        "source overlap must be rejected");

    auto source_oob = valid_spans;
    source_oob[1].source_offset_bytes = 224;
    require(!llama_moe_pool_copy_spans_validate(256, 128, source_oob, error),
        "source out-of-bounds span must be rejected");

    std::vector<llama_moe_pool_copy_span> no_spans;
    require(llama_moe_pool_copy_spans_validate(256, 0, no_spans, error),
        "zero-sized destination should accept no spans");

    return 0;
}
