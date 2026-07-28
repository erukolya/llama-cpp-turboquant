#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int32_t MOE_MISSING_ID_MAGIC = 0x4D4F4553;

void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-moe-route-remap: %s\n", message);
        std::exit(1);
    }
}

ggml_context * make_context() {
    ggml_init_params params = {};
    params.mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to create ggml context");
    return ctx;
}

ggml_backend_sched_t make_scheduler(ggml_backend_t cpu) {
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr, "failed to create backend scheduler");
    return sched;
}

void test_route_remap(ggml_backend_t cpu) {
    ggml_context * ctx = make_context();

    constexpr int64_t n_expert = 6;
    constexpr int64_t n_used   = 4;
    constexpr int64_t n_tokens = 2;

    ggml_tensor * route_map = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
    ggml_tensor * global_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(route_map);
    ggml_set_input(global_ids);

    ggml_tensor * flat_ids = ggml_reshape_1d(ctx, global_ids, ggml_nelements(global_ids));
    ggml_tensor * local_flat = ggml_get_rows(ctx, route_map, flat_ids);
    ggml_tensor * local_ids = ggml_reshape_2d(ctx, local_flat, n_used, n_tokens);
    ggml_set_output(local_ids);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, local_ids);

    ggml_backend_sched_t sched = make_scheduler(cpu);
    require(ggml_backend_sched_alloc_graph(sched, graph), "failed to allocate remap graph");

    const std::array<int32_t, n_expert> map = { 0, -1, 1, -1, 2, -1 };
    const std::array<int32_t, n_used * n_tokens> selected = {
        0, 1, 5, 2,
        4, 3, 0, 5,
    };
    const std::array<int32_t, n_used * n_tokens> expected = {
        0, -1, -1, 1,
        2, -1, 0, -1,
    };

    ggml_backend_tensor_set(route_map, map.data(), 0, sizeof(map));
    ggml_backend_tensor_set(global_ids, selected.data(), 0, sizeof(selected));

    require(
        ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
        "remap graph computation failed");
    ggml_backend_sched_synchronize(sched);

    std::array<int32_t, n_used * n_tokens> actual = {};
    ggml_backend_tensor_get(local_ids, actual.data(), 0, sizeof(actual));
    require(actual == expected, "backend remap differs from expected slot-preserving mapping");

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
}

void test_argsort_route_remap_contiguous(ggml_backend_t cpu) {
    ggml_context * ctx = make_context();

    constexpr int64_t n_expert = 6;
    constexpr int64_t n_used   = 4;
    constexpr int64_t n_tokens = 2;

    ggml_tensor * route_map = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
    ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tokens);
    ggml_set_input(route_map);
    ggml_set_input(scores);

    ggml_tensor * selected = ggml_argsort_top_k(ctx, scores, n_used);
    ggml_tensor * selected_cont = ggml_cont(ctx, selected);
    ggml_tensor * flat_ids = ggml_reshape_1d(ctx, selected_cont, ggml_nelements(selected_cont));
    ggml_tensor * local_flat = ggml_get_rows(ctx, route_map, flat_ids);
    ggml_tensor * local_ids = ggml_reshape_2d(ctx, local_flat, n_used, n_tokens);
    ggml_set_output(local_ids);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, local_ids);

    ggml_backend_sched_t sched = make_scheduler(cpu);
    require(ggml_backend_sched_alloc_graph(sched, graph),
        "failed to allocate argsort remap graph");

    const std::array<int32_t, n_expert> map = { -1, -1, -1, -1, -1, -1 };
    const std::array<float, n_expert * n_tokens> score_data = {
        6.0f, 5.0f, 3.0f, 2.0f, 1.0f, 4.0f,
        4.0f, 2.0f, 1.0f, 5.0f, 6.0f, 3.0f,
    };

    ggml_backend_tensor_set(route_map, map.data(), 0, sizeof(map));
    ggml_backend_tensor_set(scores, score_data.data(), 0, sizeof(score_data));
    require(
        ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
        "argsort remap graph computation failed");
    ggml_backend_sched_synchronize(sched);

    std::array<int32_t, n_used * n_tokens> actual = {};
    ggml_backend_tensor_get(local_ids, actual.data(), 0, sizeof(actual));
    for (int32_t value : actual) {
        require(value == -1, "argsort route remap changed a slot or map value");
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
}

void test_mul_mat_id_missing_slots(ggml_backend_t cpu) {
    ggml_context * ctx = make_context();

    constexpr int64_t n_embd   = 2;
    constexpr int64_t n_ff     = 1;
    constexpr int64_t n_expert = 2;
    constexpr int64_t n_used   = 2;
    constexpr int64_t n_tokens = 2;

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * input   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(weights);
    ggml_set_input(input);
    ggml_set_input(ids);

    ggml_tensor * output = ggml_mul_mat_id(ctx, weights, input, ids);
    std::memcpy(output->op_params, &MOE_MISSING_ID_MAGIC, sizeof(MOE_MISSING_ID_MAGIC));
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_sched_t sched = make_scheduler(cpu);
    require(ggml_backend_sched_alloc_graph(sched, graph), "failed to allocate MUL_MAT_ID graph");

    const std::array<float, n_embd * n_ff * n_expert> weight_data = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const std::array<float, n_embd * n_tokens> input_data = {
        5.0f, 6.0f,
        7.0f, 8.0f,
    };
    const std::array<int32_t, n_used * n_tokens> id_data = {
        -1, 0,
        1, -1,
    };
    const std::array<float, n_ff * n_used * n_tokens> expected = {
        0.0f, 17.0f,
        53.0f, 0.0f,
    };

    ggml_backend_tensor_set(weights, weight_data.data(), 0, sizeof(weight_data));
    ggml_backend_tensor_set(input, input_data.data(), 0, sizeof(input_data));
    ggml_backend_tensor_set(ids, id_data.data(), 0, sizeof(id_data));

    require(
        ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
        "MUL_MAT_ID graph computation failed");
    ggml_backend_sched_synchronize(sched);

    std::array<float, n_ff * n_used * n_tokens> actual = {};
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    for (size_t index = 0; index < actual.size(); ++index) {
        require(std::fabs(actual[index] - expected[index]) < 1e-6f,
            "guarded MUL_MAT_ID did not zero missing slots or preserve valid results");
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(cpu != nullptr, "failed to initialize CPU backend");

    test_route_remap(cpu);
        test_argsort_route_remap_contiguous(cpu);
test_mul_mat_id_missing_slots(cpu);

    ggml_backend_free(cpu);
    return 0;
}
