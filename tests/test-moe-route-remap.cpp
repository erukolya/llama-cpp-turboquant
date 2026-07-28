#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-moe-route-remap: %s\n", message);
        std::exit(1);
    }
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(cpu != nullptr, "failed to initialize CPU backend");

    ggml_init_params params = {};
    params.mem_size = 32 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to create ggml context");

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

    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr, "failed to create backend scheduler");
    require(ggml_backend_sched_alloc_graph(sched, graph), "failed to allocate graph");

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
        "graph computation failed");
    ggml_backend_sched_synchronize(sched);

    std::array<int32_t, n_used * n_tokens> actual = {};
    ggml_backend_tensor_get(local_ids, actual.data(), 0, sizeof(actual));
    require(actual == expected, "backend remap differs from expected slot-preserving mapping");

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return 0;
}
