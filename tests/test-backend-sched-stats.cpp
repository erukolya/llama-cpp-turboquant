#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-backend-sched-stats: %s\n", message);
        std::exit(1);
    }
}

static void require_zero(const ggml_backend_sched_moe_copy_stats & stats) {
    require(stats.weight_copy_bytes == 0, "weight_copy_bytes is not zero");
    require(stats.weight_payload_bytes == 0, "weight_payload_bytes is not zero");
    require(stats.expert_slices == 0, "expert_slices is not zero");
    require(stats.copy_calls == 0, "copy_calls is not zero");
    require(stats.weight_inputs == 0, "weight_inputs is not zero");
}

static void require_zero(const ggml_backend_sched_moe_exec_stats & stats) {
    require(stats.cpu_ops == 0, "cpu_ops is not zero");
    require(stats.accelerator_ops == 0, "accelerator_ops is not zero");
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(cpu != nullptr, "failed to initialize CPU backend");

    ggml_backend_t backends[] = {cpu};
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    require(sched != nullptr, "failed to initialize scheduler");

    require_zero(ggml_backend_sched_get_moe_copy_stats(sched));
    ggml_backend_sched_reset_moe_copy_stats(sched);
    require_zero(ggml_backend_sched_get_moe_copy_stats(sched));

    require_zero(ggml_backend_sched_get_moe_exec_stats(sched));
    ggml_backend_sched_reset_moe_exec_stats(sched);
    require_zero(ggml_backend_sched_get_moe_exec_stats(sched));

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    return 0;
}
