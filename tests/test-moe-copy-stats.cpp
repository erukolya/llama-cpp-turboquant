#include "llama.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-moe-copy-stats: %s\n", message);
        std::exit(1);
    }
}

int main() {
    const auto stats = llama_moe_copy_stats(nullptr);
    require(stats.weight_copy_bytes == 0, "null context returned non-zero weight_copy_bytes");
    require(stats.weight_payload_bytes == 0, "null context returned non-zero weight_payload_bytes");
    require(stats.expert_slices == 0, "null context returned non-zero expert_slices");
    require(stats.copy_calls == 0, "null context returned non-zero copy_calls");
    require(stats.weight_inputs == 0, "null context returned non-zero weight_inputs");
    llama_moe_copy_stats_reset(nullptr);
    return 0;
}
