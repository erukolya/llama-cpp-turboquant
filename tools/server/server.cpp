#include "arg.h"
#include "common.h"
#include "server-moe-profiler.h"

#include <memory>

static std::unique_ptr<server_moe_stats_collector> g_server_moe_stats;

static bool common_params_parse_with_moe_stats(
        int argc,
        char ** argv,
        common_params & params,
        llama_example ex,
        void (*print_usage)(int, char **) = nullptr) {
    server_moe_stats_options options;
    if (!server_moe_stats_parse_args(argc, argv, options)) {
        return false;
    }

    if (!::common_params_parse(argc, argv, params, ex, print_usage)) {
        return false;
    }

    if (options.stats_path.empty() && options.placement_path.empty()) {
        return true;
    }

    if (params.model.path.empty()) {
        std::fprintf(stderr, "error: MoE profiling is not supported in router mode; start a single-model server\n");
        return false;
    }

    if (params.cb_eval != nullptr) {
        std::fprintf(stderr, "error: MoE profiling cannot replace an existing eval callback\n");
        return false;
    }

    g_server_moe_stats = std::make_unique<server_moe_stats_collector>(std::move(options));
    params.cb_eval = server_moe_stats_collector::eval_callback;
    params.cb_eval_user_data = g_server_moe_stats.get();

    if (g_server_moe_stats->stats_enabled()) {
        // Warmup evaluates every expert and would pollute the real routing distribution.
        params.warmup = false;
        std::fprintf(stderr,
            "moe_stats: enabled, output '%s', sample 1/%u per layer (warmup disabled)\n",
            g_server_moe_stats->stats_path().c_str(),
            g_server_moe_stats->sample_rate());
    }

    if (g_server_moe_stats->placement_enabled()) {
        std::fprintf(stderr, "moe_placement: enabled, output '%s'\n",
            g_server_moe_stats->placement_path().c_str());
    }

    return true;
}

#define common_params_parse common_params_parse_with_moe_stats
#include "server-original.cpp"
#undef common_params_parse
