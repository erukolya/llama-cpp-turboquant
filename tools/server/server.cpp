#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-tools.h"

#include "arg.h"
#include "common.h"
#include "server-moe-placement.h"
#include "server-moe-profiler.h"

#include <memory>

static std::unique_ptr<server_moe_stats_collector> g_server_moe_stats;
static std::unique_ptr<server_moe_placement_report> g_server_moe_placement;

struct server_context_profiled : server_context {
    bool load_model(common_params & params) {
        if (!server_context::load_model(params)) {
            return false;
        }

        if (g_server_moe_placement) {
            llama_context * ctx = get_llama_context();
            g_server_moe_placement->capture(ctx == nullptr ? nullptr : llama_get_model(ctx));
        }

        return true;
    }
};

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

    if (!options.stats_path.empty()) {
        if (params.cb_eval != nullptr) {
            std::fprintf(stderr, "error: MoE expert statistics cannot replace an existing eval callback\n");
            return false;
        }

        server_moe_stats_options stats_options = options;
        stats_options.placement_path.clear();
        g_server_moe_stats = std::make_unique<server_moe_stats_collector>(std::move(stats_options));

        params.cb_eval = server_moe_stats_collector::eval_callback;
        params.cb_eval_user_data = g_server_moe_stats.get();

        // Warmup evaluates every expert and would pollute the real routing distribution.
        params.warmup = false;
        std::fprintf(stderr,
            "moe_stats: enabled, output '%s', sample 1/%u per layer (warmup disabled)\n",
            g_server_moe_stats->stats_path().c_str(),
            g_server_moe_stats->sample_rate());
    }

    if (!options.placement_path.empty()) {
        g_server_moe_placement = std::make_unique<server_moe_placement_report>(options.placement_path);
        std::fprintf(stderr, "moe_placement: enabled, output '%s'\n",
            g_server_moe_placement->output_path().c_str());
    }

    return true;
}

#define common_params_parse common_params_parse_with_moe_stats
#define server_context server_context_profiled
#include "server-original.cpp"
#undef server_context
#undef common_params_parse
