#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct server_moe_plan_options {
    std::string plan_path;
    bool strict = false;
    bool dry_run = false;
};

inline bool server_moe_plan_parse_args(int & argc, char ** argv, server_moe_plan_options & options) {
    bool show_help = false;
    int write_index = 1;

    for (int read_index = 1; read_index < argc; ++read_index) {
        const std::string arg = argv[read_index];

        if (arg == "-h" || arg == "--help" || arg == "--usage") {
            show_help = true;
        }

        if (arg == "--moe-expert-plan-strict") {
            options.strict = true;
            continue;
        }

        if (arg == "--moe-expert-plan-dry-run") {
            options.dry_run = true;
            continue;
        }

        if (arg == "--moe-expert-plan") {
            if (read_index + 1 >= argc || argv[read_index + 1][0] == '\0') {
                std::fprintf(stderr, "error: --moe-expert-plan requires a value\n");
                return false;
            }
            options.plan_path = argv[++read_index];
            continue;
        }

        static constexpr const char * plan_prefix = "--moe-expert-plan=";
        if (arg.compare(0, std::strlen(plan_prefix), plan_prefix) == 0) {
            options.plan_path = arg.substr(std::strlen(plan_prefix));
            if (options.plan_path.empty()) {
                std::fprintf(stderr, "error: --moe-expert-plan requires a non-empty file path\n");
                return false;
            }
            continue;
        }

        argv[write_index++] = argv[read_index];
    }

    argc = write_index;
    argv[argc] = nullptr;

    if (options.plan_path.empty()) {
        if (const char * env_path = std::getenv("LLAMA_ARG_MOE_EXPERT_PLAN")) {
            options.plan_path = env_path;
        }
    }

    if ((options.strict || options.dry_run) && options.plan_path.empty()) {
        std::fprintf(stderr,
            "error: --moe-expert-plan-strict and --moe-expert-plan-dry-run require --moe-expert-plan FNAME\n");
        return false;
    }

    if (show_help) {
        std::fprintf(stdout,
            "\nStatic MoE expert placement plan:\n"
            "  --moe-expert-plan FNAME         load and validate a static expert placement JSON plan\n"
            "                                    (env: LLAMA_ARG_MOE_EXPERT_PLAN)\n"
            "  --moe-expert-plan-strict        fail model startup when the plan is invalid or incompatible\n"
            "  --moe-expert-plan-dry-run       print the intended per-layer CPU/VRAM split; no allocation changes\n\n");
    }

    return true;
}
