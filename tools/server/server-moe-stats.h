#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct server_moe_stats_options {
    std::string output_path;
};

inline bool server_moe_stats_parse_args(int & argc, char ** argv, server_moe_stats_options & options) {
    bool use_default_path = false;
    bool show_help = false;
    int write_index = 1;

    for (int read_index = 1; read_index < argc; ++read_index) {
        const std::string arg = argv[read_index];

        if (arg == "-h" || arg == "--help" || arg == "--usage") {
            show_help = true;
        }

        if (arg == "--moe-stats") {
            use_default_path = true;
            continue;
        }

        if (arg == "--moe-stats-file") {
            if (read_index + 1 >= argc || argv[read_index + 1][0] == '\0') {
                std::fprintf(stderr, "error: --moe-stats-file requires a file path\n");
                return false;
            }

            options.output_path = argv[++read_index];
            continue;
        }

        static constexpr const char * prefix = "--moe-stats-file=";
        if (arg.compare(0, std::strlen(prefix), prefix) == 0) {
            options.output_path = arg.substr(std::strlen(prefix));
            if (options.output_path.empty()) {
                std::fprintf(stderr, "error: --moe-stats-file requires a non-empty file path\n");
                return false;
            }
            continue;
        }

        argv[write_index++] = argv[read_index];
    }

    argc = write_index;
    argv[argc] = nullptr;

    if (options.output_path.empty()) {
        if (const char * env_path = std::getenv("LLAMA_ARG_MOE_STATS_FILE")) {
            options.output_path = env_path;
        }
    }

    if (options.output_path.empty() && use_default_path) {
        options.output_path = "moe-stats.csv";
    }

    if (show_help) {
        std::fprintf(stdout,
            "\nMoE expert statistics:\n"
            "  --moe-stats                     count routed expert selections and write moe-stats.csv\n"
            "  --moe-stats-file FNAME          write routed expert selections to FNAME\n"
            "                                    (env: LLAMA_ARG_MOE_STATS_FILE)\n\n");
    }

    return true;
}

class server_moe_stats_collector {
public:
    explicit server_moe_stats_collector(std::string output_path) : output_path_(std::move(output_path)) {
    }

    ~server_moe_stats_collector() {
        dump();
    }

    server_moe_stats_collector(const server_moe_stats_collector &) = delete;
    server_moe_stats_collector & operator=(const server_moe_stats_collector &) = delete;

    bool enabled() const {
        return !output_path_.empty();
    }

    const std::string & output_path() const {
        return output_path_;
    }

    static bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
        auto * collector = static_cast<server_moe_stats_collector *>(user_data);
        if (collector == nullptr || !collector->enabled() || tensor == nullptr) {
            return false;
        }

        const bool interested = is_topk_tensor(tensor);
        if (ask) {
            return interested;
        }

        if (!interested) {
            return true;
        }

        collector->collect(tensor);
        return true;
    }

    void dump() noexcept {
        if (!enabled()) {
            return;
        }

        std::map<int, std::vector<uint64_t>> snapshot;
        uint64_t total_hits = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot = hits_;
            total_hits = total_hits_;
        }

        try {
            std::ofstream output(output_path_, std::ios::out | std::ios::trunc);
            if (!output) {
                std::fprintf(stderr, "moe_stats: failed to open '%s' for writing\n", output_path_.c_str());
                return;
            }

            output << "layer,expert,hits,layer_share\n";
            output << std::fixed << std::setprecision(9);

            for (const auto & [layer, experts] : snapshot) {
                uint64_t layer_total = 0;
                for (const uint64_t hits : experts) {
                    layer_total += hits;
                }

                for (size_t expert = 0; expert < experts.size(); ++expert) {
                    const uint64_t hits = experts[expert];
                    if (hits == 0) {
                        continue;
                    }

                    const double share = layer_total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(layer_total);
                    output << layer << ',' << expert << ',' << hits << ',' << share << '\n';
                }
            }

            output.flush();
            if (!output) {
                std::fprintf(stderr, "moe_stats: failed while writing '%s'\n", output_path_.c_str());
                return;
            }

            std::fprintf(stderr, "moe_stats: wrote %llu expert selections to '%s'\n",
                static_cast<unsigned long long>(total_hits), output_path_.c_str());
        } catch (const std::exception & error) {
            std::fprintf(stderr, "moe_stats: failed to write '%s': %s\n", output_path_.c_str(), error.what());
        } catch (...) {
            std::fprintf(stderr, "moe_stats: failed to write '%s': unknown error\n", output_path_.c_str());
        }
    }

private:
    static bool is_topk_tensor(const ggml_tensor * tensor) {
        static constexpr const char * prefix = "ffn_moe_topk";
        return tensor->type == GGML_TYPE_I32 &&
            std::strncmp(tensor->name, prefix, std::strlen(prefix)) == 0;
    }

    static int parse_layer(const char * tensor_name) {
        const char * separator = std::strrchr(tensor_name, '-');
        if (separator == nullptr || separator[1] == '\0') {
            return -1;
        }

        char * end = nullptr;
        const long layer = std::strtol(separator + 1, &end, 10);
        if (end == separator + 1 || *end != '\0' || layer < 0 || layer > INT_MAX) {
            return -1;
        }

        return static_cast<int>(layer);
    }

    void collect(const ggml_tensor * tensor) {
        const bool is_host = ggml_backend_buffer_is_host(tensor->buffer);
        std::vector<uint8_t> host_data;
        const uint8_t * data = nullptr;

        if (is_host) {
            data = static_cast<const uint8_t *>(tensor->data);
        } else {
            host_data.resize(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, host_data.data(), 0, host_data.size());
            data = host_data.data();
        }

        if (data == nullptr) {
            return;
        }

        std::vector<int32_t> selected;
        selected.reserve(static_cast<size_t>(ggml_nelements(tensor)));

        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                        const size_t offset =
                            static_cast<size_t>(i0) * tensor->nb[0] +
                            static_cast<size_t>(i1) * tensor->nb[1] +
                            static_cast<size_t>(i2) * tensor->nb[2] +
                            static_cast<size_t>(i3) * tensor->nb[3];

                        int32_t expert = -1;
                        std::memcpy(&expert, data + offset, sizeof(expert));
                        if (expert >= 0 && expert <= 65535) {
                            selected.push_back(expert);
                        }
                    }
                }
            }
        }

        if (selected.empty()) {
            return;
        }

        const int layer = parse_layer(tensor->name);
        std::lock_guard<std::mutex> lock(mutex_);
        auto & layer_hits = hits_[layer];

        for (const int32_t expert : selected) {
            const size_t index = static_cast<size_t>(expert);
            if (layer_hits.size() <= index) {
                layer_hits.resize(index + 1, 0);
            }
            ++layer_hits[index];
            ++total_hits_;
        }
    }

    std::string output_path_;
    std::mutex mutex_;
    std::map<int, std::vector<uint64_t>> hits_;
    uint64_t total_hits_ = 0;
};
