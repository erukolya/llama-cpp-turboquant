#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <climits>
#include <cmath>
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
#include <unordered_map>
#include <utility>
#include <vector>

struct server_moe_stats_options {
    std::string stats_path;
    std::string placement_path;
    uint32_t sample_rate = 16;
};

inline bool server_moe_stats_parse_positive_uint(const std::string & value, uint32_t & result) {
    if (value.empty()) {
        return false;
    }

    char * end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed == 0 || parsed > 1000000UL) {
        return false;
    }

    result = static_cast<uint32_t>(parsed);
    return true;
}

inline bool server_moe_stats_parse_args(int & argc, char ** argv, server_moe_stats_options & options) {
    bool use_default_stats_path = false;
    bool use_default_placement_path = false;
    bool sample_rate_explicit = false;
    bool show_help = false;
    int write_index = 1;

    for (int read_index = 1; read_index < argc; ++read_index) {
        const std::string arg = argv[read_index];

        if (arg == "-h" || arg == "--help" || arg == "--usage") {
            show_help = true;
        }

        if (arg == "--moe-stats") {
            use_default_stats_path = true;
            continue;
        }

        if (arg == "--moe-stats-exact") {
            use_default_stats_path = true;
            options.sample_rate = 1;
            sample_rate_explicit = true;
            continue;
        }

        if (arg == "--moe-placement") {
            use_default_placement_path = true;
            continue;
        }

        if (arg == "--moe-stats-file" || arg == "--moe-placement-file" || arg == "--moe-stats-sample") {
            if (read_index + 1 >= argc || argv[read_index + 1][0] == '\0') {
                std::fprintf(stderr, "error: %s requires a value\n", arg.c_str());
                return false;
            }

            const std::string value = argv[++read_index];
            if (arg == "--moe-stats-file") {
                options.stats_path = value;
            } else if (arg == "--moe-placement-file") {
                options.placement_path = value;
            } else {
                if (!server_moe_stats_parse_positive_uint(value, options.sample_rate)) {
                    std::fprintf(stderr, "error: --moe-stats-sample must be an integer from 1 to 1000000\n");
                    return false;
                }
                use_default_stats_path = true;
                sample_rate_explicit = true;
            }
            continue;
        }

        static constexpr const char * stats_file_prefix = "--moe-stats-file=";
        static constexpr const char * placement_file_prefix = "--moe-placement-file=";
        static constexpr const char * sample_prefix = "--moe-stats-sample=";

        if (arg.compare(0, std::strlen(stats_file_prefix), stats_file_prefix) == 0) {
            options.stats_path = arg.substr(std::strlen(stats_file_prefix));
            if (options.stats_path.empty()) {
                std::fprintf(stderr, "error: --moe-stats-file requires a non-empty file path\n");
                return false;
            }
            continue;
        }

        if (arg.compare(0, std::strlen(placement_file_prefix), placement_file_prefix) == 0) {
            options.placement_path = arg.substr(std::strlen(placement_file_prefix));
            if (options.placement_path.empty()) {
                std::fprintf(stderr, "error: --moe-placement-file requires a non-empty file path\n");
                return false;
            }
            continue;
        }

        if (arg.compare(0, std::strlen(sample_prefix), sample_prefix) == 0) {
            const std::string value = arg.substr(std::strlen(sample_prefix));
            if (!server_moe_stats_parse_positive_uint(value, options.sample_rate)) {
                std::fprintf(stderr, "error: --moe-stats-sample must be an integer from 1 to 1000000\n");
                return false;
            }
            use_default_stats_path = true;
            sample_rate_explicit = true;
            continue;
        }

        argv[write_index++] = argv[read_index];
    }

    argc = write_index;
    argv[argc] = nullptr;

    if (options.stats_path.empty()) {
        if (const char * env_path = std::getenv("LLAMA_ARG_MOE_STATS_FILE")) {
            options.stats_path = env_path;
        }
    }

    if (options.placement_path.empty()) {
        if (const char * env_path = std::getenv("LLAMA_ARG_MOE_PLACEMENT_FILE")) {
            options.placement_path = env_path;
        }
    }

    if (!sample_rate_explicit) {
        if (const char * env_sample = std::getenv("LLAMA_ARG_MOE_STATS_SAMPLE")) {
            if (!server_moe_stats_parse_positive_uint(env_sample, options.sample_rate)) {
                std::fprintf(stderr, "error: LLAMA_ARG_MOE_STATS_SAMPLE must be an integer from 1 to 1000000\n");
                return false;
            }
        }
    }

    if (options.stats_path.empty() && use_default_stats_path) {
        options.stats_path = "moe-stats.csv";
    }

    if (options.placement_path.empty() && use_default_placement_path) {
        options.placement_path = "moe-placement.csv";
    }

    if (show_help) {
        std::fprintf(stdout,
            "\nMoE profiling:\n"
            "  --moe-stats                     sample routed expert selections and write moe-stats.csv\n"
            "  --moe-stats-exact               collect every routing decision (slowest, exact counts)\n"
            "  --moe-stats-file FNAME          write expert statistics to FNAME\n"
            "                                    (env: LLAMA_ARG_MOE_STATS_FILE)\n"
            "  --moe-stats-sample N            observe each layer once per N evaluations (default: 16)\n"
            "                                    (env: LLAMA_ARG_MOE_STATS_SAMPLE)\n"
            "  --moe-placement                 write routed MoE tensor placement to moe-placement.csv\n"
            "  --moe-placement-file FNAME      write routed MoE tensor placement to FNAME\n"
            "                                    (env: LLAMA_ARG_MOE_PLACEMENT_FILE)\n\n");
    }

    return true;
}

class server_moe_stats_collector {
public:
    explicit server_moe_stats_collector(server_moe_stats_options options) :
        stats_path_(std::move(options.stats_path)),
        placement_path_(std::move(options.placement_path)),
        sample_rate_(options.sample_rate) {
    }

    ~server_moe_stats_collector() {
        dump();
    }

    server_moe_stats_collector(const server_moe_stats_collector &) = delete;
    server_moe_stats_collector & operator=(const server_moe_stats_collector &) = delete;

    bool enabled() const {
        return stats_enabled() || placement_enabled();
    }

    bool stats_enabled() const {
        return !stats_path_.empty();
    }

    bool placement_enabled() const {
        return !placement_path_.empty();
    }

    const std::string & stats_path() const {
        return stats_path_;
    }

    const std::string & placement_path() const {
        return placement_path_;
    }

    uint32_t sample_rate() const {
        return sample_rate_;
    }

    static bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
        auto * collector = static_cast<server_moe_stats_collector *>(user_data);
        if (collector == nullptr || !collector->enabled() || tensor == nullptr) {
            return false;
        }

        if (ask && collector->placement_enabled() && tensor->op == GGML_OP_MUL_MAT_ID) {
            collector->record_placement(tensor);
        }

        if (!collector->stats_enabled() || !is_topk_tensor(tensor)) {
            return false;
        }

        if (ask) {
            return collector->schedule_sample(tensor);
        }

        const uint32_t sample_weight = collector->take_pending_sample(tensor);
        if (sample_weight == 0) {
            return true;
        }

        collector->collect(tensor);
        return true;
    }

    void dump() noexcept {
        dump_stats();
        dump_placement();
    }

private:
    struct expert_hits {
        uint64_t sampled = 0;
    };

    struct layer_sampling {
        uint64_t evaluations = 0;
        uint64_t sampled_evaluations = 0;
        uint64_t router_selections = 0;
        uint64_t sampled_router_selections = 0;
    };

    struct placement_entry {
        int layer = -1;
        int64_t n_experts = 0;
        uint64_t size_bytes = 0;
        std::string tensor;
        std::string storage;
        std::string buffer;
        std::string device;
    };

    static bool is_topk_tensor(const ggml_tensor * tensor) {
        static constexpr const char * prefix = "ffn_moe_topk";
        return tensor->type == GGML_TYPE_I32 &&
            std::strncmp(tensor->name, prefix, std::strlen(prefix)) == 0;
    }

    static int parse_positive_decimal(const char * begin) {
        if (begin == nullptr || *begin < '0' || *begin > '9') {
            return -1;
        }

        long value = 0;
        const char * cur = begin;
        while (*cur >= '0' && *cur <= '9') {
            value = value * 10 + (*cur - '0');
            if (value > INT_MAX) {
                return -1;
            }
            ++cur;
        }

        return static_cast<int>(value);
    }

    static int parse_layer(const char * tensor_name) {
        if (tensor_name == nullptr) {
            return -1;
        }

        if (const char * block = std::strstr(tensor_name, "blk.")) {
            return parse_positive_decimal(block + 4);
        }

        if (const char * separator = std::strrchr(tensor_name, '-')) {
            return parse_positive_decimal(separator + 1);
        }

        return -1;
    }

    static std::string csv_escape(const std::string & value) {
        if (value.find_first_of(",\"\r\n") == std::string::npos) {
            return value;
        }

        std::string escaped = "\"";
        for (const char ch : value) {
            if (ch == '\"') {
                escaped += "\"\"";
            } else {
                escaped += ch;
            }
        }
        escaped += '\"';
        return escaped;
    }

    static std::string storage_name(ggml_backend_buffer_t buffer) {
        if (buffer == nullptr) {
            return "UNKNOWN";
        }

        if (ggml_backend_buffer_is_host(buffer)) {
            return "RAM";
        }

        const auto buft = ggml_backend_buffer_get_type(buffer);
        const auto device = buft == nullptr ? nullptr : ggml_backend_buft_get_device(buft);
        if (device == nullptr) {
            return "DEVICE";
        }

        switch (ggml_backend_dev_type(device)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   return "RAM";
            case GGML_BACKEND_DEVICE_TYPE_GPU:   return "VRAM";
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "SHARED";
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
            case GGML_BACKEND_DEVICE_TYPE_META:  return "SPLIT";
        }

        return "DEVICE";
    }

    bool schedule_sample(const ggml_tensor * tensor) {
        const int layer = parse_layer(tensor->name);
        const uint64_t selections = static_cast<uint64_t>(ggml_nelements(tensor));

        std::lock_guard<std::mutex> lock(mutex_);
        auto & sampling = sampling_[layer];
        const uint64_t call_index = sampling.evaluations++;
        sampling.router_selections += selections;

        bool sample = sample_rate_ == 1 || call_index == 0;
        if (!sample) {
            const uint64_t phase = layer >= 0 ? static_cast<uint64_t>(layer) % sample_rate_ : 0;
            sample = ((call_index + phase) % sample_rate_) == 0;
        }

        if (sample) {
            ++sampling.sampled_evaluations;
            sampling.sampled_router_selections += selections;
            pending_samples_[tensor] = 1;
        }

        return sample;
    }

    uint32_t take_pending_sample(const ggml_tensor * tensor) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = pending_samples_.find(tensor);
        if (it == pending_samples_.end()) {
            return 0;
        }

        const uint32_t weight = it->second;
        pending_samples_.erase(it);
        return weight;
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
                layer_hits.resize(index + 1);
            }
            ++layer_hits[index].sampled;
            ++total_sampled_hits_;
        }
    }

    void record_placement(const ggml_tensor * op) {
        const ggml_tensor * weight = op->src[0];
        if (weight == nullptr || weight->buffer == nullptr || weight->name[0] == '\0') {
            return;
        }

        const std::string tensor_name = weight->name;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (placements_.find(tensor_name) != placements_.end()) {
                return;
            }
        }

        placement_entry entry;
        entry.layer = parse_layer(weight->name);
        entry.n_experts = weight->ne[2];
        entry.size_bytes = static_cast<uint64_t>(ggml_nbytes(weight));
        entry.tensor = tensor_name;
        entry.storage = storage_name(weight->buffer);
        entry.buffer = ggml_backend_buffer_name(weight->buffer);

        const auto buft = ggml_backend_buffer_get_type(weight->buffer);
        const auto device = buft == nullptr ? nullptr : ggml_backend_buft_get_device(buft);
        entry.device = device == nullptr ? "unknown" : ggml_backend_dev_name(device);

        std::lock_guard<std::mutex> lock(mutex_);
        placements_.emplace(tensor_name, std::move(entry));
    }

    void dump_stats() noexcept {
        if (!stats_enabled()) {
            return;
        }

        std::map<int, std::vector<expert_hits>> hits_snapshot;
        std::map<int, layer_sampling> sampling_snapshot;
        uint64_t total_sampled_hits = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            hits_snapshot = hits_;
            sampling_snapshot = sampling_;
            total_sampled_hits = total_sampled_hits_;
        }

        try {
            std::ofstream output(stats_path_, std::ios::out | std::ios::trunc);
            if (!output) {
                std::fprintf(stderr, "moe_stats: failed to open '%s' for writing\n", stats_path_.c_str());
                return;
            }

            output << "layer,expert,hits,sampled_hits,layer_share,coverage,sample_rate\n";
            output << std::fixed << std::setprecision(9);

            uint64_t total_estimated_hits = 0;
            for (const auto & [layer, experts] : hits_snapshot) {
                const auto sampling_it = sampling_snapshot.find(layer);
                if (sampling_it == sampling_snapshot.end()) {
                    continue;
                }

                const auto & sampling = sampling_it->second;
                if (sampling.sampled_router_selections == 0) {
                    continue;
                }

                uint64_t layer_sampled_hits = 0;
                for (const auto & expert : experts) {
                    layer_sampled_hits += expert.sampled;
                }

                const double coverage = sampling.router_selections == 0 ? 0.0 :
                    static_cast<double>(sampling.sampled_router_selections) /
                    static_cast<double>(sampling.router_selections);

                for (size_t expert = 0; expert < experts.size(); ++expert) {
                    const uint64_t sampled_hits = experts[expert].sampled;
                    if (sampled_hits == 0) {
                        continue;
                    }

                    const double share = layer_sampled_hits == 0 ? 0.0 :
                        static_cast<double>(sampled_hits) / static_cast<double>(layer_sampled_hits);
                    const uint64_t estimated_hits = static_cast<uint64_t>(std::llround(
                        share * static_cast<double>(sampling.router_selections)));
                    total_estimated_hits += estimated_hits;

                    output << layer << ',' << expert << ',' << estimated_hits << ',' << sampled_hits << ','
                           << share << ',' << coverage << ',' << sample_rate_ << '\n';
                }
            }

            output.flush();
            if (!output) {
                std::fprintf(stderr, "moe_stats: failed while writing '%s'\n", stats_path_.c_str());
                return;
            }

            std::fprintf(stderr,
                "moe_stats: wrote %llu estimated expert selections (%llu sampled) to '%s'\n",
                static_cast<unsigned long long>(total_estimated_hits),
                static_cast<unsigned long long>(total_sampled_hits),
                stats_path_.c_str());
        } catch (const std::exception & error) {
            std::fprintf(stderr, "moe_stats: failed to write '%s': %s\n", stats_path_.c_str(), error.what());
        } catch (...) {
            std::fprintf(stderr, "moe_stats: failed to write '%s': unknown error\n", stats_path_.c_str());
        }
    }

    void dump_placement() noexcept {
        if (!placement_enabled()) {
            return;
        }

        std::map<std::string, placement_entry> snapshot;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot = placements_;
        }

        try {
            std::ofstream output(placement_path_, std::ios::out | std::ios::trunc);
            if (!output) {
                std::fprintf(stderr, "moe_placement: failed to open '%s' for writing\n", placement_path_.c_str());
                return;
            }

            output << "layer,tensor,expert_first,expert_last,storage,buffer,device,size_mib\n";
            output << std::fixed << std::setprecision(3);

            for (const auto & [name, entry] : snapshot) {
                const int64_t expert_last = entry.n_experts > 0 ? entry.n_experts - 1 : -1;
                output << entry.layer << ',' << csv_escape(entry.tensor) << ",0," << expert_last << ','
                       << entry.storage << ',' << csv_escape(entry.buffer) << ',' << csv_escape(entry.device) << ','
                       << static_cast<double>(entry.size_bytes) / (1024.0 * 1024.0) << '\n';
            }

            output.flush();
            if (!output) {
                std::fprintf(stderr, "moe_placement: failed while writing '%s'\n", placement_path_.c_str());
                return;
            }

            std::fprintf(stderr, "moe_placement: wrote %zu routed MoE tensors to '%s'\n",
                snapshot.size(), placement_path_.c_str());
        } catch (const std::exception & error) {
            std::fprintf(stderr, "moe_placement: failed to write '%s': %s\n", placement_path_.c_str(), error.what());
        } catch (...) {
            std::fprintf(stderr, "moe_placement: failed to write '%s': unknown error\n", placement_path_.c_str());
        }
    }

    std::string stats_path_;
    std::string placement_path_;
    uint32_t sample_rate_ = 16;

    std::mutex mutex_;
    std::map<int, std::vector<expert_hits>> hits_;
    std::map<int, layer_sampling> sampling_;
    std::unordered_map<const ggml_tensor *, uint32_t> pending_samples_;
    std::map<std::string, placement_entry> placements_;
    uint64_t total_sampled_hits_ = 0;
};
