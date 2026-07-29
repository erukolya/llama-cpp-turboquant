#pragma once

#include "../../src/llama-model.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <set>
#include <string>
#include <utility>
#include <vector>

class server_moe_placement_report {
public:
    explicit server_moe_placement_report(std::string output_path) : output_path_(std::move(output_path)) {
    }

    bool enabled() const {
        return !output_path_.empty();
    }

    const std::string & output_path() const {
        return output_path_;
    }

    void capture(const llama_model * model) noexcept {
        if (!enabled() || model == nullptr) {
            return;
        }

        try {
            entries_.clear();
            seen_.clear();

            for (size_t layer_index = 0; layer_index < model->layers.size(); ++layer_index) {
                const auto & layer = model->layers[layer_index];
                add_tensor(static_cast<int>(layer_index), layer.ffn_gate_exps);
                add_tensor(static_cast<int>(layer_index), layer.ffn_up_exps);
                add_tensor(static_cast<int>(layer_index), layer.ffn_gate_up_exps);
                add_tensor(static_cast<int>(layer_index), layer.ffn_down_exps);
            }

            dump();
        } catch (const std::exception & error) {
            std::fprintf(stderr, "moe_placement: capture failed: %s\n", error.what());
        } catch (...) {
            std::fprintf(stderr, "moe_placement: capture failed: unknown error\n");
        }
    }

private:
    struct placement_entry {
        int layer = -1;
        int64_t n_experts = 0;
        uint64_t size_bytes = 0;
        uint64_t expert_stride_bytes = 0;
        std::string tensor;
        std::string type;
        int64_t ne[4] = {0, 0, 0, 0};
        uint64_t nb[4] = {0, 0, 0, 0};
        std::string storage;
        std::string buffer;
        std::string device;
    };

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

    void add_tensor(int layer, const ggml_tensor * tensor) {
        if (tensor == nullptr || tensor->buffer == nullptr || tensor->name[0] == '\0') {
            return;
        }

        if (!seen_.insert(tensor).second) {
            return;
        }

        placement_entry entry;
        entry.layer = layer;
        entry.n_experts = tensor->ne[2];
        entry.size_bytes = static_cast<uint64_t>(ggml_nbytes(tensor));
        entry.expert_stride_bytes = tensor->ne[2] > 0 ? static_cast<uint64_t>(tensor->nb[2]) : 0;
        entry.tensor = tensor->name;
        entry.type = ggml_type_name(tensor->type);
        for (int dimension = 0; dimension < 4; ++dimension) {
            entry.ne[dimension] = tensor->ne[dimension];
            entry.nb[dimension] = static_cast<uint64_t>(tensor->nb[dimension]);
        }
        entry.storage = storage_name(tensor->buffer);

        const char * buffer_name = ggml_backend_buffer_name(tensor->buffer);
        entry.buffer = buffer_name == nullptr ? "unknown" : buffer_name;

        const auto buft = ggml_backend_buffer_get_type(tensor->buffer);
        const auto device = buft == nullptr ? nullptr : ggml_backend_buft_get_device(buft);
        const char * device_name = device == nullptr ? nullptr : ggml_backend_dev_name(device);
        entry.device = device_name == nullptr ? "unknown" : device_name;

        entries_.push_back(std::move(entry));
    }

    void dump() const {
        std::ofstream output(output_path_, std::ios::out | std::ios::trunc);
        if (!output) {
            std::fprintf(stderr, "moe_placement: failed to open '%s' for writing\n", output_path_.c_str());
            return;
        }

        output << "layer,tensor,expert_first,expert_last,storage,buffer,device,size_mib,size_bytes,n_experts,"
                  "expert_stride_bytes,type,ne0,ne1,ne2,ne3,nb0,nb1,nb2,nb3\n";
        output << std::fixed << std::setprecision(3);

        for (const auto & entry : entries_) {
            const int64_t expert_last = entry.n_experts > 0 ? entry.n_experts - 1 : -1;
            output << entry.layer << ',' << csv_escape(entry.tensor) << ",0," << expert_last << ','
                   << entry.storage << ',' << csv_escape(entry.buffer) << ',' << csv_escape(entry.device) << ','
                   << static_cast<double>(entry.size_bytes) / (1024.0 * 1024.0) << ','
                   << entry.size_bytes << ',' << entry.n_experts << ',' << entry.expert_stride_bytes << ','
                   << csv_escape(entry.type) << ','
                   << entry.ne[0] << ',' << entry.ne[1] << ',' << entry.ne[2] << ',' << entry.ne[3] << ','
                   << entry.nb[0] << ',' << entry.nb[1] << ',' << entry.nb[2] << ',' << entry.nb[3] << '\n';
        }

        output.flush();
        if (!output) {
            std::fprintf(stderr, "moe_placement: failed while writing '%s'\n", output_path_.c_str());
            return;
        }

        std::fprintf(stderr, "moe_placement: wrote %zu persistent routed MoE tensors to '%s'\n",
            entries_.size(), output_path_.c_str());
    }

    std::string output_path_;
    std::vector<placement_entry> entries_;
    std::set<const ggml_tensor *> seen_;
};
