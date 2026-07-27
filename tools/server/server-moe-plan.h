#pragma once

#include "../../src/llama-model.h"

#include "moe-expert-plan.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class server_moe_plan_validator {
public:
    server_moe_plan_validator(
            std::string plan_path,
            std::string model_path,
            bool strict,
            bool dry_run) :
        plan_path_(std::move(plan_path)),
        model_path_(std::move(model_path)),
        strict_(strict),
        dry_run_(dry_run) {
    }

    const std::string & plan_path() const {
        return plan_path_;
    }

    bool strict() const {
        return strict_;
    }

    bool dry_run() const {
        return dry_run_;
    }

    bool validate(const llama_model * model) const noexcept {
        try {
            if (model == nullptr) {
                return fail("loaded model is null");
            }

            common_moe_expert_plan plan;
            std::string load_error;
            if (!common_moe_expert_plan_load(plan_path_, plan, load_error)) {
                return fail(load_error);
            }

            const int32_t model_layer_count = static_cast<int32_t>(model->layers.size());
            int32_t experts_per_layer = -1;
            std::vector<std::string> layout_errors;
            std::vector<std::string> layout_warnings;
            uint64_t actual_selected_bytes = 0;
            uint32_t actual_selected_experts = 0;
            size_t matched_manifest_tensors = 0;

            if (!plan.model_fingerprint.empty()) {
                std::string actual_fingerprint;
                std::string fingerprint_error;
                if (!sampled_file_fingerprint(model_path_, actual_fingerprint, fingerprint_error)) {
                    layout_errors.push_back(fingerprint_error);
                } else if (actual_fingerprint != plan.model_fingerprint) {
                    layout_errors.push_back(
                        "model_fingerprint mismatch: plan=" + plan.model_fingerprint +
                        ", loaded=" + actual_fingerprint);
                }
            } else {
                layout_warnings.push_back("plan has no model_fingerprint");
            }

            if (strict_ && plan.tensor_manifest.empty()) {
                layout_errors.push_back(
                    "strict model validation requires a schema-v2 tensor_manifest; regenerate the plan with --model");
            } else if (plan.tensor_manifest.empty()) {
                layout_warnings.push_back(
                    "legacy plan has no tensor_manifest; only layer counts and expert strides can be checked");
            }

            for (int32_t layer_index = 0; layer_index < model_layer_count; ++layer_index) {
                const auto & model_layer = model->layers[static_cast<size_t>(layer_index)];
                const auto * plan_layer = find_plan_layer(plan, layer_index);
                if (plan_layer == nullptr) {
                    layout_errors.push_back("plan does not contain layer " + std::to_string(layer_index));
                    continue;
                }

                if (has_auxiliary_expert_tensors(model_layer)) {
                    layout_errors.push_back(
                        "layer " + std::to_string(layer_index) +
                        ": per-expert bias/scale tensors are present but are not yet supported by static placement");
                }

                std::vector<const ggml_tensor *> tensors;
                std::set<const ggml_tensor *> seen;
                add_tensor(tensors, seen, model_layer.ffn_gate_exps);
                add_tensor(tensors, seen, model_layer.ffn_up_exps);
                add_tensor(tensors, seen, model_layer.ffn_gate_up_exps);
                add_tensor(tensors, seen, model_layer.ffn_down_exps);

                const bool has_gate = model_layer.ffn_gate_exps != nullptr;
                const bool has_up = model_layer.ffn_up_exps != nullptr;
                const bool has_gate_up = model_layer.ffn_gate_up_exps != nullptr;
                const bool has_down = model_layer.ffn_down_exps != nullptr;

                if (!has_down || (!has_gate_up && !(has_gate && has_up))) {
                    layout_errors.push_back(
                        "layer " + std::to_string(layer_index) +
                        ": unsupported routed-expert layout; expected down + (gate_up or gate+up)");
                    continue;
                }

                int64_t layer_expert_count = -1;
                uint64_t logical_expert_bytes = 0;

                for (const ggml_tensor * tensor : tensors) {
                    validate_tensor(layer_index, tensor, layer_expert_count, logical_expert_bytes, layout_errors);

                    if (!plan.tensor_manifest.empty()) {
                        const auto * expected = find_manifest_tensor(plan, layer_index, tensor->name);
                        if (expected == nullptr) {
                            layout_errors.push_back(
                                "layer " + std::to_string(layer_index) +
                                ": loaded routed tensor '" + tensor->name + "' is absent from tensor_manifest");
                        } else {
                            ++matched_manifest_tensors;
                            compare_manifest_tensor(*expected, tensor, layout_errors);
                        }
                    }

                    if (dry_run_) {
                        std::fprintf(stderr,
                            "moe_plan:   tensor '%s' type=%s ne=[%lld,%lld,%lld,%lld] "
                            "nb=[%zu,%zu,%zu,%zu] bytes=%zu\n",
                            tensor->name,
                            ggml_type_name(tensor->type),
                            static_cast<long long>(tensor->ne[0]),
                            static_cast<long long>(tensor->ne[1]),
                            static_cast<long long>(tensor->ne[2]),
                            static_cast<long long>(tensor->ne[3]),
                            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3],
                            ggml_nbytes(tensor));
                    }
                }

                if (layer_expert_count <= 0 || logical_expert_bytes == 0) {
                    continue;
                }

                if (experts_per_layer < 0) {
                    experts_per_layer = static_cast<int32_t>(layer_expert_count);
                } else if (experts_per_layer != layer_expert_count) {
                    layout_errors.push_back("model uses different routed expert counts across layers");
                }

                if (static_cast<uint32_t>(layer_expert_count) != plan_layer->expert_count) {
                    layout_errors.push_back(
                        "layer " + std::to_string(layer_index) + ": model expert count " +
                        std::to_string(layer_expert_count) + " differs from plan " +
                        std::to_string(plan_layer->expert_count));
                }

                const uint64_t intended_gpu_bytes =
                    logical_expert_bytes * static_cast<uint64_t>(plan_layer->gpu_experts.size());
                if (intended_gpu_bytes != plan_layer->gpu_bytes) {
                    layout_errors.push_back(
                        "layer " + std::to_string(layer_index) + ": plan gpu_bytes " +
                        std::to_string(plan_layer->gpu_bytes) + " differs from loaded tensor layout " +
                        std::to_string(intended_gpu_bytes));
                }

                for (const uint32_t expert : plan_layer->gpu_experts) {
                    if (expert >= static_cast<uint32_t>(layer_expert_count)) {
                        layout_errors.push_back(
                            "layer " + std::to_string(layer_index) + ": GPU expert " +
                            std::to_string(expert) + " is outside the loaded model range");
                    }
                }

                actual_selected_bytes += intended_gpu_bytes;
                actual_selected_experts += static_cast<uint32_t>(plan_layer->gpu_experts.size());

                if (dry_run_) {
                    std::fprintf(stderr,
                        "moe_plan: layer %d: tensors=%zu layout=%s experts=%lld gpu=%zu cpu=%lld "
                        "expert_bytes=%llu gpu_bytes=%llu\n",
                        layer_index,
                        tensors.size(),
                        has_gate_up ? "gate_up+down" : "gate+up+down",
                        static_cast<long long>(layer_expert_count),
                        plan_layer->gpu_experts.size(),
                        static_cast<long long>(
                            layer_expert_count - static_cast<int64_t>(plan_layer->gpu_experts.size())),
                        static_cast<unsigned long long>(logical_expert_bytes),
                        static_cast<unsigned long long>(intended_gpu_bytes));
                }
            }

            if (!plan.tensor_manifest.empty() && matched_manifest_tensors != plan.tensor_manifest.size()) {
                layout_errors.push_back(
                    "matched " + std::to_string(matched_manifest_tensors) + " of " +
                    std::to_string(plan.tensor_manifest.size()) + " tensor_manifest entries");
            }

            common_moe_expert_plan_expectation expectation;
            expectation.layer_count = model_layer_count;
            expectation.experts_per_layer = experts_per_layer;
            expectation.require_tensor_manifest = strict_;
            const common_moe_expert_plan_validation structural =
                common_moe_expert_plan_validate(plan, expectation);

            std::vector<std::string> errors = structural.errors;
            std::vector<std::string> warnings = structural.warnings;
            errors.insert(errors.end(), layout_errors.begin(), layout_errors.end());
            warnings.insert(warnings.end(), layout_warnings.begin(), layout_warnings.end());

            if (actual_selected_bytes != plan.selected_bytes) {
                errors.push_back(
                    "loaded tensor layout selected bytes " + std::to_string(actual_selected_bytes) +
                    " differ from plan selected_bytes " + std::to_string(plan.selected_bytes));
            }
            if (actual_selected_experts != plan.selected_expert_count) {
                errors.push_back(
                    "loaded plan selects " + std::to_string(actual_selected_experts) +
                    " experts but plan selected_expert_count is " +
                    std::to_string(plan.selected_expert_count));
            }

            for (const auto & warning : warnings) {
                std::fprintf(stderr, "moe_plan: warning: %s\n", warning.c_str());
            }
            for (const auto & error : errors) {
                std::fprintf(stderr, "moe_plan: error: %s\n", error.c_str());
            }

            if (!errors.empty()) {
                return fail("plan is incompatible with the loaded model");
            }

            std::fprintf(stderr,
                "moe_plan: validated '%s': schema=%u layers=%d experts/layer=%d selected=%u "
                "bytes=%llu hit_rate=%.3f%%; dry run only, allocation and inference are unchanged\n",
                plan_path_.c_str(),
                plan.schema_version,
                model_layer_count,
                experts_per_layer,
                plan.selected_expert_count,
                static_cast<unsigned long long>(plan.selected_bytes),
                plan.estimated_gpu_hit_rate * 100.0);
            return true;
        } catch (const std::exception & error) {
            return fail(error.what());
        } catch (...) {
            return fail("unknown validation error");
        }
    }

private:
    static constexpr uint64_t fnv_offset = UINT64_C(14695981039346656037);
    static constexpr uint64_t fnv_prime = UINT64_C(1099511628211);
    static constexpr uint64_t fingerprint_sample_bytes = UINT64_C(4) * 1024 * 1024;

    static void fnv_update(uint64_t & value, const uint8_t * data, size_t size) {
        for (size_t index = 0; index < size; ++index) {
            value ^= data[index];
            value *= fnv_prime;
        }
    }

    static void fnv_update_u64_le(uint64_t & value, uint64_t input) {
        uint8_t bytes[8];
        for (size_t index = 0; index < 8; ++index) {
            bytes[index] = static_cast<uint8_t>((input >> (8 * index)) & 0xff);
        }
        fnv_update(value, bytes, sizeof(bytes));
    }

    static bool sampled_file_fingerprint(
            const std::string & path,
            std::string & result,
            std::string & error) {
        if (path.empty()) {
            error = "model path is empty; cannot verify model_fingerprint";
            return false;
        }

        std::ifstream input(path, std::ios::binary);
        if (!input) {
            error = "failed to open model for fingerprint: " + path;
            return false;
        }
        input.seekg(0, std::ios::end);
        const std::streamoff end = input.tellg();
        if (end < 0) {
            error = "failed to determine model size for fingerprint";
            return false;
        }
        const uint64_t size = static_cast<uint64_t>(end);
        uint64_t value = fnv_offset;
        fnv_update_u64_le(value, size);

        std::vector<uint8_t> buffer(static_cast<size_t>(std::min(size, fingerprint_sample_bytes)));
        input.seekg(0, std::ios::beg);
        if (!buffer.empty()) {
            input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            if (input.gcount() != static_cast<std::streamsize>(buffer.size())) {
                error = "failed to read model prefix for fingerprint";
                return false;
            }
            fnv_update(value, buffer.data(), buffer.size());
        }

        if (size > fingerprint_sample_bytes) {
            const uint64_t offset = std::max(fingerprint_sample_bytes, size - fingerprint_sample_bytes);
            fnv_update_u64_le(value, offset);
            const size_t tail_size = static_cast<size_t>(size - offset);
            buffer.resize(tail_size);
            input.clear();
            input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            if (input.gcount() != static_cast<std::streamsize>(buffer.size())) {
                error = "failed to read model suffix for fingerprint";
                return false;
            }
            fnv_update(value, buffer.data(), buffer.size());
        }

        std::ostringstream formatted;
        formatted << "sampled-fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << value;
        result = formatted.str();
        error.clear();
        return true;
    }

    static void add_tensor(
            std::vector<const ggml_tensor *> & tensors,
            std::set<const ggml_tensor *> & seen,
            const ggml_tensor * tensor) {
        if (tensor != nullptr && seen.insert(tensor).second) {
            tensors.push_back(tensor);
        }
    }

    static bool has_auxiliary_expert_tensors(const llama_layer & layer) {
        return layer.ffn_gate_exps_b != nullptr ||
            layer.ffn_down_exps_b != nullptr ||
            layer.ffn_up_exps_b != nullptr ||
            layer.ffn_gate_up_exps_b != nullptr ||
            layer.ffn_gate_exps_s != nullptr ||
            layer.ffn_down_exps_s != nullptr ||
            layer.ffn_up_exps_s != nullptr ||
            layer.ffn_gate_exps_in_s != nullptr ||
            layer.ffn_down_exps_in_s != nullptr ||
            layer.ffn_up_exps_in_s != nullptr;
    }

    static void validate_tensor(
            int32_t layer,
            const ggml_tensor * tensor,
            int64_t & layer_expert_count,
            uint64_t & logical_expert_bytes,
            std::vector<std::string> & errors) {
        if (tensor->ne[2] <= 0) {
            errors.push_back(
                "layer " + std::to_string(layer) + ": tensor '" + tensor->name +
                "' has no routed expert axis at ne[2]");
            return;
        }
        if (tensor->nb[2] == 0) {
            errors.push_back(
                "layer " + std::to_string(layer) + ": tensor '" + tensor->name +
                "' has zero expert stride nb[2]");
            return;
        }

        if (layer_expert_count < 0) {
            layer_expert_count = tensor->ne[2];
        } else if (layer_expert_count != tensor->ne[2]) {
            errors.push_back(
                "layer " + std::to_string(layer) + ": routed tensors disagree on expert count");
        }

        const uint64_t stride = static_cast<uint64_t>(tensor->nb[2]);
        const uint64_t tensor_bytes = static_cast<uint64_t>(ggml_nbytes(tensor));
        const uint64_t last_expert_offset = stride * static_cast<uint64_t>(tensor->ne[2] - 1);
        if (tensor_bytes <= last_expert_offset) {
            errors.push_back(
                "layer " + std::to_string(layer) + ": tensor '" + tensor->name +
                "' does not contain the start of its last expert slice");
        }

        logical_expert_bytes += stride;
    }

    static void compare_manifest_tensor(
            const common_moe_expert_plan_tensor & expected,
            const ggml_tensor * actual,
            std::vector<std::string> & errors) {
        const std::string label =
            "layer " + std::to_string(expected.layer) + ": tensor '" + expected.name + "'";
        if (expected.type != ggml_type_name(actual->type)) {
            errors.push_back(label + " type mismatch: plan=" + expected.type +
                ", loaded=" + ggml_type_name(actual->type));
        }
        if (expected.size_bytes != ggml_nbytes(actual)) {
            errors.push_back(label + " size_bytes mismatch");
        }
        if (expected.expert_stride_bytes != actual->nb[2]) {
            errors.push_back(label + " expert_stride_bytes mismatch");
        }
        for (size_t dimension = 0; dimension < 4; ++dimension) {
            if (expected.ne[dimension] != actual->ne[dimension]) {
                errors.push_back(label + " ne[" + std::to_string(dimension) + "] mismatch");
            }
            if (expected.nb[dimension] != actual->nb[dimension]) {
                errors.push_back(label + " nb[" + std::to_string(dimension) + "] mismatch");
            }
        }
        if (expected.expert_first != 0 ||
                expected.expert_last + 1 != static_cast<uint32_t>(actual->ne[2])) {
            errors.push_back(label + " expert range mismatch");
        }
    }

    static const common_moe_expert_plan_layer * find_plan_layer(
            const common_moe_expert_plan & plan,
            int32_t layer) {
        const auto it = std::find_if(
            plan.layers.begin(),
            plan.layers.end(),
            [layer](const common_moe_expert_plan_layer & item) {
                return item.layer == layer;
            });
        return it == plan.layers.end() ? nullptr : &*it;
    }

    static const common_moe_expert_plan_tensor * find_manifest_tensor(
            const common_moe_expert_plan & plan,
            int32_t layer,
            const std::string & name) {
        const auto it = std::find_if(
            plan.tensor_manifest.begin(),
            plan.tensor_manifest.end(),
            [layer, &name](const common_moe_expert_plan_tensor & item) {
                return item.layer == layer && item.name == name;
            });
        return it == plan.tensor_manifest.end() ? nullptr : &*it;
    }

    bool fail(const std::string & message) const {
        std::fprintf(stderr, "moe_plan: %s: %s\n",
            strict_ ? "fatal" : "disabled",
            message.c_str());
        return !strict_;
    }

    std::string plan_path_;
    std::string model_path_;
    bool strict_ = false;
    bool dry_run_ = false;
};
