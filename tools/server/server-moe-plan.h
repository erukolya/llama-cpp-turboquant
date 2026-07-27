#pragma once

#include "../../src/llama-model.h"

#include "moe-expert-plan.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <set>
#include <string>
#include <utility>
#include <vector>

class server_moe_plan_validator {
public:
    server_moe_plan_validator(std::string plan_path, bool strict, bool dry_run) :
        plan_path_(std::move(plan_path)),
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

            for (int32_t layer_index = 0; layer_index < model_layer_count; ++layer_index) {
                const auto & model_layer = model->layers[static_cast<size_t>(layer_index)];
                const auto * plan_layer = find_plan_layer(plan, layer_index);
                if (plan_layer == nullptr) {
                    layout_errors.push_back("plan does not contain layer " + std::to_string(layer_index));
                    continue;
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

                if (tensors.empty()) {
                    layout_errors.push_back("layer " + std::to_string(layer_index) + ": no routed expert tensors");
                    continue;
                }

                int64_t layer_expert_count = -1;
                uint64_t logical_expert_bytes = 0;

                for (const ggml_tensor * tensor : tensors) {
                    if (tensor->ne[2] <= 0) {
                        layout_errors.push_back(
                            "layer " + std::to_string(layer_index) + ": tensor '" + tensor->name +
                            "' has no routed expert axis at ne[2]");
                        continue;
                    }
                    if (tensor->nb[2] == 0) {
                        layout_errors.push_back(
                            "layer " + std::to_string(layer_index) + ": tensor '" + tensor->name +
                            "' has zero expert stride nb[2]");
                        continue;
                    }

                    if (layer_expert_count < 0) {
                        layer_expert_count = tensor->ne[2];
                    } else if (layer_expert_count != tensor->ne[2]) {
                        layout_errors.push_back(
                            "layer " + std::to_string(layer_index) +
                            ": routed tensors disagree on expert count");
                    }

                    const uint64_t stride = static_cast<uint64_t>(tensor->nb[2]);
                    const uint64_t tensor_bytes = static_cast<uint64_t>(ggml_nbytes(tensor));
                    const uint64_t required_bytes = stride * static_cast<uint64_t>(tensor->ne[2]);
                    if (tensor_bytes < required_bytes) {
                        layout_errors.push_back(
                            "layer " + std::to_string(layer_index) + ": tensor '" + tensor->name +
                            "' is smaller than nb[2] * ne[2]");
                    }

                    logical_expert_bytes += stride;
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
                        static_cast<long long>(layer_expert_count - static_cast<int64_t>(plan_layer->gpu_experts.size())),
                        static_cast<unsigned long long>(logical_expert_bytes),
                        static_cast<unsigned long long>(intended_gpu_bytes));
                }
            }

            common_moe_expert_plan_expectation expectation;
            expectation.layer_count = model_layer_count;
            expectation.experts_per_layer = experts_per_layer;
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
            if (plan.model_fingerprint.empty()) {
                warnings.push_back("plan has no model_fingerprint; tensor-layout validation was used instead");
            }
            if (plan.placement_fingerprint.empty()) {
                warnings.push_back("plan has no placement_fingerprint");
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
                "moe_plan: validated '%s': layers=%d experts/layer=%d selected=%u bytes=%llu hit_rate=%.3f%%; "
                "dry run only, allocation and inference are unchanged\n",
                plan_path_.c_str(),
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
    static void add_tensor(
            std::vector<const ggml_tensor *> & tensors,
            std::set<const ggml_tensor *> & seen,
            const ggml_tensor * tensor) {
        if (tensor != nullptr && seen.insert(tensor).second) {
            tensors.push_back(tensor);
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

    bool fail(const std::string & message) const {
        std::fprintf(stderr, "moe_plan: %s: %s\n",
            strict_ ? "fatal" : "disabled",
            message.c_str());
        return !strict_;
    }

    std::string plan_path_;
    bool strict_ = false;
    bool dry_run_ = false;
};
