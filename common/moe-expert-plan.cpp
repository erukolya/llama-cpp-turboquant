#include "moe-expert-plan.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

using json = nlohmann::ordered_json;

namespace {

const json & require_member(const json & object, const char * name) {
    if (!object.is_object()) {
        throw std::runtime_error("expected JSON object");
    }
    const auto it = object.find(name);
    if (it == object.end()) {
        throw std::runtime_error(std::string("missing required field '") + name + "'");
    }
    return *it;
}

uint64_t read_u64(const json & object, const char * name) {
    const json & value = require_member(object, name);
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    }
    if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value >= 0) {
            return static_cast<uint64_t>(signed_value);
        }
    }
    throw std::runtime_error(std::string("field '") + name + "' must be a non-negative integer");
}

uint32_t read_u32(const json & object, const char * name) {
    const uint64_t value = read_u64(object, name);
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("field '") + name + "' exceeds uint32 range");
    }
    return static_cast<uint32_t>(value);
}

int32_t read_i32(const json & object, const char * name) {
    const json & value = require_member(object, name);
    if (!value.is_number_integer()) {
        throw std::runtime_error(std::string("field '") + name + "' must be an integer");
    }
    const int64_t parsed = value.get<int64_t>();
    if (parsed < std::numeric_limits<int32_t>::min() || parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("field '") + name + "' exceeds int32 range");
    }
    return static_cast<int32_t>(parsed);
}

int64_t read_i64_value(const json & value, const char * name) {
    if (!value.is_number_integer()) {
        throw std::runtime_error(std::string("field '") + name + "' contains a non-integer item");
    }
    return value.get<int64_t>();
}

uint64_t read_u64_value(const json & value, const char * name) {
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    }
    if (value.is_number_integer()) {
        const int64_t parsed = value.get<int64_t>();
        if (parsed >= 0) {
            return static_cast<uint64_t>(parsed);
        }
    }
    throw std::runtime_error(std::string("field '") + name + "' contains a negative or non-integer item");
}

double read_double(const json & object, const char * name) {
    const json & value = require_member(object, name);
    if (!value.is_number()) {
        throw std::runtime_error(std::string("field '") + name + "' must be numeric");
    }
    const double parsed = value.get<double>();
    if (!std::isfinite(parsed)) {
        throw std::runtime_error(std::string("field '") + name + "' must be finite");
    }
    return parsed;
}

std::string read_string(const json & object, const char * name) {
    const json & value = require_member(object, name);
    if (!value.is_string()) {
        throw std::runtime_error(std::string("field '") + name + "' must be a string");
    }
    return value.get<std::string>();
}

std::string read_nullable_string(const json & object, const char * name) {
    const json & value = require_member(object, name);
    if (value.is_null()) {
        return {};
    }
    if (!value.is_string()) {
        throw std::runtime_error(std::string("field '") + name + "' must be a string or null");
    }
    return value.get<std::string>();
}

std::vector<uint32_t> read_u32_array(const json & object, const char * name) {
    const json & value = require_member(object, name);
    if (!value.is_array()) {
        throw std::runtime_error(std::string("field '") + name + "' must be an array");
    }
    std::vector<uint32_t> result;
    result.reserve(value.size());
    for (const auto & item : value) {
        const uint64_t parsed = read_u64_value(item, name);
        if (parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(std::string("field '") + name + "' contains an out-of-range item");
        }
        result.push_back(static_cast<uint32_t>(parsed));
    }
    return result;
}

std::vector<std::string> read_string_array(const json & object, const char * name) {
    const json & value = require_member(object, name);
    if (!value.is_array()) {
        throw std::runtime_error(std::string("field '") + name + "' must be an array");
    }
    std::vector<std::string> result;
    result.reserve(value.size());
    for (const auto & item : value) {
        if (!item.is_string()) {
            throw std::runtime_error(std::string("field '") + name + "' contains a non-string item");
        }
        result.push_back(item.get<std::string>());
    }
    return result;
}

std::array<int64_t, 4> read_i64_array4(const json & object, const char * name) {
    const json & value = require_member(object, name);
    if (!value.is_array() || value.size() != 4) {
        throw std::runtime_error(std::string("field '") + name + "' must contain exactly four integers");
    }
    std::array<int64_t, 4> result;
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = read_i64_value(value[index], name);
    }
    return result;
}

std::array<uint64_t, 4> read_u64_array4(const json & object, const char * name) {
    const json & value = require_member(object, name);
    if (!value.is_array() || value.size() != 4) {
        throw std::runtime_error(std::string("field '") + name + "' must contain exactly four integers");
    }
    std::array<uint64_t, 4> result;
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = read_u64_value(value[index], name);
    }
    return result;
}

common_moe_expert_plan parse_plan(const json & root) {
    common_moe_expert_plan plan;
    plan.schema_version = read_u32(root, "schema_version");
    plan.strategy = read_string(root, "strategy");
    plan.model_fingerprint = read_nullable_string(root, "model_fingerprint");
    plan.placement_fingerprint = read_string(root, "placement_fingerprint");
    plan.source_profiles = read_string_array(root, "source_profiles");
    plan.source_placement = read_string(root, "source_placement");
    plan.vram_budget_bytes = read_u64(root, "vram_budget_bytes");
    plan.selected_bytes = read_u64(root, "selected_bytes");
    plan.unused_budget_bytes = read_u64(root, "unused_budget_bytes");
    plan.logical_expert_count = read_u32(root, "logical_expert_count");
    plan.selected_expert_count = read_u32(root, "selected_expert_count");
    plan.estimated_total_hits = read_u64(root, "estimated_total_hits");
    plan.estimated_gpu_hits = read_u64(root, "estimated_gpu_hits");
    plan.estimated_gpu_hit_rate = read_double(root, "estimated_gpu_hit_rate");

    if (plan.schema_version >= 2) {
        const json & manifest = require_member(root, "tensor_manifest");
        if (!manifest.is_array()) {
            throw std::runtime_error("field 'tensor_manifest' must be an array");
        }
        plan.tensor_manifest.reserve(manifest.size());
        for (const auto & item : manifest) {
            common_moe_expert_plan_tensor tensor;
            tensor.layer = read_i32(item, "layer");
            tensor.name = read_string(item, "name");
            tensor.type = read_string(item, "type");
            tensor.expert_first = read_u32(item, "expert_first");
            tensor.expert_last = read_u32(item, "expert_last");
            tensor.size_bytes = read_u64(item, "size_bytes");
            tensor.expert_stride_bytes = read_u64(item, "expert_stride_bytes");
            tensor.ne = read_i64_array4(item, "ne");
            tensor.nb = read_u64_array4(item, "nb");
            plan.tensor_manifest.push_back(std::move(tensor));
        }
    }

    const json & layers = require_member(root, "layers");
    if (!layers.is_array()) {
        throw std::runtime_error("field 'layers' must be an array");
    }
    plan.layers.reserve(layers.size());
    for (const auto & item : layers) {
        common_moe_expert_plan_layer layer;
        layer.layer = read_i32(item, "layer");
        layer.expert_count = read_u32(item, "expert_count");
        layer.gpu_experts = read_u32_array(item, "gpu_experts");
        layer.gpu_bytes = read_u64(item, "gpu_bytes");
        layer.estimated_hits = read_u64(item, "estimated_hits");
        layer.estimated_gpu_hits = read_u64(item, "estimated_gpu_hits");
        layer.estimated_gpu_hit_rate = read_double(item, "estimated_gpu_hit_rate");
        plan.layers.push_back(std::move(layer));
    }
    return plan;
}

json serialize_plan(const common_moe_expert_plan & plan) {
    json layers = json::array();
    for (const auto & layer : plan.layers) {
        layers.push_back({
            {"layer", layer.layer},
            {"expert_count", layer.expert_count},
            {"gpu_experts", layer.gpu_experts},
            {"gpu_bytes", layer.gpu_bytes},
            {"estimated_hits", layer.estimated_hits},
            {"estimated_gpu_hits", layer.estimated_gpu_hits},
            {"estimated_gpu_hit_rate", layer.estimated_gpu_hit_rate},
        });
    }

    json root = {
        {"schema_version", plan.schema_version},
        {"strategy", plan.strategy},
        {"model_fingerprint", plan.model_fingerprint.empty() ? json(nullptr) : json(plan.model_fingerprint)},
        {"placement_fingerprint", plan.placement_fingerprint},
        {"source_profiles", plan.source_profiles},
        {"source_placement", plan.source_placement},
        {"vram_budget_bytes", plan.vram_budget_bytes},
        {"selected_bytes", plan.selected_bytes},
        {"unused_budget_bytes", plan.unused_budget_bytes},
        {"logical_expert_count", plan.logical_expert_count},
        {"selected_expert_count", plan.selected_expert_count},
        {"estimated_total_hits", plan.estimated_total_hits},
        {"estimated_gpu_hits", plan.estimated_gpu_hits},
        {"estimated_gpu_hit_rate", plan.estimated_gpu_hit_rate},
    };

    if (plan.schema_version >= 2 || !plan.tensor_manifest.empty()) {
        json manifest = json::array();
        for (const auto & tensor : plan.tensor_manifest) {
            manifest.push_back({
                {"layer", tensor.layer},
                {"name", tensor.name},
                {"type", tensor.type},
                {"expert_first", tensor.expert_first},
                {"expert_last", tensor.expert_last},
                {"size_bytes", tensor.size_bytes},
                {"expert_stride_bytes", tensor.expert_stride_bytes},
                {"ne", tensor.ne},
                {"nb", tensor.nb},
            });
        }
        root["tensor_manifest"] = std::move(manifest);
    }

    root["layers"] = std::move(layers);
    return root;
}

void add_error(common_moe_expert_plan_validation & result, const std::string & value) {
    result.errors.push_back(value);
}

void add_warning(common_moe_expert_plan_validation & result, const std::string & value) {
    result.warnings.push_back(value);
}

bool valid_rate(double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

} // namespace

bool common_moe_expert_plan_load(
        const std::string & path,
        common_moe_expert_plan & plan,
        std::string & error) {
    try {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to open plan file");
        }
        json root;
        input >> root;
        plan = parse_plan(root);
        error.clear();
        return true;
    } catch (const std::exception & exception) {
        error = path + ": " + exception.what();
        return false;
    }
}

bool common_moe_expert_plan_save(
        const std::string & path,
        const common_moe_expert_plan & plan,
        std::string & error) {
    try {
        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("failed to open plan file for writing");
        }
        output << serialize_plan(plan).dump(2) << '\n';
        output.flush();
        if (!output) {
            throw std::runtime_error("failed while writing plan file");
        }
        error.clear();
        return true;
    } catch (const std::exception & exception) {
        error = path + ": " + exception.what();
        return false;
    }
}

common_moe_expert_plan_validation common_moe_expert_plan_validate(
        const common_moe_expert_plan & plan,
        const common_moe_expert_plan_expectation & expectation) {
    common_moe_expert_plan_validation result;

    if (plan.schema_version != 1 && plan.schema_version != 2) {
        add_error(result, "unsupported schema_version: " + std::to_string(plan.schema_version));
    }
    if (plan.strategy.empty()) {
        add_error(result, "strategy is empty");
    }
    if (plan.placement_fingerprint.empty()) {
        add_error(result, "placement_fingerprint is empty");
    }
    if (plan.selected_bytes > plan.vram_budget_bytes) {
        add_error(result, "selected_bytes exceeds vram_budget_bytes");
    }
    if (plan.unused_budget_bytes != plan.vram_budget_bytes - std::min(plan.selected_bytes, plan.vram_budget_bytes)) {
        add_error(result, "unused_budget_bytes is inconsistent with budget and selected bytes");
    }
    if (plan.estimated_gpu_hits > plan.estimated_total_hits) {
        add_error(result, "estimated_gpu_hits exceeds estimated_total_hits");
    }
    if (!valid_rate(plan.estimated_gpu_hit_rate)) {
        add_error(result, "estimated_gpu_hit_rate is outside [0, 1]");
    }

    uint64_t summed_bytes = 0;
    uint64_t summed_hits = 0;
    uint64_t summed_gpu_hits = 0;
    uint64_t selected_count = 0;
    uint64_t logical_count = 0;
    std::set<int32_t> layer_ids;

    for (const auto & layer : plan.layers) {
        if (layer.layer < 0) {
            add_error(result, "layer index is negative");
            continue;
        }
        if (!layer_ids.insert(layer.layer).second) {
            add_error(result, "duplicate layer entry: " + std::to_string(layer.layer));
        }
        if (layer.expert_count == 0) {
            add_error(result, "layer " + std::to_string(layer.layer) + " has zero experts");
        }
        if (!std::is_sorted(layer.gpu_experts.begin(), layer.gpu_experts.end())) {
            add_error(result, "layer " + std::to_string(layer.layer) + " gpu_experts is not sorted");
        }
        if (std::adjacent_find(layer.gpu_experts.begin(), layer.gpu_experts.end()) != layer.gpu_experts.end()) {
            add_error(result, "layer " + std::to_string(layer.layer) + " gpu_experts contains duplicates");
        }
        for (uint32_t expert : layer.gpu_experts) {
            if (expert >= layer.expert_count) {
                add_error(result, "layer " + std::to_string(layer.layer) +
                    " contains out-of-range GPU expert " + std::to_string(expert));
            }
        }
        if (layer.estimated_gpu_hits > layer.estimated_hits) {
            add_error(result, "layer " + std::to_string(layer.layer) + " GPU hits exceed layer hits");
        }
        if (!valid_rate(layer.estimated_gpu_hit_rate)) {
            add_error(result, "layer " + std::to_string(layer.layer) + " GPU hit rate is outside [0, 1]");
        }
        if (layer.estimated_hits == 0 && layer.estimated_gpu_hit_rate != 0.0) {
            add_warning(result, "layer " + std::to_string(layer.layer) +
                " has a non-zero hit rate with zero hits");
        }

        summed_bytes += layer.gpu_bytes;
        summed_hits += layer.estimated_hits;
        summed_gpu_hits += layer.estimated_gpu_hits;
        selected_count += layer.gpu_experts.size();
        logical_count += layer.expert_count;
    }

    if (summed_bytes != plan.selected_bytes) {
        add_error(result, "sum of layer gpu_bytes differs from selected_bytes");
    }
    if (summed_hits != plan.estimated_total_hits) {
        add_error(result, "sum of layer estimated_hits differs from estimated_total_hits");
    }
    if (summed_gpu_hits != plan.estimated_gpu_hits) {
        add_error(result, "sum of layer estimated_gpu_hits differs from estimated_gpu_hits");
    }
    if (selected_count != plan.selected_expert_count) {
        add_error(result, "selected_expert_count differs from layer GPU expert lists");
    }
    if (logical_count != plan.logical_expert_count) {
        add_error(result, "logical_expert_count differs from layer expert counts");
    }

    if (expectation.layer_count >= 0 && static_cast<int32_t>(plan.layers.size()) != expectation.layer_count) {
        add_error(result, "plan layer count does not match model expectation");
    }
    if (expectation.experts_per_layer >= 0) {
        for (const auto & layer : plan.layers) {
            if (static_cast<int32_t>(layer.expert_count) != expectation.experts_per_layer) {
                add_error(result, "layer " + std::to_string(layer.layer) +
                    " expert count does not match model expectation");
            }
        }
    }
    if (!expectation.model_fingerprint.empty()) {
        if (plan.model_fingerprint.empty()) {
            add_error(result, "plan has no model_fingerprint but strict model matching was requested");
        } else if (plan.model_fingerprint != expectation.model_fingerprint) {
            add_error(result, "model_fingerprint mismatch");
        }
    } else if (plan.model_fingerprint.empty()) {
        add_warning(result, "plan has no model_fingerprint; only structural validation is possible");
    }

    if (plan.schema_version >= 2 && plan.tensor_manifest.empty()) {
        add_error(result, "schema v2 plan has an empty tensor_manifest");
    }
    if (expectation.require_tensor_manifest && plan.tensor_manifest.empty()) {
        add_error(result, "tensor_manifest is required");
    }

    std::set<std::pair<int32_t, std::string>> tensor_keys;
    std::map<int32_t, std::set<std::pair<uint32_t, uint32_t>>> ranges_by_layer;
    for (const auto & tensor : plan.tensor_manifest) {
        const std::string label = "tensor '" + tensor.name + "'";
        if (tensor.layer < 0) {
            add_error(result, label + " has a negative layer");
        }
        if (tensor.name.empty()) {
            add_error(result, "tensor manifest contains an empty name");
        }
        if (tensor.type.empty()) {
            add_error(result, label + " has an empty type");
        }
        if (!tensor_keys.insert({tensor.layer, tensor.name}).second) {
            add_error(result, "duplicate tensor manifest entry: layer " +
                std::to_string(tensor.layer) + ", " + tensor.name);
        }
        if (layer_ids.find(tensor.layer) == layer_ids.end()) {
            add_error(result, label + " references a layer absent from the placement plan");
        }
        if (tensor.expert_last < tensor.expert_first) {
            add_error(result, label + " has an invalid expert range");
        }
        const uint64_t expert_count =
            static_cast<uint64_t>(tensor.expert_last) - tensor.expert_first + 1;
        if (tensor.ne[2] <= 0 || static_cast<uint64_t>(tensor.ne[2]) != expert_count) {
            add_error(result, label + " ne[2] differs from its expert range");
        }
        if (tensor.expert_stride_bytes == 0 || tensor.nb[2] != tensor.expert_stride_bytes) {
            add_error(result, label + " nb[2] differs from expert_stride_bytes");
        }
        for (size_t dimension = 0; dimension < 4; ++dimension) {
            if (tensor.ne[dimension] <= 0) {
                add_error(result, label + " contains a non-positive ne value");
            }
            if (tensor.nb[dimension] == 0) {
                add_error(result, label + " contains a zero nb value");
            }
        }
        if (tensor.size_bytes == 0) {
            add_error(result, label + " has zero size_bytes");
        } else if (tensor.size_bytes <=
                tensor.expert_stride_bytes * static_cast<uint64_t>(tensor.expert_last)) {
            add_error(result, label + " does not contain the start of its last expert slice");
        }
        ranges_by_layer[tensor.layer].insert({tensor.expert_first, tensor.expert_last});
    }
    for (const auto & item : ranges_by_layer) {
        if (item.second.size() != 1) {
            add_error(result, "layer " + std::to_string(item.first) +
                " tensor manifest contains inconsistent expert ranges");
        }
    }

    return result;
}
