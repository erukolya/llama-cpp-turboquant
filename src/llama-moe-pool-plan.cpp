#include "llama-moe-pool-plan.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace {

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_mul(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

void append_span(
        std::vector<llama_moe_pool_copy_span> & spans,
        uint32_t global_expert,
        uint32_t local_expert,
        uint64_t stride) {
    if (!spans.empty()) {
        auto & previous = spans.back();
        if (previous.global_expert_first + previous.expert_count == global_expert &&
            previous.local_expert_first + previous.expert_count == local_expert) {
            ++previous.expert_count;
            previous.size_bytes += stride;
            return;
        }
    }

    llama_moe_pool_copy_span span;
    span.global_expert_first = global_expert;
    span.local_expert_first = local_expert;
    span.expert_count = 1;
    span.source_offset_bytes = static_cast<uint64_t>(global_expert) * stride;
    span.destination_offset_bytes = static_cast<uint64_t>(local_expert) * stride;
    span.size_bytes = stride;
    spans.push_back(span);
}

bool validate_destination_coverage(
        const std::vector<bool> & seen,
        const char * backend,
        std::string & error) {
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        error = std::string(backend) + " compact pool has an unmapped local expert";
        return false;
    }
    return true;
}

bool sum_span_bytes(
        const std::vector<llama_moe_pool_copy_span> & spans,
        uint64_t & total) {
    for (const auto & span : spans) {
        if (span.expert_count == 0 || span.size_bytes == 0 ||
            !checked_add(total, span.size_bytes, total)) {
            return false;
        }
    }
    return true;
}

} // namespace

bool llama_moe_pool_copy_spans_validate(
        uint64_t source_size_bytes,
        uint64_t destination_size_bytes,
        const std::vector<llama_moe_pool_copy_span> & spans,
        std::string & error) {
    if (destination_size_bytes == 0) {
        if (!spans.empty()) {
            error = "zero-sized compact tensor must not contain copy spans";
            return false;
        }
        error.clear();
        return true;
    }
    if (source_size_bytes == 0 || spans.empty()) {
        error = "non-empty compact tensor requires a source tensor and copy spans";
        return false;
    }

    struct interval {
        uint64_t begin;
        uint64_t end;
    };

    std::vector<interval> source_intervals;
    std::vector<interval> destination_intervals;
    source_intervals.reserve(spans.size());
    destination_intervals.reserve(spans.size());

    uint64_t total_bytes = 0;
    for (const auto & span : spans) {
        if (span.size_bytes == 0) {
            error = "copy span size is zero";
            return false;
        }

        uint64_t source_end = 0;
        uint64_t destination_end = 0;
        if (!checked_add(span.source_offset_bytes, span.size_bytes, source_end) ||
            source_end > source_size_bytes) {
            error = "copy span is outside the packed source tensor";
            return false;
        }
        if (!checked_add(span.destination_offset_bytes, span.size_bytes, destination_end) ||
            destination_end > destination_size_bytes) {
            error = "copy span is outside the compact destination tensor";
            return false;
        }
        if (!checked_add(total_bytes, span.size_bytes, total_bytes)) {
            error = "copy span byte count overflow";
            return false;
        }

        source_intervals.push_back({span.source_offset_bytes, source_end});
        destination_intervals.push_back({span.destination_offset_bytes, destination_end});
    }

    if (total_bytes != destination_size_bytes) {
        error = "copy spans do not contain exactly the compact destination byte count";
        return false;
    }

    auto by_begin = [](const interval & lhs, const interval & rhs) {
        return lhs.begin < rhs.begin || (lhs.begin == rhs.begin && lhs.end < rhs.end);
    };
    std::sort(source_intervals.begin(), source_intervals.end(), by_begin);
    std::sort(destination_intervals.begin(), destination_intervals.end(), by_begin);

    for (size_t i = 1; i < source_intervals.size(); ++i) {
        if (source_intervals[i].begin < source_intervals[i - 1].end) {
            error = "copy spans overlap in the packed source tensor";
            return false;
        }
    }

    uint64_t destination_cursor = 0;
    for (const auto & current : destination_intervals) {
        if (current.begin != destination_cursor) {
            error = current.begin < destination_cursor ?
                "copy spans overlap in the compact destination tensor" :
                "copy spans leave a gap in the compact destination tensor";
            return false;
        }
        destination_cursor = current.end;
    }
    if (destination_cursor != destination_size_bytes) {
        error = "copy spans do not cover the end of the compact destination tensor";
        return false;
    }

    error.clear();
    return true;
}

bool llama_moe_compact_tensor_pool_plan_build(
        const llama_moe_packed_tensor_layout & source,
        const llama_moe_load_layer_placement & placement,
        llama_moe_compact_tensor_pool_plan & plan,
        std::string & error) {
    if (source.layer < 0 || placement.layer != source.layer) {
        error = "packed tensor layer does not match placement layer";
        return false;
    }
    if (source.name.empty()) {
        error = "packed tensor name is empty";
        return false;
    }
    if (source.ne[0] <= 0 || source.ne[1] <= 0 || source.ne[2] <= 0 || source.ne[3] != 1) {
        error = "packed tensor must use a three-dimensional expert-axis-2 layout";
        return false;
    }
    if (static_cast<uint64_t>(source.ne[2]) != placement.expert_count ||
        placement.global_to_local.size() != placement.expert_count) {
        error = "packed tensor expert count does not match placement";
        return false;
    }
    if (placement.cpu_expert_count > placement.expert_count ||
        placement.gpu_expert_count > placement.expert_count ||
        placement.cpu_expert_count + placement.gpu_expert_count != placement.expert_count) {
        error = "placement backend expert counts do not match the layer expert count";
        return false;
    }
    if (source.nb[2] == 0) {
        error = "packed tensor expert stride is zero";
        return false;
    }

    uint64_t expected_source_bytes = 0;
    if (!checked_mul(source.nb[2], static_cast<uint64_t>(source.ne[2]), expected_source_bytes) ||
        expected_source_bytes != source.size_bytes) {
        error = "packed tensor size does not equal expert stride times expert count";
        return false;
    }

    llama_moe_compact_tensor_pool_plan built;
    built.layer = source.layer;
    built.source_name = source.name;
    built.expert_stride_bytes = source.nb[2];
    built.cpu_expert_count = placement.cpu_expert_count;
    built.gpu_expert_count = placement.gpu_expert_count;
    built.cpu_ne = source.ne;
    built.gpu_ne = source.ne;
    built.cpu_ne[2] = placement.cpu_expert_count;
    built.gpu_ne[2] = placement.gpu_expert_count;

    if (!checked_mul(source.nb[2], placement.cpu_expert_count, built.cpu_bytes) ||
        !checked_mul(source.nb[2], placement.gpu_expert_count, built.gpu_bytes)) {
        error = "compact tensor pool byte count overflow";
        return false;
    }

    std::vector<bool> cpu_local_seen(placement.cpu_expert_count, false);
    std::vector<bool> gpu_local_seen(placement.gpu_expert_count, false);

    for (uint32_t global_expert = 0; global_expert < placement.expert_count; ++global_expert) {
        const auto * location = placement.find(global_expert);
        if (location == nullptr) {
            error = "placement is missing a global expert";
            return false;
        }
        if (location->backend == llama_moe_load_backend::cpu) {
            if (location->local_index >= placement.cpu_expert_count) {
                error = "CPU local expert index is outside compact pool";
                return false;
            }
            if (cpu_local_seen[location->local_index]) {
                error = "CPU compact pool contains a duplicate local expert index";
                return false;
            }
            cpu_local_seen[location->local_index] = true;
            append_span(built.cpu_spans, global_expert, location->local_index, source.nb[2]);
        } else if (location->backend == llama_moe_load_backend::gpu) {
            if (location->local_index >= placement.gpu_expert_count) {
                error = "GPU local expert index is outside compact pool";
                return false;
            }
            if (gpu_local_seen[location->local_index]) {
                error = "GPU compact pool contains a duplicate local expert index";
                return false;
            }
            gpu_local_seen[location->local_index] = true;
            append_span(built.gpu_spans, global_expert, location->local_index, source.nb[2]);
        } else {
            error = "placement contains an unknown backend";
            return false;
        }
    }

    if (!validate_destination_coverage(cpu_local_seen, "CPU", error) ||
        !validate_destination_coverage(gpu_local_seen, "GPU", error)) {
        return false;
    }

    if (!llama_moe_pool_copy_spans_validate(
            source.size_bytes, built.cpu_bytes, built.cpu_spans, error) ||
        !llama_moe_pool_copy_spans_validate(
            source.size_bytes, built.gpu_bytes, built.gpu_spans, error)) {
        return false;
    }

    uint64_t pool_bytes = 0;
    if (!checked_add(built.cpu_bytes, built.gpu_bytes, pool_bytes) || pool_bytes != source.size_bytes) {
        error = "compact CPU and GPU pools do not cover the complete source tensor";
        return false;
    }

    uint64_t copied_bytes = 0;
    if (!sum_span_bytes(built.cpu_spans, copied_bytes) ||
        !sum_span_bytes(built.gpu_spans, copied_bytes) ||
        copied_bytes != source.size_bytes) {
        error = "compact copy spans do not cover the complete source tensor";
        return false;
    }

    plan = std::move(built);
    error.clear();
    return true;
}
