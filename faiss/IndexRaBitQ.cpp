/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRaBitQ.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizerMultiBit.h>
#include <faiss/impl/ResultHandler.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

namespace faiss {

namespace {

bool mode_uses_expanded_cache(RaBitQFullCodeMode mode) {
    return mode == RABITQ_FULL_CODE_EXPANDED || mode == RABITQ_FULL_CODE_INT8;
}

bool mode_uses_progressive_rows(RaBitQFullCodeMode mode) {
    return mode == RABITQ_FULL_CODE_PROGRESSIVE ||
            mode == RABITQ_FULL_CODE_NESTED_LUT7 ||
            mode == RABITQ_FULL_CODE_NESTED_LUT4 ||
            mode == RABITQ_FULL_CODE_NIBBLE_LUT4;
}

std::vector<uint8_t> train_nested_lut7(
        idx_t n,
        idx_t d,
        const float* x,
        const float* center) {
    constexpr size_t kLevels = 64;
    constexpr size_t kEntries = 32;
    std::array<std::array<uint64_t, kLevels>, 2> histogram{};
    std::vector<float> normalized_abs(static_cast<size_t>(d));
    for (idx_t row = 0; row < n; ++row) {
        float norm_sqr = 0.0f;
        for (idx_t j = 0; j < d; ++j) {
            const float value = x[row * d + j] - center[j];
            norm_sqr += value * value;
            normalized_abs[static_cast<size_t>(j)] = std::abs(value);
        }
        const float norm = std::sqrt(norm_sqr);
        if (norm < 1e-10f) {
            continue;
        }
        for (float& value : normalized_abs) {
            value /= norm;
        }
        const float prefix_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), static_cast<size_t>(d), 2);
        const float full_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), static_cast<size_t>(d), 7);
        for (idx_t j = 0; j < d; ++j) {
            const float magnitude = normalized_abs[static_cast<size_t>(j)];
            const size_t coarse =
                    std::min<size_t>(size_t(prefix_t * magnitude + 1e-5f), 1);
            const size_t fine =
                    std::min<size_t>(size_t(full_t * magnitude + 1e-5f), 63);
            ++histogram[coarse][fine];
        }
    }

    std::vector<uint8_t> lut(2 * kEntries);
    for (size_t cell = 0; cell < 2; ++cell) {
        double interval_cost[kLevels][kLevels + 1]{};
        uint8_t representative[kLevels][kLevels + 1]{};
        for (size_t begin = 0; begin < kLevels; ++begin) {
            for (size_t end = begin + 1; end <= kLevels; ++end) {
                uint64_t total = 0;
                uint64_t weighted_sum = 0;
                for (size_t level = begin; level < end; ++level) {
                    total += histogram[cell][level];
                    weighted_sum += histogram[cell][level] * level;
                }
                size_t value = (begin + end - 1) / 2;
                if (total != 0) {
                    value = static_cast<size_t>(std::nearbyint(
                            static_cast<double>(weighted_sum) / total));
                    value = std::clamp(value, begin, end - 1);
                }
                representative[begin][end] = static_cast<uint8_t>(value);
                double cost = 0.0;
                for (size_t level = begin; level < end; ++level) {
                    const double delta = static_cast<double>(level) - value;
                    cost += histogram[cell][level] * delta * delta;
                }
                interval_cost[begin][end] = cost;
            }
        }

        constexpr double kInfinity = std::numeric_limits<double>::infinity();
        double dp[kEntries + 1][kLevels + 1];
        int previous[kEntries + 1][kLevels + 1];
        for (size_t groups = 0; groups <= kEntries; ++groups) {
            for (size_t end = 0; end <= kLevels; ++end) {
                dp[groups][end] = kInfinity;
                previous[groups][end] = -1;
            }
        }
        dp[0][0] = 0.0;
        for (size_t groups = 1; groups <= kEntries; ++groups) {
            for (size_t end = groups; end <= kLevels; ++end) {
                for (size_t begin = groups - 1; begin < end; ++begin) {
                    const double candidate =
                            dp[groups - 1][begin] + interval_cost[begin][end];
                    if (candidate < dp[groups][end]) {
                        dp[groups][end] = candidate;
                        previous[groups][end] = static_cast<int>(begin);
                    }
                }
            }
        }
        size_t end = kLevels;
        for (size_t groups = kEntries; groups > 0; --groups) {
            const size_t begin = static_cast<size_t>(previous[groups][end]);
            lut[cell * kEntries + groups - 1] = representative[begin][end];
            end = begin;
        }
    }
    return lut;
}

std::vector<uint8_t> train_nested_lut4(
        idx_t n,
        idx_t d,
        const float* x,
        const float* center) {
    constexpr size_t kLevels = 8;
    constexpr size_t kEntries = 4;
    std::array<std::array<uint64_t, kLevels>, 2> histogram{};
    std::vector<float> normalized_abs(static_cast<size_t>(d));
    for (idx_t row = 0; row < n; ++row) {
        float norm_sqr = 0.0f;
        for (idx_t j = 0; j < d; ++j) {
            const float value = x[row * d + j] - center[j];
            norm_sqr += value * value;
            normalized_abs[static_cast<size_t>(j)] = std::abs(value);
        }
        const float norm = std::sqrt(norm_sqr);
        if (norm < 1e-10f) {
            continue;
        }
        for (float& value : normalized_abs) {
            value /= norm;
        }
        const float prefix_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), static_cast<size_t>(d), 2);
        const float full_t = rabitq_multibit::compute_optimal_scaling_factor(
                normalized_abs.data(), static_cast<size_t>(d), 4);
        for (idx_t j = 0; j < d; ++j) {
            const float magnitude = normalized_abs[static_cast<size_t>(j)];
            const size_t coarse =
                    std::min<size_t>(size_t(prefix_t * magnitude + 1e-5f), 1);
            const size_t fine =
                    std::min<size_t>(size_t(full_t * magnitude + 1e-5f), 7);
            ++histogram[coarse][fine];
        }
    }

    std::vector<uint8_t> lut(2 * kEntries);
    for (size_t cell = 0; cell < 2; ++cell) {
        double interval_cost[kLevels][kLevels + 1]{};
        uint8_t representative[kLevels][kLevels + 1]{};
        for (size_t begin = 0; begin < kLevels; ++begin) {
            for (size_t end = begin + 1; end <= kLevels; ++end) {
                uint64_t total = 0;
                uint64_t weighted_sum = 0;
                for (size_t level = begin; level < end; ++level) {
                    total += histogram[cell][level];
                    weighted_sum += histogram[cell][level] * level;
                }
                size_t value = (begin + end - 1) / 2;
                if (total != 0) {
                    value = static_cast<size_t>(std::nearbyint(
                            static_cast<double>(weighted_sum) / total));
                    value = std::clamp(value, begin, end - 1);
                }
                representative[begin][end] = static_cast<uint8_t>(value);
                double cost = 0.0;
                for (size_t level = begin; level < end; ++level) {
                    const double delta = static_cast<double>(level) - value;
                    cost += histogram[cell][level] * delta * delta;
                }
                interval_cost[begin][end] = cost;
            }
        }
        constexpr double kInfinity = std::numeric_limits<double>::infinity();
        double dp[kEntries + 1][kLevels + 1];
        int previous[kEntries + 1][kLevels + 1];
        for (size_t groups = 0; groups <= kEntries; ++groups) {
            for (size_t end = 0; end <= kLevels; ++end) {
                dp[groups][end] = kInfinity;
                previous[groups][end] = -1;
            }
        }
        dp[0][0] = 0.0;
        for (size_t groups = 1; groups <= kEntries; ++groups) {
            for (size_t end = groups; end <= kLevels; ++end) {
                for (size_t begin = groups - 1; begin < end; ++begin) {
                    const double candidate =
                            dp[groups - 1][begin] + interval_cost[begin][end];
                    if (candidate < dp[groups][end]) {
                        dp[groups][end] = candidate;
                        previous[groups][end] = static_cast<int>(begin);
                    }
                }
            }
        }
        size_t end = kLevels;
        for (size_t groups = kEntries; groups > 0; --groups) {
            const size_t begin = static_cast<size_t>(previous[groups][end]);
            lut[cell * kEntries + groups - 1] = representative[begin][end];
            end = begin;
        }
    }
    return lut;
}

} // namespace

// Forward declaration from RaBitQuantizer.cpp
struct RaBitQDistanceComputer;

using rabitq_utils::ExtraBitsFactors;
using rabitq_utils::SignBitFactors;
using rabitq_utils::SignBitFactorsWithError;

IndexRaBitQ::IndexRaBitQ() = default;

IndexRaBitQ::IndexRaBitQ(idx_t d_in, MetricType metric, uint8_t nb_bits_in)
        : IndexFlatCodes(0, d_in, metric), rabitq(d_in, metric, nb_bits_in) {
    // Update code size based on nb_bits
    code_size = rabitq.code_size;

    is_trained = false;
}

void IndexRaBitQ::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            full_code_mode == RABITQ_FULL_CODE_PACKED || ntotal == 0,
            "reset the RaBitQ index before retraining a nonpacked layout");
    FAISS_THROW_IF_NOT_MSG(
            (full_code_mode != RABITQ_FULL_CODE_NESTED_LUT7 &&
             full_code_mode != RABITQ_FULL_CODE_NESTED_LUT4 &&
             full_code_mode != RABITQ_FULL_CODE_NIBBLE_LUT4) ||
                    n > 0,
            "nested LUT training requires at least one vector");
    // compute a centroid
    std::vector<float> centroid(d, 0);
    for (idx_t i = 0; i < n; i++) {
        for (size_t j = 0; j < static_cast<size_t>(d); j++) {
            centroid[j] += x[i * d + j];
        }
    }

    if (n != 0) {
        for (size_t j = 0; j < static_cast<size_t>(d); j++) {
            centroid[j] /= (float)n;
        }
    }

    center = std::move(centroid);

    if (full_code_mode == RABITQ_FULL_CODE_NESTED_LUT7) {
        nested_lut7 = train_nested_lut7(n, d, x, center.data());
    } else {
        nested_lut7.clear();
    }
    if (full_code_mode == RABITQ_FULL_CODE_NESTED_LUT4 ||
        full_code_mode == RABITQ_FULL_CODE_NIBBLE_LUT4) {
        nested_lut4 = train_nested_lut4(n, d, x, center.data());
    } else {
        nested_lut4.clear();
    }

    //
    rabitq.train(n, x);
    is_trained = true;
}

void IndexRaBitQ::add(idx_t n, const float* x) {
    IndexFlatCodes::add(n, x);
    if (mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::add_sa_codes(idx_t n, const uint8_t* x, const idx_t* xids) {
    IndexFlatCodes::add_sa_codes(n, x, xids);
    if (mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::reset() {
    IndexFlatCodes::reset();
    expanded_codes.clear();
}

size_t IndexRaBitQ::remove_ids(const IDSelector& sel) {
    const size_t removed = IndexFlatCodes::remove_ids(sel);
    if (removed > 0 && mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
    return removed;
}

void IndexRaBitQ::merge_from(Index& other_index, idx_t add_id) {
    IndexFlatCodes::merge_from(other_index, add_id);
    if (mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::permute_entries(const idx_t* perm) {
    IndexFlatCodes::permute_entries(perm);
    if (mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    if (full_code_mode == RABITQ_FULL_CODE_NESTED_LUT7) {
        rabitq.compute_nested_lut7_codes_core(
                x, bytes, n, center.data(), nested_lut7.data());
    } else if (
            full_code_mode == RABITQ_FULL_CODE_NESTED_LUT4 ||
            full_code_mode == RABITQ_FULL_CODE_NIBBLE_LUT4) {
        rabitq.compute_nested_lut4_codes_core(
                x,
                bytes,
                n,
                center.data(),
                nested_lut4.data(),
                full_code_mode == RABITQ_FULL_CODE_NIBBLE_LUT4);
    } else if (full_code_mode == RABITQ_FULL_CODE_PROGRESSIVE) {
        rabitq.compute_progressive_codes_core(x, bytes, n, center.data());
    } else {
        rabitq.compute_codes_core(x, bytes, n, center.data());
    }
}

void IndexRaBitQ::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(
            !mode_uses_progressive_rows(full_code_mode),
            "sa_decode is unavailable after progressive finalization");
    rabitq.decode_core(bytes, x, n, center.data());
}

FlatCodesDistanceComputer* IndexRaBitQ::get_FlatCodesDistanceComputer() const {
    if (full_code_mode == RABITQ_FULL_CODE_PROGRESSIVE) {
        return rabitq.get_progressive_integer_distance_computer(
                codes.data(), center.data(), false);
    }
    if (full_code_mode == RABITQ_FULL_CODE_NESTED_LUT7) {
        return rabitq.get_nested_lut7_integer_distance_computer(
                codes.data(), center.data(), false, nested_lut7.data());
    }
    if (full_code_mode == RABITQ_FULL_CODE_NESTED_LUT4) {
        return rabitq.get_nested_lut4_integer_distance_computer(
                codes.data(), center.data(), false, false, nested_lut4.data());
    }
    if (full_code_mode == RABITQ_FULL_CODE_NIBBLE_LUT4) {
        return rabitq.get_nested_lut4_integer_distance_computer(
                codes.data(), center.data(), false, true, nested_lut4.data());
    }
    if (full_code_mode == RABITQ_FULL_CODE_INT8_PACKED) {
        return rabitq.get_packed_integer_distance_computer(
                codes.data(), center.data());
    }
    if (mode_uses_expanded_cache(full_code_mode)) {
        FAISS_THROW_IF_NOT_MSG(
                expanded_codes.size() ==
                        static_cast<size_t>(ntotal) * expanded_code_size(),
                "stale expanded RaBitQ cache; call set_full_code_mode again");
        return rabitq.get_expanded_distance_computer(
                expanded_codes.data(),
                center.data(),
                full_code_mode == RABITQ_FULL_CODE_INT8);
    }
    FlatCodesDistanceComputer* dc =
            rabitq.get_distance_computer(qb, center.data(), centered);
    dc->code_size = rabitq.code_size;
    dc->codes = codes.data();
    return dc;
}

size_t IndexRaBitQ::expanded_code_size() const {
    return static_cast<size_t>(d) + sizeof(rabitq_utils::ExtraBitsFactors);
}

void IndexRaBitQ::rebuild_expanded_codes() {
    if (!mode_uses_expanded_cache(full_code_mode)) {
        std::vector<uint8_t>().swap(expanded_codes);
        return;
    }
    expanded_codes.resize(static_cast<size_t>(ntotal) * expanded_code_size());
    rabitq.expand_codes(codes.data(), ntotal, expanded_codes.data());
}

void IndexRaBitQ::set_full_code_mode(uint8_t mode) {
    if (mode_uses_progressive_rows(full_code_mode)) {
        FAISS_THROW_IF_NOT_MSG(
                mode == full_code_mode,
                "cannot reinterpret progressive rows as another layout");
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            mode == RABITQ_FULL_CODE_PACKED ||
                    mode == RABITQ_FULL_CODE_EXPANDED ||
                    mode == RABITQ_FULL_CODE_INT8 ||
                    mode == RABITQ_FULL_CODE_INT8_PACKED,
            "invalid RaBitQ full-code mode");
    if (mode != RABITQ_FULL_CODE_PACKED) {
        FAISS_THROW_IF_NOT_MSG(
                metric_type == METRIC_L2,
                "integer RaBitQ ADC supports only L2");
        FAISS_THROW_IF_NOT_MSG(
                rabitq.nb_bits >= 2 && rabitq.nb_bits <= 8,
                "integer RaBitQ ADC requires 2..8 total bits");
    }
    full_code_mode = static_cast<RaBitQFullCodeMode>(mode);
    rebuild_expanded_codes();
}

bool IndexRaBitQ::expanded_integer_uses_native_dotprod() const {
    return rabitq.expanded_integer_uses_native_dotprod();
}

void IndexRaBitQ::compute_distance_subset(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(n >= 0 && k > 0, "invalid rerank shape");
    FAISS_THROW_IF_NOT_MSG(x && distances && labels, "null rerank buffer");
#pragma omp parallel
    {
        std::unique_ptr<FlatCodesDistanceComputer> dc(
                full_code_mode == RABITQ_FULL_CODE_PROGRESSIVE
                        ? rabitq.get_progressive_integer_distance_computer(
                                  codes.data(), center.data(), false)
                        : full_code_mode == RABITQ_FULL_CODE_NESTED_LUT7
                        ? rabitq.get_nested_lut7_integer_distance_computer(
                                  codes.data(),
                                  center.data(),
                                  false,
                                  nested_lut7.data())
                        : full_code_mode == RABITQ_FULL_CODE_NESTED_LUT4
                        ? rabitq.get_nested_lut4_integer_distance_computer(
                                  codes.data(),
                                  center.data(),
                                  false,
                                  false,
                                  nested_lut4.data())
                        : get_FlatCodesDistanceComputer());
        auto* batch = dynamic_cast<DistanceComputerBatch*>(dc.get());
#pragma omp for
        for (idx_t row = 0; row < n; ++row) {
            dc->set_query(x + row * d);
            const idx_t* row_labels = labels + row * k;
            float* row_distances = distances + row * k;
            idx_t begin = 0;
            if (batch && batch->preferred_batch_size() >= 16) {
                for (; begin + 16 <= k; begin += 16) {
                    int32_t ids[16];
                    bool valid = true;
                    for (int lane = 0; lane < 16; ++lane) {
                        const idx_t id = row_labels[begin + lane];
                        valid &= id >= 0 && id < ntotal &&
                                id <= std::numeric_limits<int32_t>::max();
                        ids[lane] = static_cast<int32_t>(id);
                    }
                    if (valid) {
                        batch->distances_batch_16(ids, row_distances + begin);
                    } else {
                        for (int lane = 0; lane < 16; ++lane) {
                            const idx_t id = row_labels[begin + lane];
                            row_distances[begin + lane] = id >= 0 && id < ntotal
                                    ? (*dc)(id)
                                    : std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
            if (batch && batch->preferred_batch_size() >= 8) {
                for (; begin + 8 <= k; begin += 8) {
                    int32_t ids[8];
                    bool valid = true;
                    for (int lane = 0; lane < 8; ++lane) {
                        const idx_t id = row_labels[begin + lane];
                        valid &= id >= 0 && id < ntotal &&
                                id <= std::numeric_limits<int32_t>::max();
                        ids[lane] = static_cast<int32_t>(id);
                    }
                    if (valid) {
                        batch->distances_batch_8(ids, row_distances + begin);
                    } else {
                        for (int lane = 0; lane < 8; ++lane) {
                            const idx_t id = row_labels[begin + lane];
                            row_distances[begin + lane] = id >= 0 && id < ntotal
                                    ? (*dc)(id)
                                    : std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
            const int tail = static_cast<int>(k - begin);
            if (batch && tail > 0 && tail <= 7 &&
                tail <= batch->max_tail_batch_size()) {
                int32_t ids[7];
                bool valid = true;
                for (int lane = 0; lane < tail; ++lane) {
                    const idx_t id = row_labels[begin + lane];
                    valid &= id >= 0 && id < ntotal &&
                            id <= std::numeric_limits<int32_t>::max();
                    ids[lane] = static_cast<int32_t>(id);
                }
                if (valid) {
                    batch->distances_batch_tail(
                            ids, tail, row_distances + begin);
                    begin = k;
                }
            }
            for (; begin < k; ++begin) {
                const idx_t id = row_labels[begin];
                row_distances[begin] = id >= 0 && id < ntotal
                        ? (*dc)(id)
                        : std::numeric_limits<float>::infinity();
            }
        }
    }
}

void IndexRaBitQ::prepare_progressive_layout() {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0 && codes.size() == 0,
            "progressive encoding must be selected before adding vectors");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2,
            "progressive RaBitQ currently supports only L2");
    code_size = rabitq.progressive_code_size();
    full_code_mode = RABITQ_FULL_CODE_PROGRESSIVE;
    std::vector<uint8_t>().swap(expanded_codes);
}

void IndexRaBitQ::prepare_nested_lut7_layout() {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0 && codes.size() == 0,
            "nested LUT encoding must be selected before adding vectors");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2 && rabitq.nb_bits == 7,
            "nested LUT layout requires seven-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            !is_trained || nested_lut7.size() == 64,
            "select nested LUT7 layout before training");
    code_size = rabitq.nested_lut7_code_size();
    full_code_mode = RABITQ_FULL_CODE_NESTED_LUT7;
    std::vector<uint8_t>().swap(expanded_codes);
}

void IndexRaBitQ::prepare_nested_lut4_layout() {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0 && codes.size() == 0,
            "nested LUT4 encoding must be selected before adding vectors");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2 && rabitq.nb_bits == 4,
            "nested LUT4 layout requires four-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            !is_trained || nested_lut4.size() == 8,
            "select nested LUT4 layout before training");
    code_size = rabitq.nested_lut4_code_size(false);
    full_code_mode = RABITQ_FULL_CODE_NESTED_LUT4;
    std::vector<uint8_t>().swap(expanded_codes);
}

void IndexRaBitQ::prepare_nibble_lut4_layout() {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0 && codes.size() == 0,
            "nibble LUT4 encoding must be selected before adding vectors");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2 && rabitq.nb_bits == 4,
            "nibble LUT4 layout requires four-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            !is_trained || nested_lut4.size() == 8,
            "select nested LUT4 layout before training");
    code_size = rabitq.nested_lut4_code_size(true);
    full_code_mode = RABITQ_FULL_CODE_NIBBLE_LUT4;
    std::vector<uint8_t>().swap(expanded_codes);
}

FlatCodesDistanceComputer* IndexRaBitQ::get_quantized_distance_computer(
        const uint8_t qb_in,
        bool centered_in) const {
    FAISS_THROW_IF_NOT_MSG(
            full_code_mode == RABITQ_FULL_CODE_PACKED,
            "query-quantization overrides are supported only in packed mode; "
            "use get_FlatCodesDistanceComputer() for full-code layouts");
    FlatCodesDistanceComputer* dc =
            rabitq.get_distance_computer(qb_in, center.data(), centered_in);
    dc->code_size = rabitq.code_size;
    dc->codes = codes.data();
    return dc;
}

namespace {

struct Run_search_with_dc_res {
    using T = void;

    uint8_t qb = 0;
    bool centered = false;
    uint8_t nb_bits = 1; // Number of bits per dimension
    bool full_only = false;

    template <class BlockResultHandler>
    void f(BlockResultHandler& res, const IndexRaBitQ* index, const float* xq) {
        size_t ntotal = index->ntotal;
        using SingleResultHandler =
                typename BlockResultHandler::SingleResultHandler;
        const int d = index->d;
        size_t ex_bits = nb_bits - 1;

#pragma omp parallel
        {
            std::unique_ptr<FlatCodesDistanceComputer> dc_base(
                    full_only ? index->get_FlatCodesDistanceComputer()
                              : index->get_quantized_distance_computer(
                                        qb, centered));
            SingleResultHandler resi(res);
#pragma omp for
            for (int64_t q = 0; q < static_cast<int64_t>(res.nq); q++) {
                resi.begin(q);
                dc_base->set_query(xq + d * q);

                if (ex_bits == 0 || full_only) {
                    // 1-bit: Standard single-stage search
                    for (size_t i = 0; i < ntotal; i++) {
                        if (res.is_in_selection(i)) {
                            float dis = (*dc_base)(i);
                            resi.add_result(dis, i);
                        }
                    }
                } else {
                    // Multi-bit: Two-stage search with adaptive filtering
                    // Note: Even with query quantization (qb > 0), ex-bits
                    // distance computation uses the float query to maintain
                    // consistency with encoding-time factor computation. See
                    // RaBitQuantizer.cpp for details.
                    auto* dc = dynamic_cast<RaBitQDistanceComputer*>(
                            dc_base.get());
                    FAISS_THROW_IF_MSG(
                            dc == nullptr,
                            "Failed to cast to RaBitQDistanceComputer for two-stage search");

                    bool is_similarity =
                            is_similarity_metric(index->metric_type);

                    for (size_t i = 0; i < ntotal; i++) {
                        if (res.is_in_selection(i)) {
                            const uint8_t* code =
                                    index->codes.data() + i * index->code_size;

                            float est_distance =
                                    dc->distance_to_code_1bit(code);

                            size_t code_size_base = (index->d + 7) / 8;
                            const rabitq_utils::SignBitFactorsWithError*
                                    base_fac = reinterpret_cast<
                                            const rabitq_utils::
                                                    SignBitFactorsWithError*>(
                                            code + code_size_base);

                            bool should_refine =
                                    rabitq_utils::should_refine_candidate(
                                            est_distance,
                                            base_fac->f_error,
                                            dc->g_error,
                                            resi.threshold,
                                            is_similarity);
                            if (should_refine) {
                                float dist_full =
                                        dc->distance_to_code_full(code);
                                resi.add_result(dist_full, i);
                            }
                        }
                    }
                }

                resi.end();
            }
        }
    }
};

} // namespace

void IndexRaBitQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(is_trained);

    // Extract search parameters
    uint8_t used_qb = qb;
    bool used_centered = centered;
    if (auto params = dynamic_cast<const RaBitQSearchParameters*>(params_in)) {
        FAISS_THROW_IF_NOT_MSG(
                full_code_mode == RABITQ_FULL_CODE_PACKED,
                "RaBitQSearchParameters qb/centered overrides require packed "
                "mode; full-code layouts have fixed query representations");
        used_qb = params->qb;
        used_centered = params->centered;
    }

    const IDSelector* sel = (params_in != nullptr) ? params_in->sel : nullptr;

    // Set up functor with all necessary parameters
    Run_search_with_dc_res r;
    r.qb = used_qb;
    r.centered = used_centered;
    r.nb_bits = rabitq.nb_bits; // Pass multi-bit info to functor
    r.full_only = full_code_mode != RABITQ_FULL_CODE_PACKED;

    // Use Faiss framework for all cases (single-stage and two-stage)
    dispatch_knn_ResultHandler(
            n, distances, labels, k, metric_type, sel, r, this, x);
}

void IndexRaBitQ::range_search(
        idx_t /*n*/,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params_in) const {
    uint8_t used_qb = qb;
    if (auto params = dynamic_cast<const RaBitQSearchParameters*>(params_in)) {
        FAISS_THROW_IF_NOT_MSG(
                full_code_mode == RABITQ_FULL_CODE_PACKED,
                "RaBitQSearchParameters qb/centered overrides require packed "
                "mode; full-code layouts have fixed query representations");
        used_qb = params->qb;
    }

    const IDSelector* sel = (params_in != nullptr) ? params_in->sel : nullptr;
    Run_search_with_dc_res r;
    r.qb = used_qb;
    r.full_only = full_code_mode != RABITQ_FULL_CODE_PACKED;

    dispatch_range_ResultHandler(result, radius, metric_type, sel, r, this, x);
}

} // namespace faiss
