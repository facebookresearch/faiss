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

    if (rabitq.nb_bits == 7 && n > 0) {
        if (nested_lut7_exact_encoding) {
            nested_lut7.resize(64);
            for (size_t i = 0; i < nested_lut7.size(); ++i) {
                nested_lut7[i] = static_cast<uint8_t>(i);
            }
        } else {
            nested_lut7 = train_nested_lut7(n, d, x, center.data());
        }
    } else {
        nested_lut7.clear();
    }
    if (rabitq.nb_bits == 4 && n > 0) {
        nested_lut4 = train_nested_lut4(n, d, x, center.data());
    } else {
        nested_lut4.clear();
    }

    //
    rabitq.train(n, x);
    is_trained = true;
}

void IndexRaBitQ::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            split4_sign_codes.empty(),
            "cannot add after split RQ4 finalization");
    IndexFlatCodes::add(n, x);
    nested_adaptive_error_norms.clear();
    nested_adaptive_navigation = false;
    if (mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::add_sa_codes(idx_t n, const uint8_t* x, const idx_t* xids) {
    IndexFlatCodes::add_sa_codes(n, x, xids);
    nested_adaptive_error_norms.clear();
    nested_adaptive_navigation = false;
    if (mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::reset() {
    IndexFlatCodes::reset();
    expanded_codes.clear();
    split4_sign_codes.clear();
    split4_tail_codes.clear();
    nested_adaptive_error_norms.clear();
    nested_adaptive_navigation = false;
}

size_t IndexRaBitQ::remove_ids(const IDSelector& sel) {
    const size_t removed = IndexFlatCodes::remove_ids(sel);
    if (removed > 0 && mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
    if (removed > 0) {
        nested_adaptive_error_norms.clear();
        nested_adaptive_navigation = false;
    }
    return removed;
}

void IndexRaBitQ::merge_from(Index& other_index, idx_t add_id) {
    IndexFlatCodes::merge_from(other_index, add_id);
    nested_adaptive_error_norms.clear();
    nested_adaptive_navigation = false;
    if (mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::permute_entries(const idx_t* perm) {
    IndexFlatCodes::permute_entries(perm);
    nested_adaptive_error_norms.clear();
    nested_adaptive_navigation = false;
    if (mode_uses_expanded_cache(full_code_mode)) {
        rebuild_expanded_codes();
    }
}

void IndexRaBitQ::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    if (full_code_mode == RABITQ_FULL_CODE_NESTED_LUT7) {
        rabitq.compute_nested_lut7_codes_core(
                x,
                bytes,
                n,
                center.data(),
                nested_lut7.data(),
                nested_lut7_mid_navigation,
                nested_lut7_exact_encoding);
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
    const size_t sign_stride = (static_cast<size_t>(d) + 7) / 8 +
            sizeof(rabitq_utils::SignBitFactors);
    const size_t tail_stride =
            3 * ((static_cast<size_t>(d) + 7) / 8) + sizeof(ExtraBitsFactors);
    if (full_code_mode == RABITQ_SPLIT4_NAVIGATION) {
        FAISS_THROW_IF_NOT_MSG(
                split4_sign_codes.size() ==
                        static_cast<size_t>(ntotal) * sign_stride,
                "split RQ4 sign plane is incomplete");
        FlatCodesDistanceComputer* dc = rabitq.get_distance_computer(
                qb, center.data(), centered, false);
        dc->code_size = sign_stride;
        dc->codes = split4_sign_codes.data();
        return dc;
    }
    if (full_code_mode == RABITQ_SPLIT4_ADC) {
        FAISS_THROW_IF_NOT_MSG(
                split4_sign_codes.size() ==
                                static_cast<size_t>(ntotal) * sign_stride &&
                        split4_tail_codes.size() ==
                                static_cast<size_t>(ntotal) * tail_stride,
                "split RQ4 planes are incomplete");
        return rabitq.get_split4_integer_distance_computer(
                split4_sign_codes.data(),
                sign_stride,
                split4_tail_codes.data(),
                tail_stride,
                center.data());
    }
    if (full_code_mode == RABITQ_NAVIGATION_1BIT_PACKED) {
        FlatCodesDistanceComputer* dc = rabitq.get_distance_computer(
                qb, center.data(), centered, false);
        dc->code_size = rabitq.code_size;
        dc->codes = codes.data();
        return dc;
    }
    if (full_code_mode == RABITQ_FULL_CODE_PROGRESSIVE) {
        return rabitq.get_progressive_integer_distance_computer(
                codes.data(), center.data(), true);
    }
    if (full_code_mode == RABITQ_FULL_CODE_NESTED_LUT7) {
        FAISS_THROW_IF_NOT_MSG(
                !nested_adaptive_navigation ||
                        nested_adaptive_error_norms.size() ==
                                static_cast<size_t>(ntotal),
                "adaptive nested sidecar is stale");
        return rabitq.get_nested_lut7_integer_distance_computer(
                codes.data(),
                center.data(),
                !nested_adaptive_navigation && !nested_lut7_full_navigation &&
                        !nested_lut7_mid_navigation,
                nested_lut7_mid_navigation,
                nested_lut7.data(),
                nested_adaptive_navigation ? nested_adaptive_error_norms.data()
                                           : nullptr,
                nested_adaptive_sigma);
    }
    if (full_code_mode == RABITQ_FULL_CODE_NESTED_LUT4) {
        return rabitq.get_nested_lut4_integer_distance_computer(
                codes.data(), center.data(), true, false, nested_lut4.data());
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
    if (!split4_sign_codes.empty()) {
        FAISS_THROW_IF_NOT_MSG(
                mode == RABITQ_SPLIT4_NAVIGATION || mode == RABITQ_SPLIT4_ADC,
                "cannot reinterpret finalized split RQ4 planes as legacy rows");
    }
    FAISS_THROW_IF_NOT_MSG(
            mode == RABITQ_FULL_CODE_PACKED ||
                    mode == RABITQ_FULL_CODE_EXPANDED ||
                    mode == RABITQ_FULL_CODE_INT8 ||
                    mode == RABITQ_FULL_CODE_INT8_PACKED ||
                    mode == RABITQ_NAVIGATION_1BIT_PACKED ||
                    mode == RABITQ_SPLIT4_NAVIGATION ||
                    mode == RABITQ_SPLIT4_ADC,
            "invalid RaBitQ full-code mode");
    if (mode == RABITQ_SPLIT4_NAVIGATION || mode == RABITQ_SPLIT4_ADC) {
        FAISS_THROW_IF_NOT_MSG(
                rabitq.nb_bits == 4 && !split4_sign_codes.empty() &&
                        !split4_tail_codes.empty(),
                "split RQ4 mode requires prepare_split4_layout");
    }
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

void IndexRaBitQ::prepare_split4_layout() {
    FAISS_THROW_IF_NOT_MSG(
            rabitq.nb_bits == 4 && metric_type == METRIC_L2,
            "split layout currently requires four-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            !mode_uses_progressive_rows(full_code_mode) &&
                    codes.size() == static_cast<size_t>(ntotal) * code_size,
            "split layout requires authoritative legacy RQ4 rows");

    const size_t sign_bytes = (static_cast<size_t>(d) + 7) / 8;
    const size_t extra_bytes = 3 * sign_bytes;
    const size_t sign_stride = sign_bytes + sizeof(SignBitFactors);
    const size_t tail_stride = extra_bytes + sizeof(ExtraBitsFactors);
    const size_t legacy_extra_offset =
            sign_bytes + sizeof(SignBitFactorsWithError);

    split4_sign_codes.resize(static_cast<size_t>(ntotal) * sign_stride);
    split4_tail_codes.resize(static_cast<size_t>(ntotal) * tail_stride);
#pragma omp parallel for if (ntotal > 1000)
    for (int64_t row = 0; row < ntotal; ++row) {
        const uint8_t* source = codes.data() + size_t(row) * code_size;
        uint8_t* signs = split4_sign_codes.data() + size_t(row) * sign_stride;
        uint8_t* tail = split4_tail_codes.data() + size_t(row) * tail_stride;
        memcpy(signs, source, sign_bytes);
        memcpy(signs + sign_bytes, source + sign_bytes, sizeof(SignBitFactors));
        memset(tail, 0, extra_bytes);
        const uint8_t* legacy_extra = source + legacy_extra_offset;
        for (size_t j = 0; j < static_cast<size_t>(d); ++j) {
            const uint32_t magnitude =
                    rabitq_utils::extract_code_inline(legacy_extra, j, 3);
            for (size_t bit = 0; bit < 3; ++bit) {
                if ((magnitude >> bit) & 1) {
                    rabitq_utils::set_bit_standard(tail + bit * sign_bytes, j);
                }
            }
        }
        ExtraBitsFactors factors;
        memcpy(&factors,
               source + legacy_extra_offset +
                       (3 * static_cast<size_t>(d) + 7) / 8,
               sizeof(factors));
        memcpy(tail + extra_bytes, &factors, sizeof(factors));
    }
    codes = MaybeOwnedVector<uint8_t>();
    std::vector<uint8_t>().swap(expanded_codes);
    full_code_mode = RABITQ_SPLIT4_ADC;
}

void IndexRaBitQ::compute_distance_subset(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
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
            if (batch && tail > 0 && tail <= batch->max_tail_batch_size()) {
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

void IndexRaBitQ::finalize_progressive_layout() {
    FAISS_THROW_MSG(
            "residual progressive codes require re-encoding the original "
            "vectors; call prepare_progressive_layout before add");
}

void IndexRaBitQ::prepare_progressive_layout() {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0 && codes.size() == 0,
            "progressive encoding must be selected before adding vectors");
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "train the RaBitQ storage before selecting its layout");
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
            is_trained, "train the RaBitQ storage before selecting its layout");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2 && rabitq.nb_bits == 7,
            "nested LUT layout requires seven-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            nested_lut7.size() == 64, "nested LUT codebook was not trained");
    code_size = rabitq.nested_lut7_code_size();
    full_code_mode = RABITQ_FULL_CODE_NESTED_LUT7;
    std::vector<uint8_t>().swap(expanded_codes);
}

void IndexRaBitQ::prepare_nested_lut4_layout() {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0 && codes.size() == 0,
            "nested LUT4 encoding must be selected before adding vectors");
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "train the RaBitQ storage before selecting its layout");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2 && rabitq.nb_bits == 4,
            "nested LUT4 layout requires four-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            nested_lut4.size() == 8, "nested LUT4 codebook was not trained");
    code_size = rabitq.nested_lut4_code_size(false);
    full_code_mode = RABITQ_FULL_CODE_NESTED_LUT4;
    std::vector<uint8_t>().swap(expanded_codes);
}

void IndexRaBitQ::prepare_nibble_lut4_layout() {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0 && codes.size() == 0,
            "nibble LUT4 encoding must be selected before adding vectors");
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "train the RaBitQ storage before selecting its layout");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2 && rabitq.nb_bits == 4,
            "nibble LUT4 layout requires four-bit L2 RaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            nested_lut4.size() == 8, "nested LUT4 codebook was not trained");
    code_size = rabitq.nested_lut4_code_size(true);
    full_code_mode = RABITQ_FULL_CODE_NIBBLE_LUT4;
    std::vector<uint8_t>().swap(expanded_codes);
}

void IndexRaBitQ::prepare_nested_adaptive_navigation(float sigma) {
    FAISS_THROW_IF_NOT_MSG(
            full_code_mode == RABITQ_FULL_CODE_NESTED_LUT7,
            "adaptive navigation requires nested LUT7 storage");
    FAISS_THROW_IF_NOT_MSG(
            sigma > 0.0f && std::isfinite(sigma),
            "adaptive navigation sigma must be finite and positive");
    FAISS_THROW_IF_NOT_MSG(
            codes.size() == static_cast<size_t>(ntotal) * code_size,
            "nested LUT7 rows are incomplete");

    const size_t dimension = static_cast<size_t>(d);
    const size_t prefix_bytes = (2 * dimension + 7) / 8;
    const size_t factors_offset = prefix_bytes;
    const size_t low4_offset =
            prefix_bytes + sizeof(rabitq_utils::ProgressiveBitsFactors);
    const size_t low4_bytes = (4 * dimension + 7) / 8;
    const size_t high1_offset = low4_offset + low4_bytes;
    nested_adaptive_error_norms.resize(static_cast<size_t>(ntotal));

#pragma omp parallel for if (ntotal > 1000)
    for (int64_t row = 0; row < ntotal; ++row) {
        const uint8_t* code =
                codes.data() + static_cast<size_t>(row) * code_size;
        rabitq_utils::ProgressiveBitsFactors factors;
        memcpy(&factors, code + factors_offset, sizeof(factors));
        const float prefix_scale = -0.5f * factors.f_rescale_prefix;
        const float full_scale = -0.5f * factors.f_rescale_full;
        double error_norm_sqr = 0.0;
        for (size_t j = 0; j < dimension; ++j) {
            // All three nested planes are naturally aligned. Decode them
            // directly here: the generic helpers walk individual bits and
            // made this one-time sidecar build needlessly expensive at 1M
            // vectors.
            const int prefix = (code[j >> 2] >> (2 * (j & 3))) & 3;
            const int positive = prefix >> 1;
            const int coarse = (prefix & 1) ^ (positive ? 0 : 1);
            const int local_low =
                    (code[low4_offset + (j >> 1)] >> (4 * (j & 1))) & 15;
            const int local_high =
                    (code[high1_offset + (j >> 3)] >> (j & 7)) & 1;
            const int local = local_low | (local_high << 4);
            const int magnitude = nested_lut7[coarse * 32 + local];
            const int prefix_level = positive ? coarse : -1 - coarse;
            const int full_level = positive ? magnitude : -1 - magnitude;
            const double difference = double(full_scale) * full_level -
                    double(prefix_scale) * prefix_level;
            error_norm_sqr += difference * difference;
        }
        nested_adaptive_error_norms[static_cast<size_t>(row)] =
                static_cast<float>(std::sqrt(error_norm_sqr));
    }
    nested_adaptive_sigma = sigma;
    nested_adaptive_navigation = true;
}

void IndexRaBitQ::derive_2bit_prefix(IndexRaBitQ& destination) const {
    FAISS_THROW_IF_NOT_MSG(
            rabitq.nb_bits >= 3 && rabitq.nb_bits <= 8,
            "prefix source requires 3..8 total bits");
    FAISS_THROW_IF_NOT_MSG(
            destination.d == d && destination.rabitq.nb_bits == 2 &&
                    destination.metric_type == metric_type,
            "prefix destination must be a matching 2-bit IndexRaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            destination.ntotal == 0, "prefix destination must be empty");

    const size_t sign_bytes = (static_cast<size_t>(d) + 7) / 8;
    const size_t source_ex_bits = rabitq.nb_bits - 1;
    const size_t source_extra_offset =
            sign_bytes + sizeof(SignBitFactorsWithError);
    const size_t source_extra_bytes =
            (static_cast<size_t>(d) * source_ex_bits + 7) / 8;
    const size_t destination_extra_offset =
            sign_bytes + sizeof(SignBitFactorsWithError);
    const size_t destination_extra_bytes = (static_cast<size_t>(d) + 7) / 8;
    const float prefix_scale =
            static_cast<float>(size_t(1) << (source_ex_bits - 1));

    destination.center = center;
    destination.is_trained = is_trained;
    destination.ntotal = ntotal;
    destination.codes.resize(
            static_cast<size_t>(ntotal) * destination.code_size);
#pragma omp parallel for if (ntotal > 1000)
    for (int64_t row = 0; row < ntotal; ++row) {
        const uint8_t* source = codes.data() + size_t(row) * code_size;
        uint8_t* output =
                destination.codes.data() + size_t(row) * destination.code_size;
        memset(output, 0, destination.code_size);
        memcpy(output, source, sign_bytes + sizeof(SignBitFactorsWithError));
        const uint8_t* source_extra = source + source_extra_offset;
        uint8_t* output_extra = output + destination_extra_offset;
        for (size_t j = 0; j < static_cast<size_t>(d); ++j) {
            const uint32_t low = rabitq_utils::extract_code_inline(
                    source_extra, j, source_ex_bits);
            if ((low >> (source_ex_bits - 1)) & 1) {
                rabitq_utils::set_bit_standard(output_extra, j);
            }
        }
        ExtraBitsFactors factors;
        memcpy(&factors,
               source + source_extra_offset + source_extra_bytes,
               sizeof(factors));
        factors.f_rescale_ex *= prefix_scale;
        memcpy(output + destination_extra_offset + destination_extra_bytes,
               &factors,
               sizeof(factors));
    }
    destination.full_code_mode = RABITQ_FULL_CODE_INT8_PACKED;
    destination.expanded_codes.clear();
}

void IndexRaBitQ::derive_1bit_prefix(IndexRaBitQ& destination) const {
    FAISS_THROW_IF_NOT_MSG(
            rabitq.nb_bits >= 2 && rabitq.nb_bits <= 8,
            "sign prefix source requires 2..8 total bits");
    FAISS_THROW_IF_NOT_MSG(
            !mode_uses_progressive_rows(full_code_mode),
            "derive the sign prefix before progressive layout encoding");
    FAISS_THROW_IF_NOT_MSG(
            destination.d == d && destination.rabitq.nb_bits == 1 &&
                    destination.metric_type == metric_type,
            "prefix destination must be a matching 1-bit IndexRaBitQ");
    FAISS_THROW_IF_NOT_MSG(
            destination.ntotal == 0, "prefix destination must be empty");

    const size_t sign_bytes = (static_cast<size_t>(d) + 7) / 8;
    destination.center = center;
    destination.is_trained = is_trained;
    destination.ntotal = ntotal;
    destination.codes.resize(
            static_cast<size_t>(ntotal) * destination.code_size);
#pragma omp parallel for if (ntotal > 1000)
    for (int64_t row = 0; row < ntotal; ++row) {
        const uint8_t* source = codes.data() + size_t(row) * code_size;
        uint8_t* output =
                destination.codes.data() + size_t(row) * destination.code_size;
        memcpy(output, source, sign_bytes);
        memcpy(output + sign_bytes,
               source + sign_bytes,
               sizeof(rabitq_utils::SignBitFactors));
    }
    destination.qb = qb;
}

FlatCodesDistanceComputer* IndexRaBitQ::get_quantized_distance_computer(
        const uint8_t qb_in,
        bool centered_in) const {
    if (full_code_mode != RABITQ_FULL_CODE_PACKED) {
        return get_FlatCodesDistanceComputer();
    }
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
                    index->get_quantized_distance_computer(qb, centered));
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
        used_qb = params->qb;
    }

    const IDSelector* sel = (params_in != nullptr) ? params_in->sel : nullptr;
    Run_search_with_dc_res r;
    r.qb = used_qb;
    r.centered = centered;
    r.nb_bits = rabitq.nb_bits;
    r.full_only = full_code_mode != RABITQ_FULL_CODE_PACKED;

    dispatch_range_ResultHandler(result, radius, metric_type, sel, r, this, x);
}

} // namespace faiss
