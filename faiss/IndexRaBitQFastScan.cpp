/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexRaBitQFastScan::IndexRaBitQFastScan() = default;

IndexRaBitQFastScan::IndexRaBitQFastScan(idx_t d, MetricType metric, int bbs)
        : rabitq(d, metric) {
    // RaBitQ-specific validation
    FAISS_THROW_IF_NOT_MSG(d > 0, "Dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "RaBitQ FastScan only supports L2 and Inner Product metrics");

    // RaBitQ uses 1 bit per dimension packed into 4-bit FastScan sub-quantizers
    // Each FastScan sub-quantizer handles 4 RaBitQ dimensions
    const size_t M_fastscan = (d + 3) / 4;
    constexpr size_t nbits_fastscan = 4;

    // init_fastscan will validate bbs % 32 == 0 and nbits_fastscan == 4
    init_fastscan(d, M_fastscan, nbits_fastscan, metric, bbs);

    // Set RaBitQ-specific parameters
    qb = 8;
    center.resize(d, 0.0f);

    // Pre-allocate storage vectors for efficiency
    factors_storage.clear();
    query_factors_storage.clear();
}

IndexRaBitQFastScan::IndexRaBitQFastScan(const IndexRaBitQ& orig, int bbs)
        : rabitq(orig.rabitq) {
    // RaBitQ-specific validation
    FAISS_THROW_IF_NOT_MSG(orig.d > 0, "Dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(
            orig.metric_type == METRIC_L2 ||
                    orig.metric_type == METRIC_INNER_PRODUCT,
            "RaBitQ FastScan only supports L2 and Inner Product metrics");

    // RaBitQ uses 1 bit per dimension packed into 4-bit FastScan sub-quantizers
    // Each FastScan sub-quantizer handles 4 RaBitQ dimensions
    const size_t M_fastscan = (orig.d + 3) / 4;
    constexpr size_t nbits_fastscan = 4;

    // Initialize FastScan base with the original index's parameters
    init_fastscan(orig.d, M_fastscan, nbits_fastscan, orig.metric_type, bbs);

    // Copy properties from original index
    ntotal = orig.ntotal;
    ntotal2 = roundup(ntotal, bbs);
    is_trained = orig.is_trained;
    orig_codes = orig.codes.data();
    qb = orig.qb;
    centered = orig.centered;
    center = orig.center;

    // If the original index has data, extract factors and pack codes
    if (ntotal > 0) {
        // Reserve space for factors
        factors_storage.reserve(ntotal);
        factors_storage.resize(ntotal);

        // Extract factors from original codes for each vector
        const float* centroid_data = center.data();
        const bool has_centroid = (centroid_data != nullptr);
        const float inv_d_sqrt = (orig.d == 0)
                ? 1.0f
                : (1.0f / std::sqrt(static_cast<float>(orig.d)));
        constexpr float epsilon = std::numeric_limits<float>::epsilon();
        const bool is_inner_product =
                (orig.metric_type == MetricType::METRIC_INNER_PRODUCT);

        // Use the original RaBitQ quantizer to decode and compute factors
        std::vector<float> decoded_vectors(ntotal * orig.d);
        orig.sa_decode(ntotal, orig.codes.data(), decoded_vectors.data());

        for (idx_t i = 0; i < ntotal; i++) {
            FactorsData& fac = factors_storage[i];
            const float* x_row = decoded_vectors.data() + i * orig.d;

            float norm_L2sqr = 0.0f;
            float or_L2sqr = 0.0f;
            float dp_oO = 0.0f;

            // Compute factors following the same logic as compute_codes
            for (size_t j = 0; j < orig.d; j++) {
                const float x_val = x_row[j];
                const float centroid_val =
                        has_centroid ? centroid_data[j] : 0.0f;
                const float or_minus_c = x_val - centroid_val;

                const float or_minus_c_sq = or_minus_c * or_minus_c;
                norm_L2sqr += or_minus_c_sq;
                or_L2sqr += x_val * x_val;

                const bool xb = (or_minus_c > 0.0f);
                dp_oO += xb ? or_minus_c : -or_minus_c;
            }

            const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
            const float inv_norm_L2 =
                    (norm_L2sqr < epsilon) ? 1.0f : (1.0f / sqrt_norm_L2);
            const float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
            const float inv_dp_oO = (std::abs(normalized_dp) < epsilon)
                    ? 1.0f
                    : (1.0f / normalized_dp);

            fac.or_minus_c_l2sqr =
                    is_inner_product ? (norm_L2sqr - or_L2sqr) : norm_L2sqr;
            fac.dp_multiplier = inv_dp_oO * sqrt_norm_L2;
        }

        // Convert RaBitQ bit format to FastScan 4-bit sub-quantizer format
        // This follows the same pattern as IndexPQFastScan constructor
        AlignedTable<uint8_t> fastscan_codes(ntotal * code_size);
        memset(fastscan_codes.get(), 0, ntotal * code_size);

        // Convert from RaBitQ 1-bit-per-dimension to FastScan
        // 4-bit-per-sub-quantizer
        for (idx_t i = 0; i < ntotal; i++) {
            const uint8_t* orig_code = orig.codes.data() + i * orig.code_size;
            uint8_t* fs_code = fastscan_codes.get() + i * code_size;

            // Convert each dimension's bit (same logic as compute_codes)
            for (size_t j = 0; j < orig.d; j++) {
                // Extract bit from original RaBitQ format
                const size_t orig_byte_idx = j / 8;
                const size_t orig_bit_offset = j % 8;
                const bool bit_value =
                        (orig_code[orig_byte_idx] >> orig_bit_offset) & 1;

                // Pack bit into FastScan format
                if (bit_value) {
                    const size_t m = j / 4;
                    const size_t dim_offset = j % 4;
                    const uint8_t bit_mask =
                            static_cast<uint8_t>(1 << dim_offset);
                    const size_t byte_idx = m / 2;

                    if (m % 2 == 0) {
                        fs_code[byte_idx] |= bit_mask;
                    } else {
                        fs_code[byte_idx] |= (bit_mask << 4);
                    }
                }
            }
        }

        // Pack the converted codes using pq4_pack_codes (same as other FastScan
        // constructors)
        codes.resize(ntotal2 * M2 / 2);
        pq4_pack_codes(
                fastscan_codes.get(), ntotal, M, ntotal2, bbs, M2, codes.get());
    }

    // Clear query factors storage
    query_factors_storage.clear();
}

void IndexRaBitQFastScan::train(idx_t n, const float* x) {
    // compute a centroid
    std::vector<float> centroid(d, 0);
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        for (size_t j = 0; j < d; j++) {
            centroid[j] += x[i * d + j];
        }
    }

    if (n != 0) {
        for (size_t j = 0; j < d; j++) {
            centroid[j] /= (float)n;
        }
    }

    center = std::move(centroid);

    rabitq.train(n, x);
    is_trained = true;
}

void IndexRaBitQFastScan::compute_codes(uint8_t* codes, idx_t n, const float* x)
        const {
    FAISS_ASSERT(codes != nullptr);
    FAISS_ASSERT(x != nullptr);
    FAISS_ASSERT(
            (metric_type == MetricType::METRIC_L2 ||
             metric_type == MetricType::METRIC_INNER_PRODUCT));
    if (n == 0) {
        return;
    }

    // Hoist loop-invariant computations
    const float* centroid_data = center.data();
    const bool has_centroid = (centroid_data != nullptr);
    const float inv_d_sqrt =
            (d == 0) ? 1.0f : (1.0f / std::sqrt(static_cast<float>(d)));
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    const bool is_inner_product =
            (metric_type == MetricType::METRIC_INNER_PRODUCT);

    // Pre-allocate factors_storage to avoid resize overhead
    const idx_t start_idx = ntotal;
    if (factors_storage.size() < start_idx + n) {
        factors_storage.reserve(start_idx + n + 1000);
        factors_storage.resize(start_idx + n);
    }

    memset(codes, 0, n * code_size);

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        float norm_L2sqr = 0.0f;
        float or_L2sqr = 0.0f;
        float dp_oO = 0.0f;

        uint8_t* const code = codes + i * code_size;
        FactorsData& fac = factors_storage[start_idx + i];
        const float* const x_row = x + i * d;

        // Process dimensions and pack bits directly
        for (size_t j = 0; j < d; j++) {
            const float x_val = x_row[j];
            const float centroid_val = has_centroid ? centroid_data[j] : 0.0f;
            const float or_minus_c = x_val - centroid_val;

            // Accumulate norms
            const float or_minus_c_sq = or_minus_c * or_minus_c;
            norm_L2sqr += or_minus_c_sq;
            or_L2sqr += x_val * x_val;

            const bool xb = (or_minus_c > 0.0f);
            dp_oO += xb ? or_minus_c : -or_minus_c;

            // Pack bit directly into FastScan format
            if (xb) {
                const size_t m = j / 4;
                const size_t dim_offset = j % 4;
                const uint8_t bit_mask = static_cast<uint8_t>(1 << dim_offset);
                const size_t byte_idx = m / 2;

                // FastScan packs two 4-bit codes per byte
                if (m % 2 == 0) {
                    code[byte_idx] |= bit_mask;
                } else {
                    code[byte_idx] |= (bit_mask << 4);
                }
            }
        }

        // Compute factors with optimized calculations
        const float sqrt_norm_L2 = std::sqrt(norm_L2sqr);
        const float inv_norm_L2 =
                (norm_L2sqr < epsilon) ? 1.0f : (1.0f / sqrt_norm_L2);

        const float normalized_dp = dp_oO * inv_norm_L2 * inv_d_sqrt;
        const float inv_dp_oO = (std::abs(normalized_dp) < epsilon)
                ? 1.0f
                : (1.0f / normalized_dp);

        fac.or_minus_c_l2sqr =
                is_inner_product ? (norm_L2sqr - or_L2sqr) : norm_L2sqr;
        fac.dp_multiplier = inv_dp_oO * sqrt_norm_L2;
    }
}
namespace {

// Ideal quantizer radii for quantizers of 1..8 bits, optimized to minimize
// L2 reconstruction error.
const float z_max_by_qb[8] = {
        0.79688, // qb = 1.
        1.49375,
        2.05078,
        2.50938,
        2.91250,
        3.26406,
        3.59844,
        3.91016, // qb = 8.
};

} // namespace

void IndexRaBitQFastScan::compute_float_LUT(
        float* lut,
        idx_t n,
        const float* x,
        idx_t query_offset) const {
    FAISS_THROW_IF_NOT(is_trained);

    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

    // Pre-allocate working buffers to avoid repeated allocations
    std::vector<float> rotated_q(d);
    std::vector<uint8_t> rotated_qq(d);

    // Compute lookup tables for FastScan SIMD operations
    // For each query vector, computes distance contributions for all
    // possible 4-bit codes per sub-quantizer. Also computes and stores
    // query factors for distance reconstruction.
    for (idx_t i = 0; i < n; i++) {
        const float* query = x + i * d;
        const idx_t global_query_index = query_offset + i;

        QueryFactorsData& query_factors =
                const_cast<IndexRaBitQFastScan*>(this)
                        ->query_factors_storage[global_query_index];

        if (center.data() != nullptr) {
            query_factors.qr_to_c_L2sqr = fvec_L2sqr(query, center.data(), d);
        } else {
            query_factors.qr_to_c_L2sqr = fvec_norm_L2sqr(query, d);
        }

        // Step 1: Apply centering to query vector (subtract database
        // centroid)
        for (size_t j = 0; j < d; j++) {
            rotated_q[j] =
                    query[j] - ((center.data() == nullptr) ? 0 : center[j]);
        }

        // Step 2: Quantize query vector to qb bits using uniform
        // quantization
        float v_min = std::numeric_limits<float>::max();
        float v_max = std::numeric_limits<float>::lowest();

        // Find min/max for quantization range
        if (centered) {
            float z_max = z_max_by_qb[qb - 1];
            float v_radius = z_max * std::sqrt(query_factors.qr_to_c_L2sqr / d);
            v_min = -v_radius;
            v_max = v_radius;
        } else {
            for (size_t i = 0; i < d; i++) {
                const float v_q = rotated_q[i];
                v_min = std::min(v_min, v_q);
                v_max = std::max(v_max, v_q);
            }
        }

        const uint8_t max_code = (1 << qb) - 1;
        const float delta = (v_max - v_min) / max_code;
        const float inv_delta = 1.0f / delta;

        size_t sum_qq = 0;
        int64_t sum2_signed_odd_int = 0;
        for (int32_t j = 0; j < d; j++) {
            const float v_q = rotated_q[j];
            // a default non-randomized SQ
            const uint8_t v_qq = std::clamp<float>(
                    std::round((v_q - v_min) * inv_delta), 0, max_code);
            rotated_qq[j] = v_qq;
            sum_qq += v_qq;
            if (centered) {
                int64_t signed_odd_int = int64_t(v_qq) * 2 - max_code;
                sum2_signed_odd_int += signed_odd_int * signed_odd_int;
            }
        }

        // Store query factors
        query_factors.c1 = 2 * delta * inv_d_sqrt;
        query_factors.c2 = 2 * v_min * inv_d_sqrt;
        query_factors.c34 = inv_d_sqrt * (delta * sum_qq + d * v_min);
        query_factors.int_dot_scale = std::sqrt(
                query_factors.qr_to_c_L2sqr / (sum2_signed_odd_int * d));

        query_factors.qr_norm_L2sqr = 0.0f;
        if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
            query_factors.qr_norm_L2sqr = fvec_norm_L2sqr(query, d);
        }

        // Step 4: Create lookup table for FastScan
        // Store scaled values using query factors based on quantization
        // mode Database factors will be applied during final reconstruction
        float* query_lut = lut + i * M * 16;

        if (centered) {
            // For centered mode, we use the signed odd integer quantization
            // scheme Formula: int_dot = ((1 << qb) - 1) * d - 2 *
            // xor_dot_product We precompute the XOR contribution for each
            // sub-quantizer

            const float max_code_value = (1 << qb) - 1;

            for (size_t m = 0; m < M; m++) {
                const size_t dim_start = m * 4;

                for (int code_val = 0; code_val < 16; code_val++) {
                    float xor_contribution = 0.0f;

                    // Process 4 bits per sub-quantizer
                    for (size_t dim_offset = 0; dim_offset < 4; dim_offset++) {
                        const size_t dim_idx = dim_start + dim_offset;

                        if (dim_idx < d) {
                            const bool db_bit = (code_val >> dim_offset) & 1;
                            const float query_value = rotated_qq[dim_idx];

                            // XOR contribution:
                            // If db_bit == 0: XOR result = query_value
                            // If db_bit == 1: XOR result = (2^qb - 1) -
                            // query_value
                            xor_contribution += db_bit
                                    ? (max_code_value - query_value)
                                    : query_value;
                        }
                    }

                    // Store the XOR contribution (will be scaled by -2 *
                    // int_dot_scale during distance computation)
                    query_lut[m * 16 + code_val] = xor_contribution;
                }
            }

        } else {
            // For non-centered quantization, use traditional AND dot
            // product Compute lookup table entries by processing popcount
            // and inner product together
            for (size_t m = 0; m < M; m++) {
                const size_t dim_start = m * 4;

                for (int code_val = 0; code_val < 16; code_val++) {
                    float inner_product = 0.0f;
                    int popcount = 0;

                    // Process 4 bits per sub-quantizer
                    for (size_t dim_offset = 0; dim_offset < 4; dim_offset++) {
                        const size_t dim_idx = dim_start + dim_offset;

                        if (dim_idx < d && ((code_val >> dim_offset) & 1)) {
                            inner_product += rotated_qq[dim_idx];
                            popcount++;
                        }
                    }

                    // Store pre-computed distance contribution
                    query_lut[m * 16 + code_val] =
                            query_factors.c1 * inner_product +
                            query_factors.c2 * popcount;
                }
            }
        }
    }
}

void IndexRaBitQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    const float* centroid_in =
            (center.data() == nullptr) ? nullptr : center.data();
    const uint8_t* codes = bytes;
    FAISS_ASSERT(codes != nullptr);
    FAISS_ASSERT(x != nullptr);

    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        // Access code using correct FastScan format
        const uint8_t* code = codes + i * code_size;

        // Access factors from separate storage
        const FactorsData& fac = factors_storage[i];

        for (size_t j = 0; j < d; j++) {
            // Extract j-th bit using FastScan format (matches
            // compute_codes)
            const size_t m = j / 4; // Sub-quantizer index
            const size_t dim_offset =
                    j % 4; // Bit position within sub-quantizer
            const size_t byte_idx =
                    m / 2; // Byte index (2 sub-quantizers per byte)
            const uint8_t bit_mask = static_cast<uint8_t>(1 << dim_offset);

            float bit;
            if (m % 2 == 0) {
                // Lower 4 bits of byte
                bit = ((code[byte_idx] & bit_mask) != 0) ? 1.0f : 0.0f;
            } else {
                // Upper 4 bits of byte (shifted)
                bit = ((code[byte_idx] & (bit_mask << 4)) != 0) ? 1.0f : 0.0f;
            }

            // Compute the output using RaBitQ reconstruction formula
            x[i * d + j] = (bit - 0.5f) * fac.dp_multiplier * 2 * inv_d_sqrt +
                    ((centroid_in == nullptr) ? 0 : centroid_in[j]);
        }
    }
}

void IndexRaBitQFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    uint8_t used_qb = this->qb;
    bool used_centered = this->centered;

    if (auto rabitq_params =
                dynamic_cast<const RaBitQSearchParameters*>(params)) {
        used_qb = rabitq_params->qb;
        used_centered = rabitq_params->centered;
    } else if (params != nullptr) {
        FAISS_THROW_IF_NOT_MSG(
                false, "search params not supported for this index");
    }

    FAISS_THROW_IF_NOT(k > 0);

    // Temporarily update qb and centered for this search
    const_cast<IndexRaBitQFastScan*>(this)->qb = used_qb;
    const_cast<IndexRaBitQFastScan*>(this)->centered = used_centered;

    // Pre-allocate query_factors_storage for efficient access during search
    // Size to accommodate potential parallel execution with query offsets
    query_factors_storage.clear();
    query_factors_storage.resize(n);

    // Use the faster search_dispatch_implem flow from IndexFastScan
    if (metric_type == METRIC_L2) {
        search_dispatch_implem<true>(n, x, k, distances, labels, nullptr);
    } else {
        search_dispatch_implem<false>(n, x, k, distances, labels, nullptr);
    }
}

// Template implementations for RaBitQHeapHandler
template <class C>
RaBitQHeapHandler<C>::RaBitQHeapHandler(
        const IndexRaBitQFastScan* index,
        size_t nq_val,
        size_t k_val,
        float* distances,
        int64_t* labels,
        size_t global_offset)
        : rabitq_index(index),
          heap_distances(distances),
          heap_labels(labels),
          nq(nq_val),
          k(k_val),
          global_query_offset(global_offset) {}

template <class C>
void RaBitQHeapHandler<C>::handle(
        size_t q,
        size_t b,
        simd16uint16 d0,
        simd16uint16 d1) {
    ALIGNED(32) uint16_t d32tab[32];
    d0.store(d32tab);
    d1.store(d32tab + 16);

    // Get heap pointers and query factors (computed once per batch)
    float* const heap_dis = heap_distances + q * k;
    int64_t* const heap_ids = heap_labels + q * k;
    const auto& query_factors =
            rabitq_index->query_factors_storage[q + global_query_offset];

    // Compute normalizers once per batch
    const float one_a = normalizers ? (1.0f / normalizers[2 * q]) : 1.0f;
    const float bias = normalizers ? normalizers[2 * q + 1] : 0.0f;

    // Compute loop bounds to avoid redundant bounds checking
    const size_t base_db_idx = database_offset + b * 32;
    const size_t max_vectors = (base_db_idx < rabitq_index->ntotal)
            ? std::min(size_t(32), rabitq_index->ntotal - base_db_idx)
            : 0;

    // Process distances in batch
    for (size_t i = 0; i < max_vectors; i++) {
        const size_t db_idx = base_db_idx + i;

        // Normalize distance from LUT lookup
        const float normalized_distance = d32tab[i] * one_a + bias;

        // Apply RaBitQ distance formula based on quantization mode
        const auto& db_factors = rabitq_index->factors_storage[db_idx];
        float adjusted_distance;

        if (rabitq_index->centered) {
            // For centered mode: normalized_distance contains the raw XOR
            // contribution Apply the signed odd integer quantization
            // formula: int_dot = ((1 << qb) - 1) * d - 2 * xor_dot_product
            int64_t int_dot = ((1 << rabitq_index->qb) - 1) * rabitq_index->d;
            int_dot -= 2 * static_cast<int64_t>(normalized_distance);

            adjusted_distance = query_factors.qr_to_c_L2sqr +
                    db_factors.or_minus_c_l2sqr -
                    2 * db_factors.dp_multiplier * int_dot *
                            query_factors.int_dot_scale;
        } else {
            // For non-centered quantization: use traditional formula
            float final_dot = normalized_distance - query_factors.c34;
            adjusted_distance = db_factors.or_minus_c_l2sqr +
                    query_factors.qr_to_c_L2sqr -
                    2 * db_factors.dp_multiplier * final_dot;
        }

        // Apply inner product correction if needed
        if (query_factors.qr_norm_L2sqr != 0.0f) {
            adjusted_distance =
                    -0.5f * (adjusted_distance - query_factors.qr_norm_L2sqr);
        }

        // Add to heap if better than current worst
        if (Cfloat::cmp(heap_dis[0], adjusted_distance)) {
            heap_replace_top<Cfloat>(
                    k, heap_dis, heap_ids, adjusted_distance, db_idx);
        }
    }
}

template <class C>
void RaBitQHeapHandler<C>::set_block_origin(size_t, size_t j0) {
    database_offset = j0;
}

template <class C>
void RaBitQHeapHandler<C>::begin(const float* norms) {
    normalizers = norms;

// Initialize heaps for all queries
#pragma omp parallel for
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_heapify<Cfloat>(k, heap_dis, heap_ids);
    }
}

template <class C>
void RaBitQHeapHandler<C>::end() {
// Reorder final results
#pragma omp parallel for
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_reorder<Cfloat>(k, heap_dis, heap_ids);
    }
}

// Explicit template instantiations for the required comparator types
template struct RaBitQHeapHandler<CMin<uint16_t, int>>;
template struct RaBitQHeapHandler<CMax<uint16_t, int>>;

} // namespace faiss
