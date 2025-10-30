/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/FastScanDistancePostProcessing.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/utils/utils.h>
#include <algorithm>
#include <cmath>

namespace faiss {

static inline size_t roundup(size_t a, size_t b) {
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
    init_fastscan(static_cast<int>(d), M_fastscan, nbits_fastscan, metric, bbs);

    // Override code_size to include space for factors after bit patterns
    // RaBitQ stores 1 bit per dimension, requiring (d + 7) / 8 bytes
    const size_t bit_pattern_size = (d + 7) / 8;
    code_size = bit_pattern_size + sizeof(FactorsData);

    // Set RaBitQ-specific parameters
    qb = 8;
    center.resize(d, 0.0f);

    // Pre-allocate storage vectors for efficiency
    factors_storage.clear();
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
    init_fastscan(
            static_cast<int>(orig.d),
            M_fastscan,
            nbits_fastscan,
            orig.metric_type,
            bbs);

    // Override code_size to include space for factors after bit patterns
    // RaBitQ stores 1 bit per dimension, requiring (d + 7) / 8 bytes
    const size_t bit_pattern_size = (orig.d + 7) / 8;
    code_size = bit_pattern_size + sizeof(FactorsData);

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
        // Allocate space for factors
        factors_storage.resize(ntotal);

        // Extract factors from original codes for each vector
        const float* centroid_data = center.data();

        // Use the original RaBitQ quantizer to decode and compute factors
        std::vector<float> decoded_vectors(ntotal * orig.d);
        orig.sa_decode(ntotal, orig.codes.data(), decoded_vectors.data());

        for (idx_t i = 0; i < ntotal; i++) {
            FactorsData& fac = factors_storage[i];
            const float* x_row = decoded_vectors.data() + i * orig.d;

            // Use shared utilities for computing factors
            fac = rabitq_utils::compute_vector_factors(
                    x_row, orig.d, centroid_data, orig.metric_type);
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

                // Use RaBitQUtils for consistent bit setting
                if (bit_value) {
                    rabitq_utils::set_bit_fastscan(fs_code, j);
                }
            }
        }

        // Pack the converted codes using pq4_pack_codes with custom stride
        codes.resize(ntotal2 * M2 / 2);
        pq4_pack_codes(
                fastscan_codes.get(),
                ntotal,
                M,
                ntotal2,
                bbs,
                M2,
                codes.get(),
                code_size);
    }
}

void IndexRaBitQFastScan::train(idx_t n, const float* x) {
    // compute a centroid
    std::vector<float> centroid(d, 0);
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

void IndexRaBitQFastScan::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);

    // Handle blocking to avoid excessive allocations
    constexpr idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            if (verbose) {
                printf("IndexRaBitQFastScan::add %zd/%zd\n",
                       size_t(i1),
                       size_t(n));
            }
            add(i1 - i0, x + i0 * d);
        }
        return;
    }
    InterruptCallback::check();

    // Create codes with embedded factors using our compute_codes
    AlignedTable<uint8_t> tmp_codes(n * code_size);
    compute_codes(tmp_codes.get(), n, x);

    // Extract and store factors from embedded codes for handler access
    const size_t bit_pattern_size = (d + 7) / 8;
    factors_storage.resize(ntotal + n);
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = tmp_codes.get() + i * code_size;
        const uint8_t* factors_ptr = code + bit_pattern_size;
        const FactorsData& embedded_factors =
                *reinterpret_cast<const FactorsData*>(factors_ptr);
        factors_storage[ntotal + i] = embedded_factors;
    }

    // Resize main storage (same logic as parent)
    ntotal2 = roundup(ntotal + n, bbs);
    size_t new_size = ntotal2 * M2 / 2; // assume nbits = 4
    size_t old_size = codes.size();
    if (new_size > old_size) {
        codes.resize(new_size);
        memset(codes.get() + old_size, 0, new_size - old_size);
    }

    // Use our custom packing function with correct stride
    pq4_pack_codes_range(
            tmp_codes.get(),
            M, // Number of sub-quantizers (bit patterns only)
            ntotal,
            ntotal + n, // Range to pack
            bbs,
            M2,          // Block parameters
            codes.get(), // Output
            code_size);  // CUSTOM STRIDE: includes factor space

    ntotal += n;
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
    const size_t bit_pattern_size = (d + 7) / 8;

    memset(codes, 0, n * code_size);

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        uint8_t* const code = codes + i * code_size;
        const float* const x_row = x + i * d;

        // Pack bits directly into FastScan format
        for (size_t j = 0; j < d; j++) {
            const float x_val = x_row[j];
            const float centroid_val = centroid_data ? centroid_data[j] : 0.0f;
            const float or_minus_c = x_val - centroid_val;
            const bool xb = (or_minus_c > 0.0f);

            if (xb) {
                rabitq_utils::set_bit_fastscan(code, j);
            }
        }

        // Calculate and append factors after the bit data
        FactorsData factors = rabitq_utils::compute_vector_factors(
                x_row, d, centroid_data, metric_type);

        // Append factors at the end of the code
        uint8_t* factors_ptr = code + bit_pattern_size;
        *reinterpret_cast<FactorsData*>(factors_ptr) = factors;
    }
}

void IndexRaBitQFastScan::compute_float_LUT(
        float* lut,
        idx_t n,
        const float* x,
        const FastScanDistancePostProcessing& context) const {
    FAISS_THROW_IF_NOT(is_trained);

    // Pre-allocate working buffers to avoid repeated allocations
    std::vector<float> rotated_q(d);
    std::vector<uint8_t> rotated_qq(d);

    // Compute lookup tables for FastScan SIMD operations
    // For each query vector, computes distance contributions for all
    // possible 4-bit codes per sub-quantizer. Also computes and stores
    // query factors for distance reconstruction.
    for (idx_t i = 0; i < n; i++) {
        const float* query = x + i * d;

        // Compute query factors and store in array if available
        rabitq_utils::QueryFactorsData query_factors_data =
                rabitq_utils::compute_query_factors(
                        query,
                        d,
                        center.data(),
                        qb,
                        centered,
                        metric_type,
                        rotated_q,
                        rotated_qq);

        // Store query factors in context array if provided
        if (context.query_factors) {
            context.query_factors[i] = query_factors_data;
        }

        // Create lookup table storing distance contributions for all possible
        // 4-bit codes per sub-quantizer for FastScan SIMD operations
        float* query_lut = lut + i * M * 16;

        if (centered) {
            // For centered mode, we use the signed odd integer quantization
            // scheme.
            // Formula:
            // int_dot = ((1 << qb) - 1) * d - 2 * xor_dot_product
            // We precompute the XOR contribution for each
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
                            query_factors_data.c1 * inner_product +
                            query_factors_data.c2 * popcount;
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
    const size_t bit_pattern_size = (d + 7) / 8;

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        // Access code using correct FastScan format
        const uint8_t* code = codes + i * code_size;

        // Extract factors directly from embedded codes
        const uint8_t* factors_ptr = code + bit_pattern_size;
        const FactorsData& fac =
                *reinterpret_cast<const FactorsData*>(factors_ptr);

        for (size_t j = 0; j < d; j++) {
            // Use RaBitQUtils for consistent bit extraction
            bool bit_value = rabitq_utils::extract_bit_fastscan(code, j);
            float bit = bit_value ? 1.0f : 0.0f;

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
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    // Create query factors array on stack - memory managed by caller
    std::vector<rabitq_utils::QueryFactorsData> query_factors_storage(n);

    // Use the faster search_dispatch_implem flow from IndexFastScan
    // Pass the query factors array - factors will be computed during LUT
    // computation
    FastScanDistancePostProcessing context;
    context.query_factors = query_factors_storage.data();
    if (metric_type == METRIC_L2) {
        search_dispatch_implem<true>(n, x, k, distances, labels, context);
    } else {
        search_dispatch_implem<false>(n, x, k, distances, labels, context);
    }
}

// Template implementations for RaBitQHeapHandler
template <class C, bool with_id_map>
RaBitQHeapHandler<C, with_id_map>::RaBitQHeapHandler(
        const IndexRaBitQFastScan* index,
        size_t nq_val,
        size_t k_val,
        float* distances,
        int64_t* labels,
        const IDSelector* sel_in,
        const FastScanDistancePostProcessing& ctx)
        : RHC(nq_val, index->ntotal, sel_in),
          rabitq_index(index),
          heap_distances(distances),
          heap_labels(labels),
          nq(nq_val),
          k(k_val),
          context(ctx) {
    // Initialize heaps for all queries in constructor
    // This allows us to support direct normalizer assignment
#pragma omp parallel for if (nq > 100)
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_heapify<Cfloat>(k, heap_dis, heap_ids);
    }
}

template <class C, bool with_id_map>
void RaBitQHeapHandler<C, with_id_map>::handle(
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

    // Access query factors from query_factors pointer
    rabitq_utils::QueryFactorsData query_factors_data = {};
    if (context.query_factors) {
        query_factors_data = context.query_factors[q];
    }

    // Compute normalizers once per batch
    const float one_a = normalizers ? (1.0f / normalizers[2 * q]) : 1.0f;
    const float bias = normalizers ? normalizers[2 * q + 1] : 0.0f;

    // Compute loop bounds to avoid redundant bounds checking
    const size_t base_db_idx = this->j0 + b * 32;
    const size_t max_vectors = (base_db_idx < rabitq_index->ntotal)
            ? std::min<size_t>(32, rabitq_index->ntotal - base_db_idx)
            : 0;

    // Process distances in batch
    for (size_t i = 0; i < max_vectors; i++) {
        const size_t db_idx = base_db_idx + i;

        // Normalize distance from LUT lookup
        const float normalized_distance = d32tab[i] * one_a + bias;

        // Access factors from storage (populated from embedded codes during
        // add())
        const auto& db_factors = rabitq_index->factors_storage[db_idx];

        float adjusted_distance;

        if (rabitq_index->centered) {
            // For centered mode: normalized_distance contains the raw XOR
            // contribution. Apply the signed odd integer quantization formula:
            // int_dot = ((1 << qb) - 1) * d - 2 * xor_dot_product
            int64_t int_dot = ((1 << rabitq_index->qb) - 1) * rabitq_index->d;
            int_dot -= 2 * static_cast<int64_t>(normalized_distance);

            adjusted_distance = query_factors_data.qr_to_c_L2sqr +
                    db_factors.or_minus_c_l2sqr -
                    2 * db_factors.dp_multiplier * int_dot *
                            query_factors_data.int_dot_scale;
        } else {
            // For non-centered quantization: use traditional formula
            float final_dot = normalized_distance - query_factors_data.c34;
            adjusted_distance = db_factors.or_minus_c_l2sqr +
                    query_factors_data.qr_to_c_L2sqr -
                    2 * db_factors.dp_multiplier * final_dot;
        }

        // Apply inner product correction if needed
        if (query_factors_data.qr_norm_L2sqr != 0.0f) {
            adjusted_distance = -0.5f *
                    (adjusted_distance - query_factors_data.qr_norm_L2sqr);
        }

        // Add to heap if better than current worst
        if (Cfloat::cmp(heap_dis[0], adjusted_distance)) {
            heap_replace_top<Cfloat>(
                    k, heap_dis, heap_ids, adjusted_distance, db_idx);
        }
    }
}

template <class C, bool with_id_map>
void RaBitQHeapHandler<C, with_id_map>::begin(const float* norms) {
    normalizers = norms;
    // Heap initialization is now done in constructor
}

template <class C, bool with_id_map>
void RaBitQHeapHandler<C, with_id_map>::end() {
// Reorder final results
#pragma omp parallel for if (nq > 100)
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_reorder<Cfloat>(k, heap_dis, heap_ids);
    }
}

// Implementation of virtual make_knn_handler method
void* IndexRaBitQFastScan::make_knn_handler(
        bool is_max,
        int /*impl*/,
        idx_t n,
        idx_t k,
        size_t /*ntotal*/,
        float* distances,
        idx_t* labels,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context) const {
    if (is_max) {
        return new RaBitQHeapHandler<CMax<uint16_t, int>, false>(
                this, n, k, distances, labels, sel, context);
    } else {
        return new RaBitQHeapHandler<CMin<uint16_t, int>, false>(
                this, n, k, distances, labels, sel, context);
    }
}

// Explicit template instantiations for the required comparator types
template struct RaBitQHeapHandler<CMin<uint16_t, int>, false>;
template struct RaBitQHeapHandler<CMax<uint16_t, int>, false>;
template struct RaBitQHeapHandler<CMin<uint16_t, int>, true>;
template struct RaBitQHeapHandler<CMax<uint16_t, int>, true>;

} // namespace faiss
