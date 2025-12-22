/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/FastScanDistancePostProcessing.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizerMultiBit.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/utils/utils.h>
#include <algorithm>
#include <cmath>

namespace faiss {

static inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

size_t IndexRaBitQFastScan::compute_per_vector_storage_size() const {
    const size_t ex_bits = rabitq.nb_bits - 1;

    if (ex_bits == 0) {
        // 1-bit: only SignBitFactors
        return sizeof(rabitq_utils::SignBitFactors);
    } else {
        // Multi-bit: SignBitFactorsWithError + ExtraBitsFactors +
        // mag-codes
        return sizeof(SignBitFactorsWithError) + sizeof(ExtraBitsFactors) +
                (d * ex_bits + 7) / 8;
    }
}

IndexRaBitQFastScan::IndexRaBitQFastScan() = default;

IndexRaBitQFastScan::IndexRaBitQFastScan(
        idx_t d,
        MetricType metric,
        int bbs,
        uint8_t nb_bits)
        : rabitq(d, metric, nb_bits) {
    // RaBitQ-specific validation
    FAISS_THROW_IF_NOT_MSG(d > 0, "Dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "RaBitQ FastScan only supports L2 and Inner Product metrics");
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 1 && nb_bits <= 9, "nb_bits must be between 1 and 9");

    // RaBitQ uses 1 bit per dimension packed into 4-bit FastScan sub-quantizers
    // Each FastScan sub-quantizer handles 4 RaBitQ dimensions
    const size_t M_fastscan = (d + 3) / 4;
    constexpr size_t nbits_fastscan = 4;

    // init_fastscan will validate bbs % 32 == 0 and nbits_fastscan == 4
    init_fastscan(static_cast<int>(d), M_fastscan, nbits_fastscan, metric, bbs);

    // Compute code_size directly using RaBitQuantizer
    code_size = rabitq.compute_code_size(d, nb_bits);

    // Set RaBitQ-specific parameters
    qb = 8;
    center.resize(d, 0.0f);

    // Initialize empty flat storage
    flat_storage.clear();
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

    code_size = rabitq.compute_code_size(d, rabitq.nb_bits);

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
        // Compute per-vector storage size for flat storage
        const size_t storage_size = compute_per_vector_storage_size();

        // Allocate flat storage
        flat_storage.resize(ntotal * storage_size);

        // Copy factors directly from original codes
        const size_t bit_pattern_size = (d + 7) / 8;
        for (idx_t i = 0; i < ntotal; i++) {
            const uint8_t* orig_code = orig.codes.data() + i * orig.code_size;
            const uint8_t* source_factors_ptr = orig_code + bit_pattern_size;
            uint8_t* storage = flat_storage.data() + i * storage_size;
            memcpy(storage, source_factors_ptr, storage_size);
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

    const size_t storage_size = compute_per_vector_storage_size();
    flat_storage.resize((ntotal + n) * storage_size);

    // Populate flat storage (no sign bits copying needed!)
    const size_t bit_pattern_size = (d + 7) / 8;
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = tmp_codes.get() + i * code_size;
        const idx_t vec_idx = ntotal + i;

        // Copy factors data directly to flat storage (no reordering needed)
        const uint8_t* source_factors_ptr = code + bit_pattern_size;
        uint8_t* storage = flat_storage.data() + vec_idx * storage_size;
        memcpy(storage, source_factors_ptr, storage_size);
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
    const size_t ex_bits = rabitq.nb_bits - 1;
    const size_t ex_code_size = (d * ex_bits + 7) / 8;

    memset(codes, 0, n * code_size);

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        uint8_t* const code = codes + i * code_size;
        const float* const x_row = x + i * d;

        // Compute residual once, reuse for both sign bits and ex-bits
        std::vector<float> residual(d);
        for (size_t j = 0; j < d; j++) {
            const float centroid_val = centroid_data ? centroid_data[j] : 0.0f;
            residual[j] = x_row[j] - centroid_val;
        }

        // Pack sign bits directly into FastScan format using precomputed
        // residual
        for (size_t j = 0; j < d; j++) {
            if (residual[j] > 0.0f) {
                rabitq_utils::set_bit_fastscan(code, j);
            }
        }

        SignBitFactorsWithError factors = rabitq_utils::compute_vector_factors(
                x_row, d, centroid_data, metric_type, ex_bits > 0);

        if (ex_bits == 0) {
            // 1-bit: store only SignBitFactors (8 bytes)
            memcpy(code + bit_pattern_size, &factors, sizeof(SignBitFactors));
        } else {
            // Multi-bit: store full SignBitFactorsWithError (12 bytes)
            memcpy(code + bit_pattern_size,
                   &factors,
                   sizeof(SignBitFactorsWithError));

            // Add mag-codes and ExtraBitsFactors using precomputed
            // residual
            uint8_t* ex_code =
                    code + bit_pattern_size + sizeof(SignBitFactorsWithError);
            ExtraBitsFactors ex_factors_temp;

            rabitq_multibit::quantize_ex_bits(
                    residual.data(),
                    d,
                    rabitq.nb_bits,
                    ex_code,
                    ex_factors_temp,
                    metric_type,
                    centroid_data);

            memcpy(ex_code + ex_code_size,
                   &ex_factors_temp,
                   sizeof(ExtraBitsFactors));
        }
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
        if (context.query_factors != nullptr) {
            query_factors_data.rotated_q = rotated_q;
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
        const rabitq_utils::SignBitFactors* fac =
                reinterpret_cast<const rabitq_utils::SignBitFactors*>(
                        factors_ptr);

        for (size_t j = 0; j < d; j++) {
            // Use RaBitQUtils for consistent bit extraction
            bool bit_value = rabitq_utils::extract_bit_fastscan(code, j);
            float bit = bit_value ? 1.0f : 0.0f;

            // Compute the output using RaBitQ reconstruction formula
            x[i * d + j] = (bit - 0.5f) * fac->dp_multiplier * 2 * inv_d_sqrt +
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
        const FastScanDistancePostProcessing& ctx,
        bool multi_bit)
        : RHC(nq_val, index->ntotal, sel_in),
          rabitq_index(index),
          heap_distances(distances),
          heap_labels(labels),
          nq(nq_val),
          k(k_val),
          context(ctx),
          is_multi_bit(multi_bit) {
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
    if (context.query_factors != nullptr) {
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

    // Get storage size once
    const size_t storage_size = rabitq_index->compute_per_vector_storage_size();

    // Stats tracking for multi-bit two-stage search only
    // n_1bit_evaluations: candidates evaluated using 1-bit lower bound
    // n_multibit_evaluations: candidates requiring full multi-bit distance
    size_t local_1bit_evaluations = 0;
    size_t local_multibit_evaluations = 0;

    // Process distances in batch
    for (size_t i = 0; i < max_vectors; i++) {
        const size_t db_idx = base_db_idx + i;

        // Normalize distance from LUT lookup
        const float normalized_distance = d32tab[i] * one_a + bias;

        // Access factors from flat storage
        const uint8_t* base_ptr =
                rabitq_index->flat_storage.data() + db_idx * storage_size;

        if (is_multi_bit) {
            // Track candidates actually considered for two-stage filtering
            local_1bit_evaluations++;

            const SignBitFactorsWithError& full_factors =
                    *reinterpret_cast<const SignBitFactorsWithError*>(base_ptr);

            float dist_1bit = rabitq_utils::compute_1bit_adjusted_distance(
                    normalized_distance,
                    full_factors,
                    query_factors_data,
                    rabitq_index->centered,
                    rabitq_index->qb,
                    rabitq_index->d);

            float lower_bound = compute_lower_bound(dist_1bit, db_idx, q);

            // Adaptive filtering: decide whether to compute full distance
            const bool is_similarity = rabitq_index->metric_type ==
                    MetricType::METRIC_INNER_PRODUCT;
            bool should_refine = is_similarity
                    ? (lower_bound > heap_dis[0])  // IP: keep if better
                    : (lower_bound < heap_dis[0]); // L2: keep if better

            if (should_refine) {
                local_multibit_evaluations++;
                float dist_full = compute_full_multibit_distance(db_idx, q);

                if (Cfloat::cmp(heap_dis[0], dist_full)) {
                    heap_replace_top<Cfloat>(
                            k, heap_dis, heap_ids, dist_full, db_idx);
                }
            }
        } else {
            const rabitq_utils::SignBitFactors& db_factors =
                    *reinterpret_cast<const rabitq_utils::SignBitFactors*>(
                            base_ptr);

            float adjusted_distance =
                    rabitq_utils::compute_1bit_adjusted_distance(
                            normalized_distance,
                            db_factors,
                            query_factors_data,
                            rabitq_index->centered,
                            rabitq_index->qb,
                            rabitq_index->d);

            // Add to heap if better than current worst
            if (Cfloat::cmp(heap_dis[0], adjusted_distance)) {
                heap_replace_top<Cfloat>(
                        k, heap_dis, heap_ids, adjusted_distance, db_idx);
            }
        }
    }

    // Update global stats atomically
#pragma omp atomic
    rabitq_stats.n_1bit_evaluations += local_1bit_evaluations;
#pragma omp atomic
    rabitq_stats.n_multibit_evaluations += local_multibit_evaluations;
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

template <class C, bool with_id_map>
float RaBitQHeapHandler<C, with_id_map>::compute_lower_bound(
        float dist_1bit,
        size_t db_idx,
        size_t q) const {
    // Access f_error directly from SignBitFactorsWithError in flat storage
    const size_t storage_size = rabitq_index->compute_per_vector_storage_size();
    const uint8_t* base_ptr =
            rabitq_index->flat_storage.data() + db_idx * storage_size;
    const SignBitFactorsWithError& db_factors =
            *reinterpret_cast<const SignBitFactorsWithError*>(base_ptr);
    float f_error = db_factors.f_error;

    // Get g_error from query factors (query-dependent error term)
    float g_error = 0.0f;
    if (context.query_factors != nullptr) {
        g_error = context.query_factors[q].g_error;
    }

    // Compute error adjustment: f_error * g_error
    float error_adjustment = f_error * g_error;

    return dist_1bit - error_adjustment;
}

template <class C, bool with_id_map>
float RaBitQHeapHandler<C, with_id_map>::compute_full_multibit_distance(
        size_t db_idx,
        size_t q) const {
    const size_t ex_bits = rabitq_index->rabitq.nb_bits - 1;
    const size_t dim = rabitq_index->d;

    const size_t storage_size = rabitq_index->compute_per_vector_storage_size();
    const uint8_t* base_ptr =
            rabitq_index->flat_storage.data() + db_idx * storage_size;

    const size_t ex_code_size = (dim * ex_bits + 7) / 8;
    const uint8_t* ex_code = base_ptr + sizeof(SignBitFactorsWithError);
    const ExtraBitsFactors& ex_fac = *reinterpret_cast<const ExtraBitsFactors*>(
            base_ptr + sizeof(SignBitFactorsWithError) + ex_code_size);

    // Get query factors reference (avoid copying)
    const rabitq_utils::QueryFactorsData& query_factors =
            context.query_factors[q];

    // Get sign bits from FastScan packed format
    std::vector<uint8_t> unpacked_code(rabitq_index->code_size);
    CodePackerPQ4 packer(rabitq_index->M2, rabitq_index->bbs);
    packer.unpack_1(rabitq_index->codes.get(), db_idx, unpacked_code.data());
    const uint8_t* sign_bits = unpacked_code.data();

    return rabitq_utils::compute_full_multibit_distance(
            sign_bits,
            ex_code,
            ex_fac,
            query_factors.rotated_q.data(),
            query_factors.qr_to_c_L2sqr,
            query_factors.qr_norm_L2sqr,
            dim,
            ex_bits,
            rabitq_index->metric_type);
}

// Implementation of virtual make_knn_handler method
SIMDResultHandlerToFloat* IndexRaBitQFastScan::make_knn_handler(
        bool is_max,
        int /*impl*/,
        idx_t n,
        idx_t k,
        size_t /*ntotal*/,
        float* distances,
        idx_t* labels,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context) const {
    // Use runtime boolean for multi-bit mode
    const bool multi_bit = rabitq.nb_bits > 1;

    if (is_max) {
        return new RaBitQHeapHandler<CMax<uint16_t, int>, false>(
                this, n, k, distances, labels, sel, context, multi_bit);
    } else {
        return new RaBitQHeapHandler<CMin<uint16_t, int>, false>(
                this, n, k, distances, labels, sel, context, multi_bit);
    }
}

} // namespace faiss
