/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFRaBitQFastScan.h>

#include <algorithm>
#include <cstdio>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FastScanDistancePostProcessing.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizerMultiBit.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

// Import shared utilities from RaBitQUtils
using rabitq_utils::ExtraBitsFactors;
using rabitq_utils::QueryFactorsData;
using rabitq_utils::SignBitFactors;
using rabitq_utils::SignBitFactorsWithError;

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

/*********************************************************
 * IndexIVFRaBitQFastScan implementation
 *********************************************************/

IndexIVFRaBitQFastScan::IndexIVFRaBitQFastScan() = default;

IndexIVFRaBitQFastScan::IndexIVFRaBitQFastScan(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric,
        int bbs,
        bool own_invlists,
        uint8_t nb_bits)
        : IndexIVFFastScan(quantizer, d, nlist, 0, metric, own_invlists),
          rabitq(d, metric, nb_bits) {
    FAISS_THROW_IF_NOT_MSG(d > 0, "Dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "RaBitQ only supports L2 and Inner Product metrics");
    FAISS_THROW_IF_NOT_MSG(bbs % 32 == 0, "Batch size must be multiple of 32");
    FAISS_THROW_IF_NOT_MSG(quantizer != nullptr, "Quantizer cannot be null");

    by_residual = true;
    qb = 8; // RaBitQ quantization bits
    centered = false;

    // FastScan-specific parameters: 4 bits per sub-quantizer
    const size_t M_fastscan = (d + 3) / 4;
    constexpr size_t nbits_fastscan = 4;

    this->bbs = bbs;
    this->fine_quantizer = &rabitq;
    this->M = M_fastscan;
    this->nbits = nbits_fastscan;
    this->ksub = (1 << nbits_fastscan);
    this->M2 = roundup(M_fastscan, 2);

    // Compute code_size: bit_pattern + per-vector storage (factors/ex-codes)
    const size_t bit_pattern_size = (d + 7) / 8;
    this->code_size = bit_pattern_size + compute_per_vector_storage_size();

    is_trained = false;

    if (own_invlists) {
        replace_invlists(new BlockInvertedLists(nlist, get_CodePacker()), true);
    }

    flat_storage.clear();
}

// Constructor that converts an existing IndexIVFRaBitQ to FastScan format
IndexIVFRaBitQFastScan::IndexIVFRaBitQFastScan(
        const IndexIVFRaBitQ& orig,
        int /* bbs */)
        : IndexIVFFastScan(
                  orig.quantizer,
                  orig.d,
                  orig.nlist,
                  0,
                  orig.metric_type,
                  false),
          rabitq(orig.rabitq) {}

size_t IndexIVFRaBitQFastScan::compute_per_vector_storage_size() const {
    const size_t ex_bits = rabitq.nb_bits - 1;

    if (ex_bits == 0) {
        // 1-bit: only SignBitFactors (8 bytes)
        return sizeof(SignBitFactors);
    } else {
        // Multi-bit: SignBitFactorsWithError + ExtraBitsFactors + ex-codes
        return sizeof(SignBitFactorsWithError) + sizeof(ExtraBitsFactors) +
                (d * ex_bits + 7) / 8;
    }
}

void IndexIVFRaBitQFastScan::preprocess_code_metadata(
        idx_t n,
        const uint8_t* flat_codes,
        idx_t start_global_idx) {
    // Unified approach: always use flat_storage for both 1-bit and multi-bit
    const size_t storage_size = compute_per_vector_storage_size();
    flat_storage.resize((start_global_idx + n) * storage_size);

    // Copy factors data directly to flat storage (no reordering needed)
    const size_t bit_pattern_size = (d + 7) / 8;
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = flat_codes + i * code_size;
        const uint8_t* source_factors_ptr = code + bit_pattern_size;
        uint8_t* storage =
                flat_storage.data() + (start_global_idx + i) * storage_size;
        memcpy(storage, source_factors_ptr, storage_size);
    }
}

size_t IndexIVFRaBitQFastScan::code_packing_stride() const {
    // Use code_size as stride to skip embedded factor data during packing
    return code_size;
}

void IndexIVFRaBitQFastScan::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* assign) {
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(assign != nullptr || !by_residual);

    rabitq.train(n, x);
    is_trained = true;
    init_code_packer();
}

void IndexIVFRaBitQFastScan::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(list_nos != nullptr);
    FAISS_THROW_IF_NOT(codes != nullptr);
    FAISS_THROW_IF_NOT(is_trained);

    size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    size_t total_code_size = code_size + coarse_size;
    memset(codes, 0, total_code_size * n);

    const size_t ex_bits = rabitq.nb_bits - 1;

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];

            if (list_no >= 0) {
                const float* xi = x + i * d;
                uint8_t* code_out = codes + i * total_code_size;
                uint8_t* fastscan_code = code_out + coarse_size;

                // Reconstruct centroid for residual computation
                quantizer->reconstruct(list_no, centroid.data());

                const size_t bit_pattern_size = (d + 7) / 8;

                // Pack sign bits directly into FastScan format (inline)
                for (size_t j = 0; j < d; j++) {
                    const float or_minus_c = xi[j] - centroid[j];
                    if (or_minus_c > 0.0f) {
                        rabitq_utils::set_bit_fastscan(fastscan_code, j);
                    }
                }

                // Compute factors (with or without f_error depending on mode)
                SignBitFactorsWithError factors =
                        rabitq_utils::compute_vector_factors(
                                xi,
                                d,
                                centroid.data(),
                                rabitq.metric_type,
                                ex_bits > 0);

                if (ex_bits == 0) {
                    // 1-bit: store only SignBitFactors (8 bytes)
                    memcpy(fastscan_code + bit_pattern_size,
                           &factors,
                           sizeof(SignBitFactors));
                } else {
                    // Multi-bit: store full SignBitFactorsWithError (12 bytes)
                    memcpy(fastscan_code + bit_pattern_size,
                           &factors,
                           sizeof(SignBitFactorsWithError));

                    // Compute residual (needed for quantize_ex_bits)
                    std::vector<float> residual(d);
                    for (size_t j = 0; j < d; j++) {
                        residual[j] = xi[j] - centroid[j];
                    }

                    // Quantize ex-bits
                    const size_t ex_code_size = (d * ex_bits + 7) / 8;
                    uint8_t* ex_code = fastscan_code + bit_pattern_size +
                            sizeof(SignBitFactorsWithError);
                    ExtraBitsFactors ex_factors_temp;

                    rabitq_multibit::quantize_ex_bits(
                            residual.data(),
                            d,
                            rabitq.nb_bits,
                            ex_code,
                            ex_factors_temp,
                            rabitq.metric_type,
                            centroid.data());

                    memcpy(ex_code + ex_code_size,
                           &ex_factors_temp,
                           sizeof(ExtraBitsFactors));
                }

                // Include coarse codes if requested
                if (include_listnos) {
                    encode_listno(list_no, code_out);
                }
            }
        }
    }
}

bool IndexIVFRaBitQFastScan::lookup_table_is_3d() const {
    return true;
}

// Computes lookup table for residual vectors in RaBitQ FastScan format
void IndexIVFRaBitQFastScan::compute_residual_LUT(
        const float* residual,
        QueryFactorsData& query_factors,
        float* lut_out,
        const float* original_query) const {
    FAISS_THROW_IF_NOT(qb > 0 && qb <= 8);

    std::vector<float> rotated_q(d);
    std::vector<uint8_t> rotated_qq(d);

    // Use RaBitQUtils to compute query factors - eliminates code duplication
    query_factors = rabitq_utils::compute_query_factors(
            residual,
            d,
            nullptr,
            qb,
            centered,
            metric_type,
            rotated_q,
            rotated_qq);

    // Override query norm for inner product if original query is provided
    if (metric_type == MetricType::METRIC_INNER_PRODUCT &&
        original_query != nullptr) {
        query_factors.qr_norm_L2sqr = fvec_norm_L2sqr(original_query, d);
    }

    const size_t ex_bits = rabitq.nb_bits - 1;
    if (ex_bits > 0) {
        query_factors.rotated_q = rotated_q;
    }

    if (centered) {
        const float max_code_value = (1 << qb) - 1;

        for (size_t m = 0; m < M; m++) {
            const size_t dim_start = m * 4;

            for (int code_val = 0; code_val < 16; code_val++) {
                float xor_contribution = 0.0f;

                for (size_t dim_offset = 0; dim_offset < 4; dim_offset++) {
                    const size_t dim_idx = dim_start + dim_offset;

                    if (dim_idx < d) {
                        const bool db_bit = (code_val >> dim_offset) & 1;
                        const float query_value = rotated_qq[dim_idx];

                        xor_contribution += db_bit
                                ? (max_code_value - query_value)
                                : query_value;
                    }
                }

                lut_out[m * 16 + code_val] = xor_contribution;
            }
        }
    } else {
        for (size_t m = 0; m < M; m++) {
            const size_t dim_start = m * 4;

            for (int code_val = 0; code_val < 16; code_val++) {
                float inner_product = 0.0f;
                int popcount = 0;

                for (size_t dim_offset = 0; dim_offset < 4; dim_offset++) {
                    const size_t dim_idx = dim_start + dim_offset;

                    if (dim_idx < d && ((code_val >> dim_offset) & 1)) {
                        inner_product += rotated_qq[dim_idx];
                        popcount++;
                    }
                }
                lut_out[m * 16 + code_val] = query_factors.c1 * inner_product +
                        query_factors.c2 * popcount;
            }
        }
    }
}

void IndexIVFRaBitQFastScan::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !store_pairs, "store_pairs not supported for RaBitQFastScan");
    FAISS_THROW_IF_NOT_MSG(!stats, "stats not supported for this index");

    size_t nprobe = this->nprobe;
    if (params) {
        FAISS_THROW_IF_NOT(params->max_codes == 0);
        nprobe = params->nprobe;
    }

    std::vector<QueryFactorsData> query_factors_storage(n * nprobe);
    FastScanDistancePostProcessing context;
    context.query_factors = query_factors_storage.data();
    context.nprobe = nprobe;

    const CoarseQuantized cq = {nprobe, centroid_dis, assign};
    search_dispatch_implem(n, x, k, distances, labels, cq, context, params);
}

void IndexIVFRaBitQFastScan::compute_LUT(
        size_t n,
        const float* x,
        const CoarseQuantized& cq,
        AlignedTable<float>& dis_tables,
        AlignedTable<float>& biases,
        const FastScanDistancePostProcessing& context) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(by_residual);

    size_t nprobe = cq.nprobe;

    size_t dim12 = 16 * M;

    dis_tables.resize(n * nprobe * dim12);
    biases.resize(n * nprobe);

    if (n * nprobe > 0) {
        memset(biases.get(), 0, sizeof(float) * n * nprobe);
    }
    std::unique_ptr<float[]> xrel(new float[n * nprobe * d]);

#pragma omp parallel for if (n * nprobe > 1000)
    for (idx_t ij = 0; ij < n * nprobe; ij++) {
        idx_t i = ij / nprobe;
        float* xij = &xrel[ij * d];
        idx_t cij = cq.ids[ij];

        if (cij >= 0) {
            quantizer->compute_residual(x + i * d, xij, cij);

            // Create QueryFactorsData for this query-list combination
            QueryFactorsData query_factors_data;

            compute_residual_LUT(
                    xij,
                    query_factors_data,
                    dis_tables.get() + ij * dim12,
                    x + i * d);

            // Store query factors using compact indexing (ij directly)
            if (context.query_factors != nullptr) {
                context.query_factors[ij] = query_factors_data;
            }

        } else {
            memset(xij, -1, sizeof(float) * d);
            memset(dis_tables.get() + ij * dim12, -1, sizeof(float) * dim12);
        }
    }
}

void IndexIVFRaBitQFastScan::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    // Get centroid for this list
    std::vector<float> centroid(d);
    quantizer->reconstruct(list_no, centroid.data());

    // Unpack bit pattern from packed format
    const size_t bit_pattern_size = (d + 7) / 8;
    std::vector<uint8_t> fastscan_code(bit_pattern_size, 0);

    InvertedLists::ScopedCodes list_codes(invlists, list_no);
    for (size_t m = 0; m < M; m++) {
        uint8_t c =
                pq4_get_packed_element(list_codes.get(), bbs, M2, offset, m);

        size_t byte_idx = m / 2;
        if (m % 2 == 0) {
            fastscan_code[byte_idx] =
                    (fastscan_code[byte_idx] & 0xF0) | (c & 0x0F);
        } else {
            fastscan_code[byte_idx] =
                    (fastscan_code[byte_idx] & 0x0F) | ((c & 0x0F) << 4);
        }
    }

    // Get dp_multiplier directly from flat_storage
    InvertedLists::ScopedIds list_ids(invlists, list_no);
    idx_t global_id = list_ids[offset];

    float dp_multiplier = 1.0f;
    if (global_id >= 0) {
        const size_t storage_size = compute_per_vector_storage_size();
        const size_t storage_capacity = flat_storage.size() / storage_size;

        if (static_cast<size_t>(global_id) < storage_capacity) {
            const uint8_t* base_ptr =
                    flat_storage.data() + global_id * storage_size;
            const auto& base_factors =
                    *reinterpret_cast<const SignBitFactors*>(base_ptr);
            dp_multiplier = base_factors.dp_multiplier;
        }
    }

    // Decode residual directly using dp_multiplier
    std::vector<float> residual(d);
    decode_fastscan_to_residual(
            fastscan_code.data(), residual.data(), dp_multiplier);

    // Reconstruct: x = centroid + residual
    for (size_t j = 0; j < d; j++) {
        recons[j] = centroid[j] + residual[j];
    }
}

void IndexIVFRaBitQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(bytes != nullptr);
    FAISS_THROW_IF_NOT(x != nullptr);

    size_t coarse_size = coarse_code_size();
    size_t total_code_size = code_size + coarse_size;
    std::vector<float> centroid(d);
    std::vector<float> residual(d);
    const size_t bit_pattern_size = (d + 7) / 8;

#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code_i = bytes + i * total_code_size;
        float* x_i = x + i * d;

        idx_t list_no = decode_listno(code_i);

        if (list_no >= 0 && list_no < nlist) {
            quantizer->reconstruct(list_no, centroid.data());

            const uint8_t* fastscan_code = code_i + coarse_size;

            const uint8_t* factors_ptr = fastscan_code + bit_pattern_size;
            const auto& base_factors =
                    *reinterpret_cast<const SignBitFactors*>(factors_ptr);

            decode_fastscan_to_residual(
                    fastscan_code, residual.data(), base_factors.dp_multiplier);

            for (size_t j = 0; j < d; j++) {
                x_i[j] = centroid[j] + residual[j];
            }
        } else {
            memset(x_i, 0, sizeof(float) * d);
        }
    }
}

void IndexIVFRaBitQFastScan::decode_fastscan_to_residual(
        const uint8_t* fastscan_code,
        float* residual,
        float dp_multiplier) const {
    memset(residual, 0, sizeof(float) * d);

    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

    for (size_t j = 0; j < d; j++) {
        bool bit_value = rabitq_utils::extract_bit_fastscan(fastscan_code, j);

        float bit_as_float = bit_value ? 1.0f : 0.0f;
        residual[j] = (bit_as_float - 0.5f) * dp_multiplier * 2 * inv_d_sqrt;
    }
}

// Implementation of virtual make_knn_handler method
SIMDResultHandlerToFloat* IndexIVFRaBitQFastScan::make_knn_handler(
        bool is_max,
        int /* impl */,
        idx_t n,
        idx_t k,
        float* distances,
        idx_t* labels,
        const IDSelector* /* sel */,
        const FastScanDistancePostProcessing& context,
        const float* /* normalizers */) const {
    const size_t ex_bits = rabitq.nb_bits - 1;
    const bool is_multibit = ex_bits > 0;

    if (is_max) {
        return new IVFRaBitQHeapHandler<CMax<uint16_t, int64_t>>(
                this, n, k, distances, labels, &context, is_multibit);
    } else {
        return new IVFRaBitQHeapHandler<CMin<uint16_t, int64_t>>(
                this, n, k, distances, labels, &context, is_multibit);
    }
}

/*********************************************************
 * IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler implementation
 *********************************************************/

template <class C>
IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::IVFRaBitQHeapHandler(
        const IndexIVFRaBitQFastScan* idx,
        size_t nq_val,
        size_t k_val,
        float* distances,
        int64_t* labels,
        const FastScanDistancePostProcessing* ctx,
        bool multibit)
        : simd_result_handlers::ResultHandlerCompare<C, true>(
                  nq_val,
                  0,
                  nullptr),
          index(idx),
          heap_distances(distances),
          heap_labels(labels),
          nq(nq_val),
          k(k_val),
          context(ctx),
          is_multibit(multibit) {
    current_list_no = 0;
    probe_indices.clear();

    // Initialize heaps in constructor (standard pattern from HeapHandler)
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_heapify<Cfloat>(k, heap_dis, heap_ids);
    }
}

template <class C>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::handle(
        size_t q,
        size_t b,
        simd16uint16 d0,
        simd16uint16 d1) {
    // Store the original local query index before adjust_with_origin changes it
    size_t local_q = q;
    this->adjust_with_origin(q, d0, d1);

    ALIGNED(32) uint16_t d32tab[32];
    d0.store(d32tab);
    d1.store(d32tab + 16);

    float* const heap_dis = heap_distances + q * k;
    int64_t* const heap_ids = heap_labels + q * k;

    FAISS_THROW_IF_NOT_FMT(
            !probe_indices.empty() && local_q < probe_indices.size(),
            "set_list_context() must be called before handle() - probe_indices size: %zu, local_q: %zu, global_q: %zu",
            probe_indices.size(),
            local_q,
            q);

    // Access query factors directly from array via ProcessingContext
    if (!context || !context->query_factors) {
        FAISS_THROW_MSG(
                "Query factors not available: FastScanDistancePostProcessing with query_factors required");
    }

    // Use probe_rank from probe_indices for compact storage indexing
    size_t probe_rank = probe_indices[local_q];
    size_t nprobe = context->nprobe > 0 ? context->nprobe : index->nprobe;
    size_t storage_idx = q * nprobe + probe_rank;

    const auto& query_factors = context->query_factors[storage_idx];

    const float one_a =
            this->normalizers ? (1.0f / this->normalizers[2 * q]) : 1.0f;
    const float bias = this->normalizers ? this->normalizers[2 * q + 1] : 0.0f;

    uint64_t idx_base = this->j0 + b * 32;
    if (idx_base >= this->ntotal) {
        return;
    }

    size_t max_positions = std::min<size_t>(32, this->ntotal - idx_base);

    // Stats tracking for two-stage search
    // n_1bit_evaluations: candidates evaluated using 1-bit lower bound
    // n_multibit_evaluations: candidates requiring full multi-bit distance
    size_t local_1bit_evaluations = 0;
    size_t local_multibit_evaluations = 0;

    // Process each candidate vector in the SIMD batch
    for (size_t j = 0; j < max_positions; j++) {
        const int64_t result_id = this->adjust_id(b, j);

        if (result_id < 0) {
            continue;
        }

        const float normalized_distance = d32tab[j] * one_a + bias;

        // Get database factors from flat_storage
        const size_t storage_size = index->compute_per_vector_storage_size();
        const uint8_t* base_ptr =
                index->flat_storage.data() + result_id * storage_size;

        if (is_multibit) {
            // Track candidates actually considered for two-stage filtering
            local_1bit_evaluations++;

            // Multi-bit: use SignBitFactorsWithError and two-stage search
            const SignBitFactorsWithError& full_factors =
                    *reinterpret_cast<const SignBitFactorsWithError*>(base_ptr);

            // Compute 1-bit adjusted distance using shared helper
            float dist_1bit = rabitq_utils::compute_1bit_adjusted_distance(
                    normalized_distance,
                    full_factors,
                    query_factors,
                    index->centered,
                    index->qb,
                    index->d);

            // Compute lower bound using error bound
            float lower_bound =
                    compute_lower_bound(dist_1bit, result_id, local_q, q);

            // Adaptive filtering: decide whether to compute full distance
            const bool is_similarity =
                    index->metric_type == MetricType::METRIC_INNER_PRODUCT;
            bool should_refine = is_similarity
                    ? (lower_bound > heap_dis[0])  // IP: keep if better
                    : (lower_bound < heap_dis[0]); // L2: keep if better

            if (should_refine) {
                local_multibit_evaluations++;

                // Compute local_offset: position within current inverted list
                size_t local_offset = this->j0 + b * 32 + j;

                // Compute full multi-bit distance
                float dist_full = compute_full_multibit_distance(
                        result_id, local_q, q, local_offset);

                // Update heap if this distance is better
                if (Cfloat::cmp(heap_dis[0], dist_full)) {
                    heap_replace_top<Cfloat>(
                            k, heap_dis, heap_ids, dist_full, result_id);
                }
            }
        } else {
            const auto& db_factors =
                    *reinterpret_cast<const SignBitFactors*>(base_ptr);

            // Compute adjusted distance using shared helper
            float adjusted_distance =
                    rabitq_utils::compute_1bit_adjusted_distance(
                            normalized_distance,
                            db_factors,
                            query_factors,
                            index->centered,
                            index->qb,
                            index->d);

            if (Cfloat::cmp(heap_dis[0], adjusted_distance)) {
                heap_replace_top<Cfloat>(
                        k, heap_dis, heap_ids, adjusted_distance, result_id);
            }
        }
    }

    // Update global stats atomically
#pragma omp atomic
    rabitq_stats.n_1bit_evaluations += local_1bit_evaluations;
#pragma omp atomic
    rabitq_stats.n_multibit_evaluations += local_multibit_evaluations;
}

template <class C>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::set_list_context(
        size_t list_no,
        const std::vector<int>& probe_map) {
    current_list_no = list_no;
    probe_indices = probe_map;
}

template <class C>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::begin(
        const float* norms) {
    this->normalizers = norms;
}

template <class C>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::end() {
#pragma omp parallel for
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_reorder<Cfloat>(k, heap_dis, heap_ids);
    }
}

template <class C>
float IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::compute_lower_bound(
        float dist_1bit,
        size_t db_idx,
        size_t local_q,
        size_t global_q) const {
    // Access f_error from SignBitFactorsWithError in flat storage
    const size_t storage_size = index->compute_per_vector_storage_size();
    const uint8_t* base_ptr =
            index->flat_storage.data() + db_idx * storage_size;
    const SignBitFactorsWithError& db_factors =
            *reinterpret_cast<const SignBitFactorsWithError*>(base_ptr);
    float f_error = db_factors.f_error;

    // Get g_error from query factors
    // Use local_q to access probe_indices (batch-local), global_q for storage
    float g_error = 0.0f;
    if (context && context->query_factors) {
        size_t probe_rank = probe_indices[local_q];
        size_t nprobe = context->nprobe > 0 ? context->nprobe : index->nprobe;
        size_t storage_idx = global_q * nprobe + probe_rank;
        g_error = context->query_factors[storage_idx].g_error;
    }

    // Compute error adjustment: f_error * g_error
    float error_adjustment = f_error * g_error;

    return dist_1bit - error_adjustment;
}

template <class C>
float IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::
        compute_full_multibit_distance(
                size_t db_idx,
                size_t local_q,
                size_t global_q,
                size_t local_offset) const {
    const size_t ex_bits = index->rabitq.nb_bits - 1;
    const size_t dim = index->d;

    const size_t storage_size = index->compute_per_vector_storage_size();
    const uint8_t* base_ptr =
            index->flat_storage.data() + db_idx * storage_size;

    const size_t ex_code_size = (dim * ex_bits + 7) / 8;
    const uint8_t* ex_code = base_ptr + sizeof(SignBitFactorsWithError);
    const ExtraBitsFactors& ex_fac = *reinterpret_cast<const ExtraBitsFactors*>(
            base_ptr + sizeof(SignBitFactorsWithError) + ex_code_size);

    // Use local_q to access probe_indices (batch-local), global_q for storage
    size_t probe_rank = probe_indices[local_q];
    size_t nprobe = context->nprobe > 0 ? context->nprobe : index->nprobe;
    size_t storage_idx = global_q * nprobe + probe_rank;
    const auto& query_factors = context->query_factors[storage_idx];

    size_t list_no = current_list_no;
    InvertedLists::ScopedCodes list_codes(index->invlists, list_no);

    std::vector<uint8_t> unpacked_code(index->code_size);
    CodePackerPQ4 packer(index->M2, index->bbs);
    packer.unpack_1(list_codes.get(), local_offset, unpacked_code.data());
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
            index->metric_type);
}

} // namespace faiss
