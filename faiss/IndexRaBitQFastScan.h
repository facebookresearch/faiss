/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <faiss/IndexFastScan.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/impl/RaBitQStats.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/Heap.h>

namespace faiss {

// Import shared utilities from RaBitQUtils
using rabitq_utils::ExtraBitsFactors;
using rabitq_utils::QueryFactorsData;
using rabitq_utils::SignBitFactors;
using rabitq_utils::SignBitFactorsWithError;

/** Fast-scan version of RaBitQ index that processes 32 database vectors at a
 * time using SIMD operations. Similar to IndexPQFastScan but adapted for
 * RaBitQ's bit-level quantization with factors.
 *
 * The key differences from IndexRaBitQ:
 * - Processes vectors in batches of 32
 * - Uses 4-bit groupings for SIMD optimization (4 dimensions per 4-bit unit)
 * - Separates factors from quantized bits for efficient processing
 * - Leverages existing PQ4 FastScan infrastructure where possible
 */
struct IndexRaBitQFastScan : IndexFastScan {
    /// RaBitQ quantizer for encoding/decoding
    RaBitQuantizer rabitq;

    /// Center of all points (same as IndexRaBitQ)
    std::vector<float> center;

    /// Default number of bits to quantize a query with
    uint8_t qb = 8;

    // quantize the query with a zero-centered scalar quantizer.
    bool centered = false;

    IndexRaBitQFastScan();

    explicit IndexRaBitQFastScan(
            idx_t d,
            MetricType metric = METRIC_L2,
            int bbs = 32,
            uint8_t nb_bits = 1);

    /// build from an existing IndexRaBitQ
    explicit IndexRaBitQFastScan(const IndexRaBitQ& orig, int bbs = 32);

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void compute_codes(uint8_t* codes, idx_t n, const float* x) const override;

    /// Compute per-vector auxiliary data size in block aux region
    size_t compute_per_vector_storage_size() const;

    void compute_float_LUT(
            float* lut,
            idx_t n,
            const float* x,
            const FastScanDistancePostProcessing& context) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /// Return CodePackerRaBitQ with enlarged block size
    CodePacker* get_CodePacker() const override;

    /// Remove vectors and compact both PQ4 codes and auxiliary data
    size_t remove_ids(const IDSelector& sel) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// RaBitQ scanner wired through rabitq_make_knn_scanner
    std::unique_ptr<FastScanCodeScanner> make_knn_scanner(
            bool is_max,
            idx_t n,
            idx_t k,
            size_t ntotal,
            float* distances,
            idx_t* labels,
            const IDSelector* sel,
            const FastScanDistancePostProcessing& context = {}) const override;

    /// Override to create RaBitQ-specific handlers
    SIMDResultHandlerToFloat* make_knn_handler(
            bool is_max,
            int /*impl*/,
            idx_t n,
            idx_t k,
            size_t /*ntotal*/,
            float* distances,
            idx_t* labels,
            const IDSelector* sel,
            const FastScanDistancePostProcessing& context) const override;
};

/** SIMD result handler for RaBitQ FastScan that applies distance corrections
 * and maintains heaps directly during SIMD operations.
 *
 * This handler processes batches of 32 distance computations from SIMD kernels,
 * applies RaBitQ-specific adjustments (factors and normalizers), and
 * immediately updates result heaps without intermediate storage. This
 * eliminates the need for post-processing and provides significant memory and
 * performance benefits.
 *
 * Key optimizations:
 * - Direct heap integration (no intermediate result storage)
 * - Batch-level computation of normalizers and query factors
 * - Preserves exact mathematical equivalence to original RaBitQ distances
 * - Runtime boolean for multi-bit support
 *
 * @tparam C Comparator type (CMin/CMax) for heap operations
 * @tparam with_id_map Whether to use id mapping (similar to HeapHandler)
 * @tparam SL SIMD level for per-TU template instantiation
 */
template <
        class C,
        bool with_id_map = false,
        SIMDLevel SL = SINGLE_SIMD_LEVEL_256>
struct RaBitQHeapHandler
        : simd_result_handlers::ResultHandlerCompare<C, with_id_map, SL> {
    using RHC = simd_result_handlers::ResultHandlerCompare<C, with_id_map, SL>;
    using RHC::normalizers;
    static constexpr SIMDLevel SL256 = simd256_level_selector<SL>::value;
    using simd16uint16 = simd16uint16_tpl<SL256>;

    const IndexRaBitQFastScan* rabitq_index;
    float* heap_distances; // [nq * k]
    int64_t* heap_labels;  // [nq * k]
    const size_t nq, k;
    const FastScanDistancePostProcessing*
            context;         // Processing context with query offset
    const bool is_multi_bit; // Runtime flag for multi-bit mode

    // Cached block-layout constants (invariant for handler lifetime)
    const size_t storage_size;
    const size_t packed_block_size;
    const size_t full_block_size;
    std::unique_ptr<CodePacker> packer; // cached for unpack in hot path

    // Use float-based comparator for heap operations
    using Cfloat = typename std::conditional<
            C::is_max,
            CMax<float, int64_t>,
            CMin<float, int64_t>>::type;

    RaBitQHeapHandler(
            const IndexRaBitQFastScan* index,
            size_t nq_val,
            size_t k_val,
            float* distances,
            int64_t* labels,
            const IDSelector* sel_in,
            const FastScanDistancePostProcessing* ctx,
            bool multi_bit)
            : RHC(nq_val, index->ntotal, sel_in),
              rabitq_index(index),
              heap_distances(distances),
              heap_labels(labels),
              nq(nq_val),
              k(k_val),
              context(ctx),
              is_multi_bit(multi_bit),
              storage_size(index->compute_per_vector_storage_size()),
              packed_block_size(((index->M2 + 1) / 2) * index->bbs),
              full_block_size(index->get_block_stride()),
              packer(index->get_CodePacker()) {
#pragma omp parallel for if (nq > 100)
        for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
            float* heap_dis = heap_distances + q * k;
            int64_t* heap_ids = heap_labels + q * k;
            heap_heapify<Cfloat>(k, heap_dis, heap_ids);
        }
    }

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) override {
        ALIGNED(32) uint16_t d32tab[32];
        d0.store(d32tab);
        d1.store(d32tab + 16);

        float* const heap_dis = heap_distances + q * k;
        int64_t* const heap_ids = heap_labels + q * k;

        rabitq_utils::QueryFactorsData query_factors_data = {};
        if (context && context->query_factors != nullptr) {
            query_factors_data = context->query_factors[q];
        }

        const float one_a = normalizers ? (1.0f / normalizers[2 * q]) : 1.0f;
        const float bias = normalizers ? normalizers[2 * q + 1] : 0.0f;

        const size_t base_db_idx = this->j0 + b * 32;
        const size_t max_vectors = (base_db_idx < rabitq_index->ntotal)
                ? std::min<size_t>(32, rabitq_index->ntotal - base_db_idx)
                : 0;

        const size_t block_idx = base_db_idx / rabitq_index->bbs;
        const uint8_t* aux_base = rabitq_index->codes.get() +
                block_idx * full_block_size + packed_block_size;

        size_t local_1bit_evaluations = 0;
        size_t local_multibit_evaluations = 0;

        for (size_t i = 0; i < max_vectors; i++) {
            const size_t db_idx = base_db_idx + i;
            const float normalized_distance = d32tab[i] * one_a + bias;
            const uint8_t* base_ptr = aux_base + i * storage_size;

            if (is_multi_bit) {
                local_1bit_evaluations++;

                const SignBitFactorsWithError& full_factors =
                        *reinterpret_cast<const SignBitFactorsWithError*>(
                                base_ptr);

                float dist_1bit = rabitq_utils::compute_1bit_adjusted_distance(
                        normalized_distance,
                        full_factors,
                        query_factors_data,
                        rabitq_index->centered,
                        rabitq_index->qb,
                        rabitq_index->d);

                const bool is_similarity = rabitq_index->metric_type ==
                        MetricType::METRIC_INNER_PRODUCT;
                bool should_refine = rabitq_utils::should_refine_candidate(
                        dist_1bit,
                        full_factors.f_error,
                        context && context->query_factors
                                ? context->query_factors[q].g_error
                                : 0.0f,
                        heap_dis[0],
                        is_similarity);

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

                if (Cfloat::cmp(heap_dis[0], adjusted_distance)) {
                    heap_replace_top<Cfloat>(
                            k, heap_dis, heap_ids, adjusted_distance, db_idx);
                }
            }
        }

#pragma omp atomic
        rabitq_stats.n_1bit_evaluations += local_1bit_evaluations;
#pragma omp atomic
        rabitq_stats.n_multibit_evaluations += local_multibit_evaluations;
    }

    void begin(const float* norms) {
        normalizers = norms;
    }

    void end() {
#pragma omp parallel for if (nq > 100)
        for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
            float* heap_dis = heap_distances + q * k;
            int64_t* heap_ids = heap_labels + q * k;
            heap_reorder<Cfloat>(k, heap_dis, heap_ids);
        }
    }

   private:
    float compute_full_multibit_distance(size_t db_idx, size_t q) const {
        const size_t ex_bits = rabitq_index->rabitq.nb_bits - 1;
        const size_t dim = rabitq_index->d;

        const uint8_t* base_ptr = rabitq_utils::get_block_aux_ptr(
                rabitq_index->codes.get(),
                db_idx,
                rabitq_index->bbs,
                packed_block_size,
                full_block_size,
                storage_size);

        const size_t ex_code_size = (dim * ex_bits + 7) / 8;
        const uint8_t* ex_code = base_ptr + sizeof(SignBitFactorsWithError);
        const ExtraBitsFactors& ex_fac =
                *reinterpret_cast<const ExtraBitsFactors*>(
                        base_ptr + sizeof(SignBitFactorsWithError) +
                        ex_code_size);

        const rabitq_utils::QueryFactorsData& query_factors =
                context->query_factors[q];

        std::vector<uint8_t> unpacked_code(rabitq_index->code_size);
        packer->unpack_1(
                rabitq_index->codes.get(), db_idx, unpacked_code.data());
        const uint8_t* sign_bits = unpacked_code.data();

        return rabitq_utils::compute_full_multibit_distance(
                sign_bits,
                ex_code,
                ex_fac,
                query_factors.rotated_q.data(),
                (rabitq_index->metric_type == MetricType::METRIC_INNER_PRODUCT)
                        ? query_factors.q_dot_c
                        : query_factors.qr_to_c_L2sqr,
                dim,
                ex_bits,
                rabitq_index->metric_type);
    }
};

} // namespace faiss
