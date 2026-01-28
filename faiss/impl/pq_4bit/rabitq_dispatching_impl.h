/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * Internal header containing template implementations for RaBitQ FastScan.
 * This is included by rabitq-avx2.cpp and rabitq-avx512.cpp which are
 * compiled with appropriate SIMD flags.
 *
 * DO NOT include this header in files compiled without AVX2/AVX512 flags.
 */

#include <faiss/IndexIVFRaBitQFastScan.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/FastScanDistancePostProcessing.h>
#include <faiss/impl/RaBitQStats.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/pq_4bit/dispatching.h>
#include <faiss/impl/pq_4bit/kernels_simd256.h>
#include <faiss/impl/pq_4bit/pq4_fast_scan.h>
#include <faiss/utils/Heap.h>

namespace faiss {

// Template implementations for RaBitQHeapHandler
template <class C, bool with_id_map, SIMDLevel SL>
RaBitQHeapHandler<C, with_id_map, SL>::RaBitQHeapHandler(
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
#pragma omp parallel for if (nq > 100)
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_heapify<Cfloat>(k, heap_dis, heap_ids);
    }
}

template <class C, bool with_id_map, SIMDLevel SL>
void RaBitQHeapHandler<C, with_id_map, SL>::handle(
        size_t q,
        size_t b,
        simd16uint16 d0,
        simd16uint16 d1) {
    ALIGNED(32) uint16_t d32tab[32];
    d0.store(d32tab);
    d1.store(d32tab + 16);

    float* const heap_dis = heap_distances + q * k;
    int64_t* const heap_ids = heap_labels + q * k;

    rabitq_utils::QueryFactorsData query_factors_data = {};
    if (context.query_factors != nullptr) {
        query_factors_data = context.query_factors[q];
    }

    const float one_a = normalizers ? (1.0f / normalizers[2 * q]) : 1.0f;
    const float bias = normalizers ? normalizers[2 * q + 1] : 0.0f;

    const size_t base_db_idx = this->j0 + b * 32;
    const size_t max_vectors = (base_db_idx < rabitq_index->ntotal)
            ? std::min<size_t>(32, rabitq_index->ntotal - base_db_idx)
            : 0;

    const size_t storage_size = rabitq_index->compute_per_vector_storage_size();

    size_t local_1bit_evaluations = 0;
    size_t local_multibit_evaluations = 0;

    for (size_t i = 0; i < max_vectors; i++) {
        const size_t db_idx = base_db_idx + i;
        const float normalized_distance = d32tab[i] * one_a + bias;
        const uint8_t* base_ptr =
                rabitq_index->flat_storage.data() + db_idx * storage_size;

        if (is_multi_bit) {
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

            const bool is_similarity = rabitq_index->metric_type ==
                    MetricType::METRIC_INNER_PRODUCT;
            bool should_refine = rabitq_utils::should_refine_candidate(
                    dist_1bit,
                    full_factors.f_error,
                    context.query_factors ? context.query_factors[q].g_error
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

template <class C, bool with_id_map, SIMDLevel SL>
void RaBitQHeapHandler<C, with_id_map, SL>::begin(const float* norms) {
    normalizers = norms;
}

template <class C, bool with_id_map, SIMDLevel SL>
void RaBitQHeapHandler<C, with_id_map, SL>::end() {
#pragma omp parallel for if (nq > 100)
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_reorder<Cfloat>(k, heap_dis, heap_ids);
    }
}

template <class C, bool with_id_map, SIMDLevel SL>
float RaBitQHeapHandler<C, with_id_map, SL>::compute_lower_bound(
        float dist_1bit,
        size_t db_idx,
        size_t q) const {
    const size_t storage_size = rabitq_index->compute_per_vector_storage_size();
    const uint8_t* base_ptr =
            rabitq_index->flat_storage.data() + db_idx * storage_size;
    const SignBitFactorsWithError& db_factors =
            *reinterpret_cast<const SignBitFactorsWithError*>(base_ptr);
    float f_error = db_factors.f_error;

    float g_error = 0.0f;
    if (context.query_factors != nullptr) {
        g_error = context.query_factors[q].g_error;
    }

    float error_adjustment = f_error * g_error;
    return dist_1bit - error_adjustment;
}

template <class C, bool with_id_map, SIMDLevel SL>
float RaBitQHeapHandler<C, with_id_map, SL>::compute_full_multibit_distance(
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

    const rabitq_utils::QueryFactorsData& query_factors =
            context.query_factors[q];

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

// Wrapper class that adds accumulate_loop functionality to RaBitQHeapHandler
template <class C, bool with_id_map, SIMDLevel SL>
struct RaBitQScannerMixIn : RaBitQHeapHandler<C, with_id_map, SL> {
    using Base = RaBitQHeapHandler<C, with_id_map, SL>;
    DummyScaler<SL> scaler;

    RaBitQScannerMixIn(
            const IndexRaBitQFastScan* index,
            size_t nq_val,
            size_t k_val,
            float* distances,
            int64_t* labels,
            const IDSelector* sel_in,
            const FastScanDistancePostProcessing& ctx,
            bool multi_bit)
            : Base(index,
                   nq_val,
                   k_val,
                   distances,
                   labels,
                   sel_in,
                   ctx,
                   multi_bit),
              scaler(-1) {}

    void accumulate_loop(
            int nq,
            size_t nb,
            int bbs,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT) override {
        pq4_accumulate_loop_fixed_scaler<SL, Base>(
                nq, nb, bbs, nsq, codes, LUT, *this, scaler);
    }

    void accumulate_loop_qbs(
            int qbs,
            size_t nb,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT) override {
        pq4_accumulate_loop_qbs_fixed_scaler<Base, DummyScaler<SL>>(
                qbs, nb, nsq, codes, LUT, *this, scaler);
    }
};

// Factory function implementation for flat RaBitQ
template <SIMDLevel SL>
PQ4CodeScanner* make_rabitq_flat_knn_handler_impl_body(
        const IndexRaBitQFastScan* index,
        bool is_max,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* labels,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool multi_bit) {
    if (is_max) {
        return new RaBitQScannerMixIn<CMax<uint16_t, int64_t>, false, SL>(
                index, nq, k, distances, labels, sel, context, multi_bit);
    } else {
        return new RaBitQScannerMixIn<CMin<uint16_t, int64_t>, false, SL>(
                index, nq, k, distances, labels, sel, context, multi_bit);
    }
}

/*********************************************************
 * IVF RaBitQ template implementations
 *********************************************************/

// Template implementations for IVFRaBitQHeapHandler
template <class C, SIMDLevel SL>
IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C, SL>::IVFRaBitQHeapHandler(
        const IndexIVFRaBitQFastScan* idx,
        size_t nq_val,
        size_t k_val,
        float* distances,
        int64_t* labels,
        const FastScanDistancePostProcessing* ctx,
        bool multibit)
        : simd_result_handlers::ResultHandlerCompare<C, true, SL>(
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

    // Initialize heaps in constructor
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_heapify<Cfloat>(k, heap_dis, heap_ids);
    }
}

template <class C, SIMDLevel SL>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C, SL>::handle(
        size_t q,
        size_t b,
        simd16uint16 d0,
        simd16uint16 d1) {
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

    if (!context || !context->query_factors) {
        FAISS_THROW_MSG(
                "Query factors not available: FastScanDistancePostProcessing with query_factors required");
    }

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

    size_t local_1bit_evaluations = 0;
    size_t local_multibit_evaluations = 0;

    for (size_t j = 0; j < max_positions; j++) {
        const int64_t result_id = this->adjust_id(b, j);

        if (result_id < 0) {
            continue;
        }

        const float normalized_distance = d32tab[j] * one_a + bias;

        const size_t storage_size = index->compute_per_vector_storage_size();
        const uint8_t* base_ptr =
                index->flat_storage.data() + result_id * storage_size;

        if (is_multibit) {
            local_1bit_evaluations++;

            const SignBitFactorsWithError& full_factors =
                    *reinterpret_cast<const SignBitFactorsWithError*>(base_ptr);

            float dist_1bit = rabitq_utils::compute_1bit_adjusted_distance(
                    normalized_distance,
                    full_factors,
                    query_factors,
                    index->centered,
                    index->qb,
                    index->d);

            const bool is_similarity =
                    index->metric_type == MetricType::METRIC_INNER_PRODUCT;

            float g_error = query_factors.g_error;

            bool should_refine = rabitq_utils::should_refine_candidate(
                    dist_1bit,
                    full_factors.f_error,
                    g_error,
                    heap_dis[0],
                    is_similarity);
            if (should_refine) {
                local_multibit_evaluations++;

                size_t local_offset = this->j0 + b * 32 + j;

                float dist_full = compute_full_multibit_distance(
                        result_id, local_q, q, local_offset);

                if (Cfloat::cmp(heap_dis[0], dist_full)) {
                    heap_replace_top<Cfloat>(
                            k, heap_dis, heap_ids, dist_full, result_id);
                    nup++;
                }
            }
        } else {
            const auto& db_factors =
                    *reinterpret_cast<const SignBitFactors*>(base_ptr);

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
                nup++;
            }
        }
    }

#pragma omp atomic
    rabitq_stats.n_1bit_evaluations += local_1bit_evaluations;
#pragma omp atomic
    rabitq_stats.n_multibit_evaluations += local_multibit_evaluations;
}

template <class C, SIMDLevel SL>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C, SL>::set_list_context(
        size_t list_no,
        const std::vector<int>& probe_map) {
    current_list_no = list_no;
    probe_indices = probe_map;
}

template <class C, SIMDLevel SL>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C, SL>::begin(
        const float* norms) {
    this->normalizers = norms;
}

template <class C, SIMDLevel SL>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C, SL>::end() {
#pragma omp parallel for
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_reorder<Cfloat>(k, heap_dis, heap_ids);
    }
}

template <class C, SIMDLevel SL>
float IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C, SL>::
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

// Wrapper that adds accumulate_loop() methods to IVFRaBitQHeapHandler
template <class C, SIMDLevel SL>
struct IVFRaBitQScannerMixIn
        : IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C, SL> {
    using Base = IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C, SL>;
    DummyScaler<SL> scaler;

    IVFRaBitQScannerMixIn(
            const IndexIVFRaBitQFastScan* idx,
            size_t nq_val,
            size_t k_val,
            float* distances,
            int64_t* labels,
            const FastScanDistancePostProcessing* ctx,
            bool multibit)
            : Base(idx, nq_val, k_val, distances, labels, ctx, multibit),
              scaler(-1) {}

    void accumulate_loop(
            int nq,
            size_t nb,
            int bbs,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT) override {
        pq4_accumulate_loop_fixed_scaler<SL, Base>(
                nq, nb, bbs, nsq, codes, LUT, *this, scaler);
    }

    void accumulate_loop_qbs(
            int qbs,
            size_t nb,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT) override {
        pq4_accumulate_loop_qbs_fixed_scaler<Base, DummyScaler<SL>>(
                qbs, nb, nsq, codes, LUT, *this, scaler);
    }
};

// Factory function implementation for IVF RaBitQ
template <SIMDLevel SL>
PQ4CodeScanner* make_ivf_rabitq_handler_impl_body(
        bool is_max,
        const IndexIVFRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* labels,
        const FastScanDistancePostProcessing* context,
        bool multibit) {
    if (is_max) {
        return new IVFRaBitQScannerMixIn<CMax<uint16_t, int64_t>, SL>(
                index, nq, k, distances, labels, context, multibit);
    } else {
        return new IVFRaBitQScannerMixIn<CMin<uint16_t, int64_t>, SL>(
                index, nq, k, distances, labels, context, multibit);
    }
}

} // namespace faiss
