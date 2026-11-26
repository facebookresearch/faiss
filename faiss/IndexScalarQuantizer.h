/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_SCALAR_QUANTIZER_H
#define FAISS_INDEX_SCALAR_QUANTIZER_H

#include <stdint.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/sq-fp16-gemm.h>

namespace faiss {

/**
 * Flat index built on a scalar quantizer.
 */
struct IndexScalarQuantizer : IndexFlatCodes {
    /// Used to encode the vectors
    ScalarQuantizer sq;

    /** Optional batched search path for QT_fp16 with METRIC_L2 or
     * METRIC_INNER_PRODUCT, meant for the case where every stored vector is
     * scored against every query, e.g. an IVF coarse quantizer.
     *
     * When enabled, search() keeps a packed copy of the fp16 codes (built on
     * first use, dropped by add / remove / reset / merge) and scores whole
     * blocks of queries against whole blocks of codes with an fp16
     * matrix-multiply kernel (scalar, AVX-512/F16C/FMA or AMX-FP16, see
     * fp16_gemm::GemmKernel). Queries are rounded to fp16 first; non-finite
     * queries and queries that overflow fp16 are rejected. With
     * fp16_gemm_rescore, the distances of the selected results are then
     * recomputed with the regular ScalarQuantizer distance computer on the
     * rounded queries. Results are sorted by distance, ties by smaller id.
     * ID selectors are not supported on this path. Only search() is
     * affected; range_search() and the distance computers are unchanged.
     *
     * Off by default since the packed copy doubles the memory used by the
     * codes. These settings are not serialized: set them again after
     * read_index(). Concurrent search() calls are safe; callers that modify
     * codes directly must call invalidate_fp16_gemm_cache().
     */
    bool use_fp16_gemm = false;
    /// AUTO picks the best kernel available at runtime; an explicit kernel
    /// that is not available makes search() throw.
    fp16_gemm::GemmKernel fp16_gemm_kernel = fp16_gemm::GemmKernel::AUTO;
    /// queries per block (rounded up to a multiple of 32)
    size_t fp16_gemm_bs_x = 256;
    /// stored vectors per block (rounded up to a multiple of 32)
    size_t fp16_gemm_bs_y = 1024;
    /// recompute the returned distances with the ScalarQuantizer
    bool fp16_gemm_rescore = true;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param qtype  type of scalar quantizer (e.g., QT_4bit)
     * @param metric distance metric used for search (default: METRIC_L2)
     */
    IndexScalarQuantizer(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2);

    IndexScalarQuantizer();

    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /* standalone codec interface */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    // The overrides below only drop the fp16 gemm packed copy.
    void add(idx_t n, const float* x) override;
    void add_sa_codes(idx_t n, const uint8_t* codes_in, const idx_t* xids)
            override;
    void reset() override;
    size_t remove_ids(const IDSelector& sel) override;
    void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    /// Drops the fp16 gemm packed copy; it is rebuilt on the next search.
    void invalidate_fp16_gemm_cache();
    /// Kernel search() uses when use_fp16_gemm is set (throws if
    /// fp16_gemm_kernel is not available on this host).
    fp16_gemm::GemmKernel fp16_gemm_effective_kernel() const;
    /// Bytes held by the fp16 gemm packed copy (0 if not built).
    size_t fp16_gemm_packed_bytes() const;
    /// Number of queries answered by the fp16 gemm path.
    size_t fp16_gemm_nb_queries() const;

   private:
    // Not copied with the index: a copy rebuilds it on first search.
    struct FP16GemmState {
        std::shared_ptr<const fp16_gemm::PackedCodes> packed;
        std::mutex mutex;
        std::atomic<size_t> nb_queries{0};

        FP16GemmState() = default;
        FP16GemmState(const FP16GemmState& other)
                : nb_queries(other.nb_queries.load(std::memory_order_relaxed)) {
        }
        FP16GemmState& operator=(const FP16GemmState& other) {
            if (this != &other) {
                std::lock_guard<std::mutex> lock(mutex);
                packed.reset();
                nb_queries.store(
                        other.nb_queries.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
            }
            return *this;
        }
    };
    mutable FP16GemmState fp16_gemm_state_;

    void search_fp16_gemm(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const;
};

/** An IVF implementation where the components of the residuals are
 * encoded with a scalar quantizer. All distance computations
 * are asymmetric, so the encoded vectors are decoded and approximate
 * distances are computed.
 */

/// Search parameters for TurboQuant full types (QT_*_tq).
struct IVFSQTurboQSearchParameters : IVFSearchParameters {
    /// Query quantization bits for integer MSE pre-screening.
    /// 0 = float path (default), 1-8 = integer popcount path.
    uint8_t qb = 0;

    /// Also use integer popcount for QJL stage (requires qb > 0).
    bool int_qjl = false;
};

struct IndexIVFScalarQuantizer : IndexIVF {
    ScalarQuantizer sq;

    IndexIVFScalarQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool by_residual = true,
            bool own_invlists = true);

    IndexIVFScalarQuantizer();

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void decode_vectors(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            float* x) const override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    /* standalone codec interface */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

} // namespace faiss

#endif
