/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_INDEX_SQ_FP16_GEMM_H
#define FAISS_INDEX_SQ_FP16_GEMM_H

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/sq-fp16-gemm.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace faiss {

/** A QT_fp16 IndexScalarQuantizer specialized for the case where every
 * database vector is scored against every query, as when it is used as an
 * IVF coarse quantizer.
 *
 * Instead of going through ScalarQuantizer::get_distance_computer() one
 * database vector at a time, search() packs the fp16 codes once and runs a
 * dedicated small-matrix-multiply kernel (scalar, AVX-512/F16C/FMA, or
 * AMX-FP16, chosen at runtime, see fp16_gemm::GemmKernel) that computes
 * every query/database inner product directly from the fp16 bit patterns.
 * By default the kernel's output is only used to pick the top-k candidates;
 * rescore_native then recomputes their final distances through the regular
 * ScalarQuantizer distance computer, so returned distances always match
 * what plain IndexScalarQuantizer would have returned bit-for-bit.
 *
 * Only QT_fp16 is supported, and every float value going through the
 * public float interface (train, add, search) is first rounded to fp16.
 * Concurrent read-only search() calls are safe; callers must otherwise
 * serialize search() against anything that changes codes, kernel or
 * metric_type.
 */
struct IndexSQFP16Gemm : IndexScalarQuantizer {
    fp16_gemm::GemmKernel kernel = fp16_gemm::GemmKernel::AUTO;
    bool use_gemm = true;
    size_t bs_x = 256;
    size_t bs_y = 1024;
    bool rescore_native = true;
    mutable std::atomic<size_t> nb_gemm_queries{0};

    explicit IndexSQFP16Gemm(int d, MetricType metric = METRIC_INNER_PRODUCT);
    IndexSQFP16Gemm();

    void train(idx_t n, const float* x) override;
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void add(idx_t n, const float* x) override;
    void reset() override;
    void add_sa_codes(idx_t n, const uint8_t* codes_in, const idx_t* xids)
            override;
    size_t remove_ids(const IDSelector& sel) override;
    void merge_from(Index& other, idx_t add_id = 0) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Must be called after codes are modified directly; invalidate_pack()
    /// is cheaper when the cache will be rebuilt lazily on next search.
    void repack();
    void invalidate_pack();
    fp16_gemm::GemmKernel effective_kernel() const;
    size_t packed_bytes() const;

   private:
    struct PackedCache {
        int d = 0;
        idx_t ntotal = 0;
        MetricType metric = METRIC_INNER_PRODUCT;
        size_t bs_y = 0;
        const uint8_t* source_codes = nullptr;
        std::vector<uint16_t> rows;
        std::vector<uint16_t> packed;
        std::vector<size_t> offset;
        std::vector<float> norms;
    };
    mutable std::shared_ptr<const PackedCache> cache_;
    mutable std::mutex cache_mutex_;

    void check_format() const;
    std::shared_ptr<const PackedCache> ensure_packed() const;
    std::shared_ptr<const PackedCache> build_pack() const;
    void rescore(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    template <class C>
    void knn_gemm(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const PackedCache& cache) const;
};

} // namespace faiss

#endif
