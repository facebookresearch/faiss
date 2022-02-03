/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

/** Fast scan version of IndexPQ and IndexAdditiveQuantizer. Works for 4-bit
 * sub-codes for now. This works only for search time. Training and addition are
 * pefromed by sub-classes.
 *
 *
 * The codes are not stored sequentially but grouped in blocks of size bbs.
 * This makes it possible to compute distances quickly with SIMD instructions.
 *
 * Implementations:
 * 12: blocked loop with internal loop on Q with qbs
 * 13: same with reservoir accumulator to store results
 * 14: no qbs with heap accumulator
 * 15: no qbs with reservoir accumulator
 */

struct IndexFastScan : Index {
    // implementation to select
    int implem = 0;
    // skip some parts of the computation (for timing)
    int skip = 0;

    // size of the kernel
    int bbs;     // set at build time
    int qbs = 0; // query block size 0 = use default

    // derived from quantizer
    size_t M, code_size;

    // packed version of the codes
    size_t ntotal2;
    size_t M2;

    AlignedTable<uint8_t> codes;

    // this is for testing purposes only (set when initialized by IndexPQ)
    const uint8_t* orig_codes = nullptr;

    IndexFastScan(
            int d,
            size_t M,
            size_t nbits,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexFastScan();

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    virtual void compute_float_LUT(idx_t n, const float* x, float* lut)
            const = 0;

    /// default implementation calls compute_float_LUT
    virtual void compute_quantized_LUT(
            idx_t n,
            const float* x,
            uint8_t* lut,
            float* normalizers) const;

    template <bool is_max>
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    template <class C>
    void search_implem_2(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    template <class C>
    void search_implem_12(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl) const;

    template <class C>
    void search_implem_14(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            int impl) const;
};

struct IndexPQFastScan : IndexFastScan {
    ProductQuantizer pq;

    IndexPQFastScan(
            int d,
            size_t M,
            size_t nbits,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexPQFastScan();

    /// build from an existing IndexPQ
    explicit IndexPQFastScan(const IndexPQ& orig, int bbs = 32);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;

    void compute_float_LUT(idx_t n, const float* x, float* lut) const override;
};

struct IndexResidualQuantizerFastScan : IndexFastScan {
    ResidualQuantizer rq;

    /// build from an existing IndexPQ
    explicit IndexResidualQuantizerFastScan(
            const IndexResidualQuantizer& orig,
            int bbs = 32);

    // void train(idx_t n, const float* x) override;
    // void add(idx_t n, const float* x) override;

    void compute_float_LUT(idx_t n, const float* x, float* lut) const override;

    void add(idx_t n, const float* x) override;
};

struct FastScanStats {
    uint64_t t0, t1, t2, t3;
    FastScanStats() {
        reset();
    }
    void reset() {
        memset(this, 0, sizeof(*this));
    }
};

FAISS_API extern FastScanStats FastScan_stats;

} // namespace faiss
