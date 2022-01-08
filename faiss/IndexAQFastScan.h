/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/impl/AdditiveQuantizer.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

/** Fast scan version of IndexAQ. Works for 4-bit AQ for now.
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

struct IndexAQFastScan : Index {
    AdditiveQuantizer* aq;
    using Search_type_t = AdditiveQuantizer::Search_type_t;

    // implementation to select
    int implem = 0;
    // skip some parts of the computation (for timing)
    int skip = 0;

    // size of the kernel
    int bbs;     // set at build time
    int qbs = 0; // query block size 0 = use default

    size_t M;
    size_t nbits;
    size_t ksub;
    size_t code_size;

    // packed version of the codes
    size_t ntotal2;
    size_t M2;

    AlignedTable<uint8_t> codes;

    // this is for testing purposes only (set when initialized by IndexAQ)
    const uint8_t* orig_codes = nullptr;

    size_t max_training_points = 0;

    IndexAQFastScan(
            AdditiveQuantizer* aq,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    void init(
            AdditiveQuantizer* aq,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexAQFastScan();

    ~IndexAQFastScan();

    /// build from an existing IndexAQ
    explicit IndexAQFastScan(const IndexAdditiveQuantizer& orig, int bbs = 32);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void reset() override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    // called by search function
    void compute_quantized_LUT(
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

    void compute_codes(uint8_t* tmp_codes, idx_t n, const float* x) const;

    void compute_LUT(float* lut, idx_t n, const float* x) const;
};

/** Index based on a residual quantizer. Stored vectors are
 * approximated by residual quantization codes.
 * Can also be used as a codec
 */
struct IndexRQFastScan : IndexAQFastScan {
    /// The residual quantizer used to encode the vectors
    ResidualQuantizer rq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexRQFastScan(
            int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_norm_rq2x4,
            int bbs = 32);

    IndexRQFastScan();
};

struct IndexLSQFastScan : IndexAQFastScan {
    LocalSearchQuantizer lsq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexLSQFastScan(
            int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_norm_lsq2x4,
            int bbs = 32);

    IndexLSQFastScan();
};

struct AQFastScanStats {
    uint64_t t0, t1, t2, t3;
    AQFastScanStats() {
        reset();
    }
    void reset() {
        memset(this, 0, sizeof(*this));
    }
};

FAISS_API extern AQFastScanStats AQFastScan_stats;

} // namespace faiss