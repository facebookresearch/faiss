/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexFastScan.h>
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

struct IndexAQFastScan : IndexFastScan {
    AdditiveQuantizer* aq;
    using Search_type_t = AdditiveQuantizer::Search_type_t;

    size_t max_train_points = 0;

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

    void compute_codes(uint8_t* codes, idx_t n, const float* x) const override;

    void compute_float_LUT(float* lut, idx_t n, const float* x) const override;
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
     * @param metric  metric type
     * @param search_type AQ search type
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
     * @param metric  metric type
     * @param search_type AQ search type
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

} // namespace faiss