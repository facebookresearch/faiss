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

struct IndexAdditiveQuantizerFastScan : IndexFastScan {
    AdditiveQuantizer* aq;
    using Search_type_t = AdditiveQuantizer::Search_type_t;

    bool rescale_norm = true;
    int norm_scale = 1;

    // max number of training vectors
    size_t max_train_points = 0;

    explicit IndexAdditiveQuantizerFastScan(
            AdditiveQuantizer* aq,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    void init(
            AdditiveQuantizer* aq,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexAdditiveQuantizerFastScan();

    ~IndexAdditiveQuantizerFastScan() override;

    /// build from an existing IndexAQ
    explicit IndexAdditiveQuantizerFastScan(
            const IndexAdditiveQuantizer& orig,
            int bbs = 32);

    void train(idx_t n, const float* x) override;

    void estimate_norm_scale(idx_t n, const float* x);

    void compute_codes(uint8_t* codes, idx_t n, const float* x) const override;

    void compute_float_LUT(float* lut, idx_t n, const float* x) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    /** Decode a set of vectors.
     *
     *  NOTE: The codes in the IndexAdditiveQuantizerFastScan object are non-
     *        contiguous. But this method requires a contiguous representation.
     *
     * @param n       number of vectors
     * @param bytes   input encoded vectors, size n * code_size
     * @param x       output vectors, size n * d
     */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

/** Index based on a residual quantizer. Stored vectors are
 * approximated by residual quantization codes.
 * Can also be used as a codec
 */
struct IndexResidualQuantizerFastScan : IndexAdditiveQuantizerFastScan {
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
    IndexResidualQuantizerFastScan(
            int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_norm_rq2x4,
            int bbs = 32);

    IndexResidualQuantizerFastScan();
};

struct IndexLocalSearchQuantizerFastScan : IndexAdditiveQuantizerFastScan {
    LocalSearchQuantizer lsq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     * @param metric  metric type
     * @param search_type AQ search type
     */
    IndexLocalSearchQuantizerFastScan(
            int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_norm_lsq2x4,
            int bbs = 32);

    IndexLocalSearchQuantizerFastScan();
};

} // namespace faiss
