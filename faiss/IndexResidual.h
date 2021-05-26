/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_INDEX_RESIDUAL_H
#define FAISS_INDEX_RESIDUAL_H

#include <stdint.h>

#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/** Index based on a residual quantizer. Stored vectors are
 * approximated by residual quantization codes.
 * Can also be used as a codec
 */
struct IndexResidual : Index {
    /// The residual quantizer used to encode the vectors
    ResidualQuantizer rq;

    enum Search_type_t {
        ST_decompress, ///< decompress database vector
        ST_LUT_nonorm, ///< use a LUT, don't include norms (OK for IP or
                       ///< normalized vectors)
        ST_norm_float, ///< use a LUT, and store float32 norm with the vectors
        ST_norm_qint8, ///< use a LUT, and store 8bit-quantized norm
    };
    Search_type_t search_type;

    /// min/max for quantization of norms
    float norm_min, norm_max;

    /// size of residual quantizer codes + norms
    size_t code_size;

    /// Codes. Size ntotal * rq.code_size
    std::vector<uint8_t> codes;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexResidual(
            int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2,
            Search_type_t search_type = ST_decompress);

    IndexResidual(
            int d,
            const std::vector<size_t>& nbits,
            MetricType metric = METRIC_L2,
            Search_type_t search_type = ST_decompress);

    IndexResidual();

    /// set search type and update parameters
    void set_search_type(Search_type_t search_type);

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    /// not implemented
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    void reset() override;

    /* The standalone codec interface */
    size_t sa_code_size() const override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    //    DistanceComputer* get_distance_computer() const override;
};

/** A "virtual" index where the elements are the residual quantizer centroids.
 *
 * Intended for use as a coarse quantizer in an IndexIVF.
 */
struct ResidualCoarseQuantizer : Index {
    /// The residual quantizer used to encode the vectors
    ResidualQuantizer rq;

    /// factor between the beam size and the search k
    /// if negative, use exact search-to-centroid
    float beam_factor;

    /// norms of centroids, useful for knn-search
    std::vector<float> centroid_norms;

    /// computes centroid norms if required
    void set_beam_factor(float new_beam_factor);

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    ResidualCoarseQuantizer(
            int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2);

    ResidualCoarseQuantizer(
            int d,
            const std::vector<size_t>& nbits,
            MetricType metric = METRIC_L2);

    ResidualCoarseQuantizer();

    void train(idx_t n, const float* x) override;

    /// N/A
    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /// N/A
    void reset() override;
};

} // namespace faiss

#endif
