/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_INDEX_ADDITIVE_QUANTIZER_H
#define FAISS_INDEX_ADDITIVE_QUANTIZER_H

#include <faiss/impl/AdditiveQuantizer.h>

#include <cstdint>
#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ProductAdditiveQuantizer.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/// Abstract class for additive quantizers. The search functions are in common.
struct IndexAdditiveQuantizer : IndexFlatCodes {
    // the quantizer, this points to the relevant field in the inheriting
    // classes
    AdditiveQuantizer* aq;
    using Search_type_t = AdditiveQuantizer::Search_type_t;

    explicit IndexAdditiveQuantizer(
            idx_t d,
            AdditiveQuantizer* aq,
            MetricType metric = METRIC_L2);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /* The standalone codec interface */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;
};

/** Index based on a residual quantizer. Stored vectors are
 * approximated by residual quantization codes.
 * Can also be used as a codec
 */
struct IndexResidualQuantizer : IndexAdditiveQuantizer {
    /// The residual quantizer used to encode the vectors
    ResidualQuantizer rq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexResidualQuantizer(
            int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress);

    IndexResidualQuantizer(
            int d,
            const std::vector<size_t>& nbits,
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress);

    IndexResidualQuantizer();

    void train(idx_t n, const float* x) override;
};

struct IndexLocalSearchQuantizer : IndexAdditiveQuantizer {
    LocalSearchQuantizer lsq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexLocalSearchQuantizer(
            int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress);

    IndexLocalSearchQuantizer();

    void train(idx_t n, const float* x) override;
};

/** Index based on a product residual quantizer.
 */
struct IndexProductResidualQuantizer : IndexAdditiveQuantizer {
    /// The product residual quantizer used to encode the vectors
    ProductResidualQuantizer prq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param nsplits  number of residual quantizers
     * @param Msub      number of subquantizers per RQ
     * @param nbits  number of bit per subvector index
     */
    IndexProductResidualQuantizer(
            int d,          ///< dimensionality of the input vectors
            size_t nsplits, ///< number of residual quantizers
            size_t Msub,    ///< number of subquantizers per RQ
            size_t nbits,   ///< number of bit per subvector index
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress);

    IndexProductResidualQuantizer();

    void train(idx_t n, const float* x) override;
};

/** Index based on a product local search quantizer.
 */
struct IndexProductLocalSearchQuantizer : IndexAdditiveQuantizer {
    /// The product local search quantizer used to encode the vectors
    ProductLocalSearchQuantizer plsq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param nsplits  number of local search quantizers
     * @param Msub     number of subquantizers per LSQ
     * @param nbits  number of bit per subvector index
     */
    IndexProductLocalSearchQuantizer(
            int d,          ///< dimensionality of the input vectors
            size_t nsplits, ///< number of local search quantizers
            size_t Msub,    ///< number of subquantizers per LSQ
            size_t nbits,   ///< number of bit per subvector index
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress);

    IndexProductLocalSearchQuantizer();

    void train(idx_t n, const float* x) override;
};

/** A "virtual" index where the elements are the residual quantizer centroids.
 *
 * Intended for use as a coarse quantizer in an IndexIVF.
 */
struct AdditiveCoarseQuantizer : Index {
    AdditiveQuantizer* aq;

    explicit AdditiveCoarseQuantizer(
            idx_t d = 0,
            AdditiveQuantizer* aq = nullptr,
            MetricType metric = METRIC_L2);

    /// norms of centroids, useful for knn-search
    std::vector<float> centroid_norms;

    /// N/A
    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;
    void train(idx_t n, const float* x) override;

    /// N/A
    void reset() override;
};

/** The ResidualCoarseQuantizer is a bit specialized compared to the
 * default AdditiveCoarseQuantizer because it can use a beam search
 * at search time (slow but may be useful for very large vocabularies) */
struct ResidualCoarseQuantizer : AdditiveCoarseQuantizer {
    /// The residual quantizer used to encode the vectors
    ResidualQuantizer rq;

    /// factor between the beam size and the search k
    /// if negative, use exact search-to-centroid
    float beam_factor;

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

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    ResidualCoarseQuantizer();
};

struct LocalSearchCoarseQuantizer : AdditiveCoarseQuantizer {
    /// The residual quantizer used to encode the vectors
    LocalSearchQuantizer lsq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    LocalSearchCoarseQuantizer(
            int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2);

    LocalSearchCoarseQuantizer();
};

} // namespace faiss

#endif
