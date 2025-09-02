/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_INDEX_IVF_ADDITIVE_QUANTIZER_H
#define FAISS_INDEX_IVF_ADDITIVE_QUANTIZER_H

#include <faiss/impl/AdditiveQuantizer.h>

#include <cstdint>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ProductAdditiveQuantizer.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/// Abstract class for IVF additive quantizers.
/// The search functions are in common.
struct IndexIVFAdditiveQuantizer : IndexIVF {
    // the quantizer
    AdditiveQuantizer* aq;
    int use_precomputed_table = 0; // for future use

    using Search_type_t = AdditiveQuantizer::Search_type_t;

    IndexIVFAdditiveQuantizer(
            AdditiveQuantizer* aq,
            Index* quantizer,
            size_t d,
            size_t nlist,
            MetricType metric = METRIC_L2,
            bool own_invlists = true);

    explicit IndexIVFAdditiveQuantizer(AdditiveQuantizer* aq);

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

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void sa_decode(idx_t n, const uint8_t* codes, float* x) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    ~IndexIVFAdditiveQuantizer() override;
};

/** IndexIVF based on a residual quantizer. Stored vectors are
 * approximated by residual quantization codes.
 */
struct IndexIVFResidualQuantizer : IndexIVFAdditiveQuantizer {
    /// The residual quantizer used to encode the vectors
    ResidualQuantizer rq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexIVFResidualQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            const std::vector<size_t>& nbits,
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress,
            bool own_invlists = true);

    IndexIVFResidualQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,     /* number of subquantizers */
            size_t nbits, /* number of bit per subvector index */
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress,
            bool own_invlists = true);

    IndexIVFResidualQuantizer();

    virtual ~IndexIVFResidualQuantizer();
};

/** IndexIVF based on a residual quantizer. Stored vectors are
 * approximated by residual quantization codes.
 */
struct IndexIVFLocalSearchQuantizer : IndexIVFAdditiveQuantizer {
    /// The LSQ quantizer used to encode the vectors
    LocalSearchQuantizer lsq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexIVFLocalSearchQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,     /* number of subquantizers */
            size_t nbits, /* number of bit per subvector index */
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress,
            bool own_invlists = true);

    IndexIVFLocalSearchQuantizer();

    virtual ~IndexIVFLocalSearchQuantizer();
};

/** IndexIVF based on a product residual quantizer. Stored vectors are
 * approximated by product residual quantization codes.
 */
struct IndexIVFProductResidualQuantizer : IndexIVFAdditiveQuantizer {
    /// The product residual quantizer used to encode the vectors
    ProductResidualQuantizer prq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param nsplits  number of residual quantizers
     * @param Msub   number of subquantizers per RQ
     * @param nbits  number of bit per subvector index
     */
    IndexIVFProductResidualQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t nsplits,
            size_t Msub,
            size_t nbits,
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress,
            bool own_invlists = true);

    IndexIVFProductResidualQuantizer();

    virtual ~IndexIVFProductResidualQuantizer();
};

/** IndexIVF based on a product local search quantizer. Stored vectors are
 * approximated by product local search quantization codes.
 */
struct IndexIVFProductLocalSearchQuantizer : IndexIVFAdditiveQuantizer {
    /// The product local search quantizer used to encode the vectors
    ProductLocalSearchQuantizer plsq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param nsplits  number of local search quantizers
     * @param Msub   number of subquantizers per LSQ
     * @param nbits  number of bit per subvector index
     */
    IndexIVFProductLocalSearchQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t nsplits,
            size_t Msub,
            size_t nbits,
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress,
            bool own_invlists = true);

    IndexIVFProductLocalSearchQuantizer();

    virtual ~IndexIVFProductLocalSearchQuantizer();
};

} // namespace faiss

#endif
