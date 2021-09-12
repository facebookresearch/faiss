/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/// Abstract class for IVF additive quantizers.
/// The search functions are in common.
struct IndexIVFAdditiveQuantizer : IndexIVF {
    // the quantizer
    AdditiveQuantizer* aq;
    bool by_residual = true;
    int use_precomputed_table = 0; // for future use

    using Search_type_t = AdditiveQuantizer::Search_type_t;

    IndexIVFAdditiveQuantizer(
            AdditiveQuantizer* aq,
            Index* quantizer,
            size_t d,
            size_t nlist,
            MetricType metric = METRIC_L2);

    explicit IndexIVFAdditiveQuantizer(AdditiveQuantizer* aq);

    void train_residual(idx_t n, const float* x) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs) const override;

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
            Search_type_t search_type = AdditiveQuantizer::ST_decompress);

    IndexIVFResidualQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,     /* number of subquantizers */
            size_t nbits, /* number of bit per subvector index */
            MetricType metric = METRIC_L2,
            Search_type_t search_type = AdditiveQuantizer::ST_decompress);

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
            Search_type_t search_type = AdditiveQuantizer::ST_decompress);

    IndexIVFLocalSearchQuantizer();

    virtual ~IndexIVFLocalSearchQuantizer();
};

} // namespace faiss

#endif
