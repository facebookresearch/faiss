/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_SCALAR_QUANTIZER_H
#define FAISS_INDEX_SCALAR_QUANTIZER_H

#include <stdint.h>
#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

/**
 * The uniform quantizer has a range [vmin, vmax]. The range can be
 * the same for all dimensions (uniform) or specific per dimension
 * (default).
 */

struct IndexScalarQuantizer : IndexFlatCodes {
    /// Used to encode the vectors
    ScalarQuantizer sq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
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
            idx_t* labels) const override;

    DistanceComputer* get_distance_computer() const override;

    /* standalone codec interface */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

/** An IVF implementation where the components of the residuals are
 * encoded with a scalar quantizer. All distance computations
 * are asymmetric, so the encoded vectors are decoded and approximate
 * distances are computed.
 */

struct IndexIVFScalarQuantizer : IndexIVF {
    ScalarQuantizer sq;
    bool by_residual;

    IndexIVFScalarQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool encode_residual = true);

    IndexIVFScalarQuantizer();

    void train_residual(idx_t n, const float* x) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx) override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    /* standalone codec interface */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

} // namespace faiss

#endif
