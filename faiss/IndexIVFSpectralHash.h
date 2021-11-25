/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_IVFSH_H
#define FAISS_INDEX_IVFSH_H

#include <vector>

#include <faiss/IndexIVF.h>

namespace faiss {

struct VectorTransform;
struct IndexPreTransform;

/** Inverted list that stores binary codes of size nbit. Before the
 * binary conversion, the dimension of the vectors is transformed from
 * dim d into dim nbit by vt (a random rotation by default).
 *
 * Each coordinate is subtracted from a value determined by
 * threshold_type, and split into intervals of size period. Half of
 * the interval is a 0 bit, the other half a 1.
 *
 */
struct IndexIVFSpectralHash : IndexIVF {
    /// transformation from d to nbit dim
    VectorTransform* vt;
    /// own the vt
    bool own_fields;

    /// nb of bits of the binary signature
    int nbit;
    /// interval size for 0s and 1s
    float period;

    enum ThresholdType {
        Thresh_global,        ///< global threshold at 0
        Thresh_centroid,      ///< compare to centroid
        Thresh_centroid_half, ///< central interval around centroid
        Thresh_median         ///< median of training set
    };
    ThresholdType threshold_type;

    /// Trained threshold.
    /// size nlist * nbit or 0 if Thresh_global
    std::vector<float> trained;

    IndexIVFSpectralHash(
            Index* quantizer,
            size_t d,
            size_t nlist,
            int nbit,
            float period);

    IndexIVFSpectralHash();

    void train_residual(idx_t n, const float* x) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs) const override;

    /** replace the vector transform for an empty (and possibly untrained) index
     */
    void replace_vt(VectorTransform* vt, bool own = false);

    /** convenience function to get the VT from an index constucted by an
     * index_factory (should end in "LSH") */
    void replace_vt(IndexPreTransform* index, bool own = false);

    ~IndexIVFSpectralHash() override;
};

} // namespace faiss

#endif
