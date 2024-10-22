/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/VectorTransform.h>

namespace faiss {

/** An IVF index with a quantizer that has a different input dimension from the
 * payload size. The vectors to encode are obtained from the input vectors by a
 * VectorTransform.
 */
struct IndexIVFIndependentQuantizer : Index {
    /// quantizer is fed directly with the input vectors
    Index* quantizer = nullptr;

    /// transform before the IVF vectors are applied
    VectorTransform* vt = nullptr;

    /// the IVF index, controls nlist and nprobe
    IndexIVF* index_ivf = nullptr;

    /// whether *this owns the 3 fields
    bool own_fields = false;

    IndexIVFIndependentQuantizer(
            Index* quantizer,
            IndexIVF* index_ivf,
            VectorTransform* vt = nullptr);

    IndexIVFIndependentQuantizer() {}

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    ~IndexIVFIndependentQuantizer() override;
};

} // namespace faiss
