/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>

namespace faiss {

/** Index that queries in a base_index (a fast one) and refines the
 *  results with an exact search, hopefully improving the results.
 */
struct IndexRefine : Index {
    /// faster index to pre-select the vectors that should be filtered
    Index* base_index;

    /// refinement index
    Index* refine_index;

    bool own_fields;       ///< should the base index be deallocated?
    bool own_refine_index; ///< same with the refinement index

    /// factor between k requested in search and the k requested from
    /// the base_index (should be >= 1)
    float k_factor = 1;

    /// initialize from empty index
    IndexRefine(Index* base_index, Index* refine_index);

    IndexRefine();

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    // reconstruct is routed to the refine_index
    void reconstruct(idx_t key, float* recons) const override;

    /* standalone codec interface: the base_index codes are interleaved with the
     * refine_index ones */
    size_t sa_code_size() const override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    /// The sa_decode decodes from the index_refine, which is assumed to be more
    /// accurate
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    ~IndexRefine() override;
};

/** Version where the refinement index is an IndexFlat. It has one additional
 * constructor that takes a table of elements to add to the flat refinement
 * index */
struct IndexRefineFlat : IndexRefine {
    explicit IndexRefineFlat(Index* base_index);
    IndexRefineFlat(Index* base_index, const float* xb);

    IndexRefineFlat();

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss
