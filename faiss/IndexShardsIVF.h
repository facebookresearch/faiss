/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/IndexShards.h>

namespace faiss {

/**
 * IndexShards with a common coarse quantizer. All the indexes added should be
 * IndexIVFInterface indexes so that the search_precomputed can be called.
 */
struct IndexShardsIVF : public IndexShards, Level1Quantizer {
    explicit IndexShardsIVF(
            Index* quantizer,
            size_t nlist,
            bool threaded = false,
            bool successive_ids = true);

    void addIndex(Index* index) override;

    void add_with_ids(idx_t n, const component_t* x, const idx_t* xids)
            override;

    void train(idx_t n, const component_t* x) override;

    void search(
            idx_t n,
            const component_t* x,
            idx_t k,
            distance_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss
