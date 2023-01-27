/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/impl/NNDescent.h>
#include <faiss/utils/utils.h>

namespace faiss {

/** The NNDescent index is a normal random-access index with an NNDescent
 * link structure built on top */

struct IndexNNDescent : Index {
    // internal storage of vectors (32 bits)
    using storage_idx_t = NNDescent::storage_idx_t;

    /// Faiss results are 64-bit

    // the link strcuture
    NNDescent nndescent;

    // the sequential storage
    bool own_fields;
    Index* storage;

    explicit IndexNNDescent(
            int d = 0,
            int K = 32,
            MetricType metric = METRIC_L2);
    explicit IndexNNDescent(Index* storage, int K = 32);

    ~IndexNNDescent() override;

    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;
};

/** Flat index topped with with a NNDescent structure to access elements
 *  more efficiently.
 */

struct IndexNNDescentFlat : IndexNNDescent {
    IndexNNDescentFlat();
    IndexNNDescentFlat(int d, int K, MetricType metric = METRIC_L2);
};

} // namespace faiss
