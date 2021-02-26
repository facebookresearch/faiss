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
#include <faiss/impl/NSG.h>
#include <faiss/utils/utils.h>

namespace faiss {

/** The NSG index is a normal random-access index with a NSG
 * link structure built on top */

struct IndexNSG : Index {
    using storage_idx_t = NSG::storage_idx_t;

    // the link strcuture
    NSG nsg;

    // the sequential storage
    bool own_fields;
    Index* storage;

    // the index is built or not
    bool is_built;

    explicit IndexNSG(int d = 0, int M = 32, MetricType metric = METRIC_L2);
    explicit IndexNSG(Index* storage, int M = 32);

    ~IndexNSG() override;

    void build(idx_t n, const float* x, idx_t* knn_graph, int GK);

    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;
};

/** Flat index topped with with a NSG structure to access elements
 *  more efficiently.
 */

struct IndexNSGFlat : IndexNSG {
    IndexNSGFlat();
    IndexNSGFlat(int d, int R, MetricType metric = METRIC_L2);
};

} // namespace faiss
