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
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/NSG.h>
#include <faiss/utils/utils.h>

namespace faiss {

/** The NSG index is a normal random-access index with a NSG
 * link structure built on top */

struct IndexNSG : Index {
    /// the link structure
    NSG nsg;

    /// the sequential storage
    bool own_fields = false;
    Index* storage = nullptr;

    /// the index is built or not
    bool is_built = false;

    /// K of KNN graph for building
    int GK = 64;

    /// indicate how to build a knn graph
    /// - 0: build NSG with brute force search
    /// - 1: build NSG with NNDescent
    char build_type = 0;

    /// parameters for nndescent
    int nndescent_S = 10;
    int nndescent_R = 100;
    int nndescent_L; // set to GK + 50
    int nndescent_iter = 10;

    explicit IndexNSG(int d = 0, int R = 32, MetricType metric = METRIC_L2);
    explicit IndexNSG(Index* storage, int R = 32);

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
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    void check_knn_graph(const idx_t* knn_graph, idx_t n, int K) const;
};

/** Flat index topped with with a NSG structure to access elements
 *  more efficiently.
 */

struct IndexNSGFlat : IndexNSG {
    IndexNSGFlat();
    IndexNSGFlat(int d, int R, MetricType metric = METRIC_L2);
};

/** PQ index topped with with a NSG structure to access elements
 *  more efficiently.
 */
struct IndexNSGPQ : IndexNSG {
    IndexNSGPQ();
    IndexNSGPQ(int d, int pq_m, int M, int pq_nbits = 8);
    void train(idx_t n, const float* x) override;
};

/** SQ index topped with with a NSG structure to access elements
 *  more efficiently.
 */
struct IndexNSGSQ : IndexNSG {
    IndexNSGSQ();
    IndexNSGSQ(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int M,
            MetricType metric = METRIC_L2);
};

} // namespace faiss
