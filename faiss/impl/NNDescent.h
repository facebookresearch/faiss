/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <algorithm>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

namespace faiss {

/** Implementation of NNDescent which is one of the most popular
 *  KNN graph building algorithms
 *
 * Efficient K-Nearest Neighbor Graph Construction for Generic
 * Similarity Measures
 *
 *  Dong, Wei, Charikar Moses, and Kai Li, WWW 2011
 *
 * This implmentation is heavily influenced by the efanna
 * implementation by Cong Fu and the KGraph library by Wei Dong
 * (https://github.com/ZJULearning/efanna_graph)
 * (https://github.com/aaalgo/kgraph)
 *
 * The NNDescent object stores only the neighbor link structure,
 * see IndexNNDescent.h for the full index object.
 */

struct VisitedTable;
struct DistanceComputer;

namespace nndescent {

struct Neighbor {
    int id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(int id, float distance, bool f)
            : id(id), distance(distance), flag(f) {}

    inline bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

struct Nhood {
    std::mutex lock;
    std::vector<Neighbor> pool; // candidate pool (a max heap)
    int M;                      // number of new neighbors to be operated

    std::vector<int> nn_old;  // old neighbors
    std::vector<int> nn_new;  // new neighbors
    std::vector<int> rnn_old; // reverse old neighbors
    std::vector<int> rnn_new; // reverse new neighbors

    Nhood() = default;

    Nhood(int l, int s, std::mt19937& rng, int N);

    Nhood& operator=(const Nhood& other);

    Nhood(const Nhood& other);

    void insert(int id, float dist);

    template <typename C>
    void join(C callback) const;
};

} // namespace nndescent

struct NNDescent {
    using storage_idx_t = int;

    using KNNGraph = std::vector<nndescent::Nhood>;

    explicit NNDescent(const int d, const int K);

    ~NNDescent();

    void build(DistanceComputer& qdis, const int n, bool verbose);

    void search(
            DistanceComputer& qdis,
            const int topk,
            idx_t* indices,
            float* dists,
            VisitedTable& vt) const;

    void reset();

    /// Initialize the KNN graph randomly
    void init_graph(DistanceComputer& qdis);

    /// Perform NNDescent algorithm
    void nndescent(DistanceComputer& qdis, bool verbose);

    /// Perform local join on each node
    void join(DistanceComputer& qdis);

    /// Sample new neighbors for each node to peform local join later
    void update();

    /// Sample a small number of points to evaluate the quality of KNNG built
    void generate_eval_set(
            DistanceComputer& qdis,
            std::vector<int>& c,
            std::vector<std::vector<int>>& v,
            int N);

    /// Evaluate the quality of KNNG built
    float eval_recall(
            std::vector<int>& ctrl_points,
            std::vector<std::vector<int>>& acc_eval_set);

    bool has_built;

    int K; // K in KNN graph
    int S; // number of sample neighbors to be updated for each node
    int R; // size of reverse links, 0 means the reverse links will not be used
    int L; // size of the candidate pool in building
    int iter;        // number of iterations to iterate over
    int search_L;    // size of candidate pool in searching
    int random_seed; // random seed for generators

    int d; // dimensions

    int ntotal;

    KNNGraph graph;
    std::vector<int> final_graph;
};

} // namespace faiss
