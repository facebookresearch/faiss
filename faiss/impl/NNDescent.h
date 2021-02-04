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
 * implementation by Cong Fu
 * (https://github.com/ZJULearning/efanna_graph)
 *
 * The NNDescent object stores only the neighbor link structure,
 * see IndexNNDescent.h for the full index object.
 */

struct VisitedTable;
struct DistanceComputer;
typedef std::lock_guard<std::mutex> LockGuard;

namespace nndescent {

struct Neighbor {
  int id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(int id, float distance, bool f)
      : id(id), distance(distance), flag(f) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

struct Nhood {
  std::mutex lock;
  std::vector<Neighbor> pool;
  int M;

  std::vector<int> nn_old;
  std::vector<int> nn_new;
  std::vector<int> rnn_old;
  std::vector<int> rnn_new;

  Nhood() = default;

  Nhood(int l, int s, std::mt19937 &rng, int N);

  Nhood &operator=(const Nhood &other);

  Nhood(const Nhood &other);

  void insert(int id, float dist);

  template <typename C> void join(C callback) const;
};

} // namespace nndescent

struct NNDescent {

  using storage_idx_t = int;
  using idx_t = Index::idx_t;

  using KNNGraph = std::vector<nndescent::Nhood>;

  explicit NNDescent(const int d, const int K);

  ~NNDescent();

  void build(DistanceComputer &qdis, const int n, bool verbose);

  void search(DistanceComputer &qdis, const int topk, idx_t *indices,
              float *dists, VisitedTable &vt) const;

  void reset();

  void init_graph(DistanceComputer &qdis);

  void nndescent(DistanceComputer &qdis, bool verbose);

  void join(DistanceComputer &qdis);

  void update();

  void generate_eval_set(DistanceComputer &qdis, std::vector<int> &c,
                         std::vector<std::vector<int>> &v, int N);

  float eval_recall(std::vector<int> &ctrl_points,
                    std::vector<std::vector<int>> &acc_eval_set);

  bool has_built;

  int K;
  int S, R, L, iter;
  int search_L;

  // dimensions
  int d;

  int ntotal;

  mutable std::mt19937 rng;

  KNNGraph graph;
  std::vector<int> final_graph;
};

} // namespace faiss
