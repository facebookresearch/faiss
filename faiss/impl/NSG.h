/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

namespace faiss {

/** Implementation of the Navigating Spreading-out Graph (NSG)
 * datastructure.
 *
 * Fast Approximate Nearest Neighbor Search With The
 * Navigating Spreading-out Graph
 *
 *  Cong Fu, Chao Xiang, Changxu Wang, Deng Cai, VLDB 2019
 *
 * This implmentation is heavily influenced by the NSG
 * implementation by ZJULearning Group
 * (https://github.com/zjulearning/nsg)
 *
 * The NSG object stores only the neighbor link structure, see
 * IndexNSG.h for the full index object.
 */

struct DistanceComputer; // from AuxIndexStructures
struct Neighbor;
struct SimpleNeighbor;

namespace nsg {

/**************************************************************
 * Auxiliary structures
 **************************************************************/

template <class node_t>
struct Graph {
  node_t *data;
  int K;
  int N;
  bool own_fields;

  Graph(node_t *data, int N, int K)
      : data(data), N(N), K(K), own_fields(false) {}

  Graph(int N, int K) : N(N), K(K), own_fields(true) {
    data = new node_t[N * K];
  }

  ~Graph() {
    if (own_fields)
      delete[] data;
  }

  inline node_t at(int i, int j) const { return data[i * K + j]; }

  inline node_t &at(int i, int j) { return data[i * K + j]; }
};

/// set implementation optimized for fast access.
struct VisitedTable {
  std::vector<uint8_t> visited;
  int visno;

  explicit VisitedTable(int size) : visited(size), visno(1) {}

  /// set flog #no to true
  void set(int no) { visited[no] = visno; }

  /// get flag #no
  bool get(int no) const { return visited[no] == visno; }

  /// reset all flags to false
  void advance() {
    visno++;
    if (visno == 250) {
      // 250 rather than 255 because sometimes we use visno and visno+1
      memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
      visno = 1;
    }
  }
};

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {

  /// owned by this
  DistanceComputer *basedis;

  explicit NegativeDistanceComputer(DistanceComputer *basedis)
      : basedis(basedis) {}

  void set_query(const float *x) override { basedis->set_query(x); }

  /// compute distance of vector i to current query
  float operator()(idx_t i) override { return -(*basedis)(i); }

  /// compute distance between two stored vectors
  float symmetric_dis(idx_t i, idx_t j) override {
    return -basedis->symmetric_dis(i, j);
  }

  virtual ~NegativeDistanceComputer() { delete basedis; }
};

DistanceComputer *storage_distance_computer(const Index *storage);

}  // namespace faiss::nsg

struct NSG {

  /// internal storage of vectors (32 bits: this is expensive)
  using storage_idx_t = int;

  /// Faiss results are 64-bit
  using idx_t = Index::idx_t;

  // It needs to be smaller than 0
  static const int EMPTY_ID = -1;

  int ntotal; // nb of nodes
  int R;      // nb of neighbors per node
  int L;      // expansion factor at construction time
  int C;      // candidate pool size

  int search_L;   // expansion factor at search time
  int enterpoint; // enterpoint

  std::shared_ptr<nsg::Graph<int>> final_graph;
  bool is_built;

  explicit NSG(int R = 32);

  void build(Index *storage, idx_t n, const nsg::Graph<idx_t> &knn_graph, bool verbose);

  void reset();

  /// search interface
  void search(DistanceComputer &dis, int k, idx_t *I, float *D,
              nsg::VisitedTable &vt) const;

  void init_graph(Index *storage, const nsg::Graph<idx_t> &knn_graph);

  template <bool collect_fullset, class index_t>
  void search_on_graph(const nsg::Graph<index_t> &graph, DistanceComputer &dis,
                       nsg::VisitedTable &vt, int ep, int pool_size,
                       std::vector<Neighbor> &retset,
                       std::vector<Neighbor> &fullset) const;

  void add_reverse_links(int q, std::vector<std::mutex> &locks,
                         DistanceComputer &dis,
                         nsg::Graph<SimpleNeighbor> &graph);

  void sync_prune(int q, std::vector<Neighbor> &pool, DistanceComputer &dis,
                  nsg::VisitedTable &vt, const nsg::Graph<idx_t> &knn_graph,
                  nsg::Graph<SimpleNeighbor> &graph);

  void link(Index *storage, const nsg::Graph<idx_t> &knn_graph,
            nsg::Graph<SimpleNeighbor> &graph);

  void tree_grow(Index *storage);

  void dfs(nsg::VisitedTable &vt, int root, int &cnt);

  void find_root(Index *storage, nsg::VisitedTable &vt, int &root);

  void check_graph();
};

} // namespace faiss
