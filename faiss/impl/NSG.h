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
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
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
struct Node;

namespace nsg {

/***********************************************************
 * Graph structure to store a graph.
 *
 * It is represented by an adjacency matrix `data`, where
 * data[i, j] is the j-th neighbor of node i.
 ***********************************************************/

template <class node_t>
struct Graph {
  node_t *data;     // the flattened adjacency matrix
  int K;            // nb of neighbors per node
  int N;            // total nb of nodes
  bool own_fields;  // the underlying data owned by itself or not

  // construct from a known graph
  Graph(node_t *data, int N, int K)
      : data(data), N(N), K(K), own_fields(false) {}

  // construct an empty graph
  // NOTE: the newly allocated data needs to be destroyed at destruction time
  Graph(int N, int K) : N(N), K(K), own_fields(true) {
    data = new node_t[N * K];
  }

  // release the allocated memory if needed
  ~Graph() {
    if (own_fields) {
      delete[] data;
    }
  }

  // access the j-th neighbor of node i
  inline node_t at(int i, int j) const { return data[i * K + j]; }

  // access the j-th neighbor of node i by reference
  inline node_t &at(int i, int j) { return data[i * K + j]; }
};

DistanceComputer *storage_distance_computer(const Index *storage);

} // namespace nsg

struct NSG {

  /// internal storage of vectors (32 bits: this is expensive)
  using storage_idx_t = int;

  /// Faiss results are 64-bit
  using idx_t = Index::idx_t;

  // It needs to be smaller than 0
  static const int EMPTY_ID = -1;

  // nb of nodes
  int ntotal;

  // construction-time parameters
  int R;      // nb of neighbors per node
  int L;      // length of the search path at construction time
  int C;      // candidate pool size at construction time

  // search-time parameters
  int search_L;   // length of the search path

  // enterpoint
  int enterpoint;

  // the built graph structure
  std::shared_ptr<nsg::Graph<int>> final_graph;

  // NSG is built or not
  bool is_built;

  // random generator
  mutable RandomGenerator rng;

  explicit NSG(int R = 32);

  // build NSG from a KNN graph
  void build(Index *storage, idx_t n, const nsg::Graph<idx_t> &knn_graph,
             bool verbose);

  // reset the graph
  void reset();

  // search interface
  void search(DistanceComputer &dis, int k, idx_t *I, float *D,
              VisitedTable &vt) const;

  // Compute the center point
  void init_graph(Index *storage, const nsg::Graph<idx_t> &knn_graph);

  // Search on a built graph.
  // If collect_fullset is true, the visited nodes will be
  // collected in `fullset`.
  template <bool collect_fullset, class index_t>
  void search_on_graph(const nsg::Graph<index_t> &graph, DistanceComputer &dis,
                       VisitedTable &vt, int ep, int pool_size,
                       std::vector<Neighbor> &retset,
                       std::vector<Node> &fullset) const;

  // Add reverse links
  void add_reverse_links(int q, std::vector<std::mutex> &locks,
                         DistanceComputer &dis, nsg::Graph<Node> &graph);

  void sync_prune(int q, std::vector<Node> &pool, DistanceComputer &dis,
                  VisitedTable &vt, const nsg::Graph<idx_t> &knn_graph,
                  nsg::Graph<Node> &graph);

  void link(Index *storage, const nsg::Graph<idx_t> &knn_graph,
            nsg::Graph<Node> &graph);

  // make NSG be fully connected
  int tree_grow(Index *storage);

  // count the size of connected component
  // using depth first search start by root
  int dfs(VisitedTable &vt, int root) const;

  // attach one unlinked node
  void attach_unlinked(Index *storage, VisitedTable &vt);

  // check the integrity of the NSG built
  void check_graph() const;
};

} // namespace faiss
