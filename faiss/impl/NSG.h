/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

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
 * implementation by Cong Fu
 * (https://github.com/zjulearning/nsg)
 *
 * The NSG object stores only the neighbor link structure, see
 * IndexNSG.h for the full index object.
 */

struct VisitedTable;
struct DistanceComputer; // from AuxIndexStructures
struct Neighbor;
struct SimpleNeighbor;


struct NSG {
  /// internal storage of vectors (32 bits: this is expensive)
  using storage_idx_t = int;

  /// Faiss results are 64-bit
  using idx_t = Index::idx_t;

  int ntotal; // nb of nodes
  int R;      // nb of neighbors per node
  int L;      // expansion factor at construction time
  int C;      // candidate pool size

  int search_L; // expansion factor at search time

  int width; // max degree, do not assigned it directly!!!

  int enter_point; // enterpoint

  std::vector<std::vector<int>> final_graph;
  std::vector<int> compact_graph;
  bool is_built;

  explicit NSG(int R = 32);

  void build(Index *storage, idx_t n, int *knn_graph, int GK);

  void reset();

  /// search interface
  void search(DistanceComputer &dis, int k, idx_t *I, float *D,
              VisitedTable &vt) const;

  void init_graph(Index *storage, int *knn_graph, int GK);

  template <bool collect_fullset>
  void search_on_graph(const int *graph, int GK, DistanceComputer &dis,
                       VisitedTable &vt, int ep, int pool_size,
                       std::vector<Neighbor> &retset,
                       std::vector<Neighbor> &fullset) const;

  void add_reverse_links(int q, std::vector<std::mutex> &locks,
                         DistanceComputer &dis,
                         SimpleNeighbor *graph);

  void sync_prune(int q, std::vector<Neighbor> &pool,DistanceComputer &dis,
                  VisitedTable &vt, const int *knn_graph, int GK,
                  SimpleNeighbor *graph) ;

  void link(Index *storage, int *knn_graph, int GK, SimpleNeighbor *graph);

  void tree_grow(Index *storage);

  void dfs(VisitedTable &vt, int root, int &cnt);

  void find_root(Index *storage, VisitedTable &vt, int &root);

  void compact(int width);
};

} // namespace faiss
