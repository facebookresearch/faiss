/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <numeric>

#include "faiss/Index.h"
#include <faiss/impl/FaissAssert.h>

#include "svs/orchestrators/vamana.h"
#include "svs/orchestrators/dynamic_vamana.h"

namespace faiss {

struct IndexSVSUncompressed : Index {

  IndexSVSUncompressed(
      idx_t d, 
      MetricType metric = METRIC_L2,
      idx_t num_threads = 32,
      idx_t graph_max_degree = 64
  );

  ~IndexSVSUncompressed() override;

  void add(idx_t n, const float* x) override;

  void search(
      idx_t n,
      const float* x,
      idx_t k,
      float* distances,
      idx_t* labels,
      const SearchParameters* params = nullptr) const override;

  void reset();

  private:

    void init_impl(idx_t n, const float* x, const std::vector<size_t>& labels);

    // sequential labels
    size_t nlabels{0};

    std::unique_ptr<svs::DynamicVamana> impl;


    size_t num_threads;
    // default parameters
    // can be tuned by providing APIs
    idx_t graph_max_degree = 64;
    float alpha = 1.2;
    idx_t search_window_size = 10;
    idx_t search_buffer_capacity = 10;
    idx_t construction_window_size = 40;
    idx_t max_candidate_pool_size = 200;
    idx_t prune_to = 60;
    bool use_full_search_history = true;
};

} // namespace faiss
