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

#include "svs/index/flat/flat.h"

namespace faiss {

struct IndexSVSFlatFlat : IndexSVS {

  IndexSVSFlat(
      idx_t d, 
      MetricType metric = METRIC_L2,
      idx_t num_threads = 32,
      idx_t graph_max_degree = 64
  );

  ~IndexSVSFlat() override;

  void add(idx_t n, const float* x) override;

  void search(
      idx_t n,
      const float* x,
      idx_t k,
      float* distances,
      idx_t* labels,
      const SearchParameters* params = nullptr) const override;

  void reset();

  virtual void init_impl(idx_t n, const float* x);

  // sequential labels
  size_t nlabels{0};

  std::unique_ptr<svs::DynamicVamana> impl;


  size_t num_threads;
};

} // namespace faiss
