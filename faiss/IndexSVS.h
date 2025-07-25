/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <numeric>
#include <vector>

#include "faiss/Index.h"
#include "faiss/impl/FaissAssert.h"

namespace svs {
class DynamicVamana;
}

namespace faiss {

struct IndexSVS : Index {
    // sequential labels
    size_t nlabels{0};

    // default parameters
    size_t num_threads = 1;
    size_t graph_max_degree = 64;
    float alpha = 1.2;
    size_t search_window_size = 10;
    size_t search_buffer_capacity = 10;
    size_t construction_window_size = 40;
    size_t max_candidate_pool_size = 200;
    size_t prune_to = 60;
    bool use_full_search_history = true;

    IndexSVS();

    IndexSVS(idx_t d, MetricType metric = METRIC_L2);

    virtual ~IndexSVS() override;

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    void serialize_impl(std::ostream& out) const;
    void deserialize_impl(std::istream& in);

   protected:
    svs::DynamicVamana* impl{nullptr};

   private:
    virtual void init_impl(idx_t n, const float* x);
};

} // namespace faiss
