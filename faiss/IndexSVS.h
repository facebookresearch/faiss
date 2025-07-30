/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <numeric>
#include <utility>
#include <vector>

#include <filesystem>
#include <sstream>

#include "faiss/Index.h"
#include "faiss/impl/FaissAssert.h"

namespace svs {
class DynamicVamana;
}

namespace faiss {

namespace detail {
struct SVSTempDirectory {
    std::filesystem::path root;
    std::filesystem::path config;
    std::filesystem::path graph;
    std::filesystem::path data;

    SVSTempDirectory();
    ~SVSTempDirectory();

    void write_files_to_stream(std::ostream& out) const;
    void write_stream_to_files(std::istream& in) const;
};
} // namespace detail

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

    /* Serialization and deserialization helpers */
    void serialize_impl(std::ostream& out) const;
    virtual void deserialize_impl(std::istream& in);

    /* The actual SVS implementation */
    svs::DynamicVamana* impl{nullptr};

    /* Initializes the implementation, using the provided data */
    virtual void init_impl(idx_t n, const float* x);
};

} // namespace faiss
