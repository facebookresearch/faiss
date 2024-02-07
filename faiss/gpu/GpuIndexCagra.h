/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <faiss/gpu/GpuIndex.h>

namespace faiss {
namespace gpu {

class RaftCagra;

enum class graph_build_algo {
    /* Use IVF-PQ to build all-neighbors knn graph */
    IVF_PQ,
    /* Experimental, use NN-Descent to build all-neighbors knn graph */
    NN_DESCENT
};

struct GpuIndexCagraConfig : public GpuIndexConfig {
    /** Degree of input graph for pruning. */
    size_t intermediate_graph_degree = 128;
    /** Degree of output graph. */
    size_t graph_degree = 64;
    /** ANN algorithm to build knn graph. */
    graph_build_algo build_algo = graph_build_algo::IVF_PQ;
    /** Number of Iterations to run if building with NN_DESCENT */
    size_t nn_descent_niter = 20;
};

enum class search_algo {
    /** For large batch sizes. */
    SINGLE_CTA,
    /** For small batch sizes. */
    MULTI_CTA,
    MULTI_KERNEL,
    AUTO
};

enum class hash_mode { HASH, SMALL, AUTO };

struct SearchParametersCagra : SearchParameters {
    /** Maximum number of queries to search at the same time (batch size). Auto
     * select when 0.*/
    size_t max_queries = 0;

    /** Number of intermediate search results retained during the search.
     *
     *  This is the main knob to adjust trade off between accuracy and search
     * speed. Higher values improve the search accuracy.
     */
    size_t itopk_size = 64;

    /** Upper limit of search iterations. Auto select when 0.*/
    size_t max_iterations = 0;

    // In the following we list additional search parameters for fine tuning.
    // Reasonable default values are automatically chosen.

    /** Which search implementation to use. */
    search_algo algo = search_algo::AUTO;

    /** Number of threads used to calculate a single distance. 4, 8, 16, or 32.
     */
    size_t team_size = 0;

    /** Number of graph nodes to select as the starting point for the search in
     * each iteration. aka search width?*/
    size_t search_width = 1;
    /** Lower limit of search iterations. */
    size_t min_iterations = 0;

    /** Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. */
    size_t thread_block_size = 0;
    /** Hashmap type. Auto selection when AUTO. */
    hash_mode hashmap_mode = hash_mode::AUTO;
    /** Lower limit of hashmap bit length. More than 8. */
    size_t hashmap_min_bitlen = 0;
    /** Upper limit of hashmap fill rate. More than 0.1, less than 0.9.*/
    float hashmap_max_fill_rate = 0.5;

    /** Number of iterations of initial random seed node selection. 1 or more.
     */
    uint32_t num_random_samplings = 1;
    /** Bit mask used for initial random seed node selection. */
    uint64_t rand_xor_mask = 0x128394;
};

struct GpuIndexCagra : public GpuIndex {
   public:
    GpuIndexCagra(
            GpuResourcesProvider* provider,
            int dims,
            faiss::MetricType metric = faiss::METRIC_L2,
            GpuIndexCagraConfig config = GpuIndexCagraConfig());

    ~GpuIndexCagra() {}

    /// Trains CAGRA based on the given vector data
    void train(idx_t n, const float* x) override;

    void reset() {}

   protected:
    bool addImplRequiresIDs_() const {}

    void addImpl_(idx_t n, const float* x, const idx_t* ids) {}
    /// Called from GpuIndex for search
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* search_params) const override;

    /// Our configuration options
    const GpuIndexCagraConfig cagraConfig_;

    /// Instance that we own; contains the inverted lists
    std::shared_ptr<RaftCagra> index_;
};

} // namespace gpu
} // namespace faiss
