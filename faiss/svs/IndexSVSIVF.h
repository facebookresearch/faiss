/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2026 Intel Corporation
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

#include <faiss/Index.h>
#include <faiss/svs/IndexSVSFaissUtils.h>
#include <faiss/svs/IndexSVSVamana.h> // for SVSStorageKind, to_svs_storage_kind

#include <svs/runtime/api_defs.h>

#include <iostream>

namespace faiss {

struct SearchParametersSVSIVF : public SearchParameters {
    size_t n_probes = 0;
    float k_reorder = 0;
};

struct IndexSVSIVF : Index {
    /// Number of centroids / clusters
    size_t num_centroids = 1000;
    /// Minibatch size for k-means clustering
    size_t minibatch_size = 10000;
    /// Number of iterations for k-means clustering
    size_t num_iterations = 10;
    /// Whether to use hierarchical clustering
    bool is_hierarchical = true;
    /// Fraction of data to use for training (0.0 to 1.0)
    float training_fraction = 0.1f;
    /// Number of level-1 clusters for hierarchical clustering
    size_t hierarchical_level1_clusters = 0;
    /// Random seed for clustering
    size_t seed = 42;

    /// Number of probes for search
    size_t n_probes = 10;
    /// Reranking multiplier for compressed datasets
    float k_reorder = 1.0f;

    /// Number of threads for inter-query parallelism (0 = use all available)
    size_t num_threads = 0;
    /// Number of threads for intra-query parallelism (cluster exploration)
    size_t intra_query_threads = 1;

    SVSStorageKind storage_kind;

    IndexSVSIVF();

    IndexSVSIVF(
            idx_t d,
            size_t nlist,
            MetricType metric = METRIC_L2,
            SVSStorageKind storage = SVSStorageKind::SVS_FP32);

    ~IndexSVSIVF() override;

    // static member that exposes whether or not LVQ/LeanVec are enabled for
    // this build and runtime.
    static bool is_lvq_leanvec_enabled();

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    size_t remove_ids(const IDSelector& sel) override;

    void reset() override;

    /* Serialization and deserialization helpers */
    void serialize_impl(std::ostream& out) const;
    virtual void deserialize_impl(std::istream& in);

    /* The actual SVS implementation */
    svs_runtime::DynamicIVFIndex* impl{nullptr};

   protected:
    /* Initializes the implementation from training data */
    virtual void create_impl(idx_t n, const float* x);
};

} // namespace faiss
