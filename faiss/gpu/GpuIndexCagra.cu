/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/impl/RaftCagra.cuh>
#include "GpuIndexCagra.h"

namespace faiss {
namespace gpu {

GpuIndexCagra::GpuIndexCagra(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        GpuIndexCagraConfig config)
        : GpuIndex(provider->getResources(), dims, metric, 0.0f, config),
          cagraConfig_(config) {}

void GpuIndexCagra::train(idx_t n, const float* x) {
    if (this->is_trained) {
        FAISS_ASSERT(index_);
        return;
    }

    FAISS_ASSERT(!index_);

    index_ = std::make_shared<RaftCagra>(
            this->resources_.get(),
            this->d,
            cagraConfig_.intermediate_graph_degree,
            cagraConfig_.graph_degree,
            static_cast<faiss::cagra_build_algo>(cagraConfig_.build_algo),
            cagraConfig_.nn_descent_niter,
            this->metric_type,
            this->metric_arg);

    index_->train(n, x);

    this->is_trained = true;
    this->ntotal += n;
}

void GpuIndexCagra::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* search_params) const {
    FAISS_ASSERT(this->is_trained && index_);
    FAISS_ASSERT(n > 0);

    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, this->d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<idx_t, 2, true> outLabels(const_cast<idx_t*>(labels), {n, k});

    auto params = dynamic_cast<const SearchParametersCagra*>(search_params);

    index_->search(
            queries,
            k,
            outDistances,
            outLabels,
            params->max_queries,
            params->itopk_size,
            params->max_iterations,
            static_cast<faiss::cagra_search_algo>(params->algo),
            params->team_size,
            params->search_width,
            params->min_iterations,
            params->thread_block_size,
            static_cast<faiss::cagra_hash_mode>(params->hashmap_mode),
            params->hashmap_min_bitlen,
            params->hashmap_max_fill_rate,
            params->num_random_samplings,
            params->rand_xor_mask);
}

} // namespace gpu
} // namespace faiss