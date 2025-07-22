/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSLVQ4x4.h>

namespace faiss {

IndexSVSLVQ4x4::IndexSVSLVQ4x4(
    idx_t d, 
    MetricType metric,
    size_t num_threads,
    size_t graph_max_degree
): IndexSVS(d, metric, num_threads, graph_max_degree) {
}

IndexSVSLVQ4x4::~IndexSVSLVQ4x4() {
}

void IndexSVSLVQ4x4::init_impl(idx_t n, const float* x) {

    // TODO: support ConstSimpleDataView in SVS shared/static lib
    const auto data = svs::data::SimpleDataView<float>(const_cast<float*>(x), n, d);
    std::vector<size_t> labels(n);
    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto compressed_data = storage_type::compress(data, threadpool, 0, blocked_alloc_type{});

    svs::threads::parallel_for(
        threadpool,
        svs::threads::StaticPartition(n),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for(auto i : is) {
                labels[i] = ntotal + i;
            }
        }
    );
    ntotal += n;


    svs::index::vamana::VamanaBuildParameters build_parameters{alpha, graph_max_degree, construction_window_size, max_candidate_pool_size, prune_to, use_full_search_history};
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
          impl = new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                std::move(build_parameters),
                std::move(compressed_data), 
                std::move(labels),
                svs::DistanceIP(),
                std::move(threadpool))
              );
          break;
        case METRIC_L2:
          impl = new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                std::move(build_parameters),
                std::move(compressed_data),
                std::move(labels),
                svs::DistanceL2(),
                std::move(threadpool))
              );
          break;
        default:
          FAISS_ASSERT(!"not supported SVS distance");
    }
}

} // namespace faiss
