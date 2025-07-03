/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSUncompressed.h>
#include "faiss/Index.h"

namespace faiss {

IndexSVSUncompressed::IndexSVSUncompressed(
    idx_t d, 
    MetricType metric,
    idx_t num_threads,
    idx_t graph_max_degree
):Index(d, metric), num_threads{num_threads}, graph_max_degree{graph_max_degree} {
}

void IndexSVSUncompressed::add(idx_t n, const float* x) {

    // construct sequential labels
    std::vector<size_t> labels(n);
    std::iota(labels.begin(), labels.end(), nlabels);
    nlabels += n;


    if(!impl) {
        init_impl(n, x, labels);
        return;
    }

    auto data = svs::data::ConstSimpleDataView<float>(x, n, d);
    impl->add_points(data, labels);
}

void IndexSVSUncompressed::reset() {
    impl.reset();
    nlabels = 0;
}

IndexSVSUncompressed::~IndexSVSUncompressed() {
}

void IndexSVSUncompressed::search(
    idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    const SearchParameters* params
) const {
    FAISS_THROW_IF_NOT(k > 0);

    auto queries = svs::data::ConstSimpleDataView<float>(x, n, d);

    // TODO: use params for SVS search parameters
    auto sp = impl->get_search_parameters();
    sp.buffer_config({search_window_size, search_buffer_capacity});
    auto results = svs::QueryResult<size_t>{queries.size(), k};
    impl->search(results.view(), queries, sp);

    svs::threads::parallel_for(
        impl->get_threadpool_handle(),
        svs::threads::StaticPartition(n),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for(auto i : is) {
                for(idx_t j = 0; j < k; ++j) {
                    labels[j + i * k] = results.index(i, j);
                    distances[j + i * k] = results.distance(i, j);
                }
            }
        }
    );
}

void IndexSVSUncompressed::init_impl(idx_t n, const float* x, const std::vector<size_t>& labels) {
    auto data = svs::data::SimpleData<float>(n, d);

    auto threadpool = svs::threads::as_threadpool(num_threads);

    svs::threads::parallel_for(
        threadpool,
        svs::threads::StaticPartition(n),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for(auto i : is) {
                data.set_datum(i, std::span<const float>(x + i * d, d));
            }
        }
    );

    svs::index::vamana::VamanaBuildParameters build_parameters{alpha, graph_max_degree, construction_window_size, max_candidate_pool_size, prune_to, use_full_search_history};

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
          impl = std::make_unique<svs::DynamicVamana>(svs::DynamicVamana::build<float>(build_parameters, std::move(data), labels, svs::DistanceIP(), std::move(threadpool)));
          break;
        case METRIC_L2:
          impl = std::make_unique<svs::DynamicVamana>(svs::DynamicVamana::build<float>(build_parameters, std::move(data), labels, svs::DistanceL2(), std::move(threadpool)));
          break;
        default:
          FAISS_ASSERT(!"not supported SVS distance");
    }
}

} // namespace faiss
