/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSFlat.h>
#include "faiss/Index.h"

namespace faiss {

IndexSVSFlat::IndexSVSFlat(
    idx_t d, 
    MetricType metric,
    idx_t num_threads,
):Index(d, metric), num_threads{num_threads} {
}

void IndexSVSFlat::add(idx_t n, const float* x) {

    if(!impl) {
        init_impl(n, x);
        return;
    }

    // construct sequential labels
    std::vector<size_t> labels(n);

    svs::threads::parallel_for(
        impl->get_threadpool_handle(),
        svs::threads::StaticPartition(n),
        [&](auto is, auto SVS_UNUSED(tid)) {
            for(auto i : is) {
                labels[i] = nlabels + i;
            }
        }
    );

    auto data = svs::data::ConstSimpleDataView<float>(x, n, d);
    impl->add_points(data, labels);
}

void IndexSVSFlat::reset() {
    impl.reset();
}

IndexSVSFlat::~IndexSVSFlat() {
}

void IndexSVSFlat::search(
    idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    const SearchParameters* params
) const {
    FAISS_THROW_IF_NOT(k > 0);

    auto queries = svs::data::ConstSimpleDataView<float>(x, n, d);

    auto results = svs::QueryResult<size_t>{queries.size(), k};
    // TODO: Tuable Flat parameters 
    impl->search(results.view(), queries, {});

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

void IndexSVSFlat::init_impl(idx_t n, const float* x) {
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

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
          impl = std::make_unique<svs::index::flat::FlatIndex>(
                std::move(data), 
                svs::DistanceIP(),
                std::move(threadpool)
              );
          break;
        case METRIC_L2:
          impl = std::make_unique<svs::index::flat::FlatIndex>(
                std::move(data),
                svs::DistanceL2(),
                std::move(threadpool)
              );
          break;
        default:
          FAISS_ASSERT(!"not supported SVS distance");
    }
}

} // namespace faiss
