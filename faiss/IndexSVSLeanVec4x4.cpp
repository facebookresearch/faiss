/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSLeanVec4x4.h>

namespace faiss {

IndexSVSLeanVec4x4::IndexSVSLeanVec4x4(
        idx_t d,
        MetricType metric,
        size_t num_threads,
        size_t graph_max_degree,
        size_t leanvec_dims)
        : IndexSVS(d, metric, num_threads, graph_max_degree) {
    leanvec_d = leanvec_dims == 0 ? d / 2 : leanvec_d;
    is_trained = false;
}

IndexSVSLeanVec4x4::~IndexSVSLeanVec4x4() {}

void IndexSVSLeanVec4x4::train(idx_t n, const float* x) {
    auto data = svs::data::ConstSimpleDataView<float>(x, n, d);
    auto means = utils::compute_medioid(data, num_threads);
    leanvec_matrix = compute_leanvec_matrix<svs::Dynamic, svs::Dynamic>(
            data, means, num_threads, leanvec_d);
    is_trained = true;
}

void IndexSVSLeanVec4x4::reset() {
    is_trained = false;
    IndexSVS::reset();
}

void IndexSVSLeanVec4x4::init_impl(idx_t n, const float* x) {
    // TODO: support ConstSimpleDataView in SVS shared/static lib
    const auto data =
            svs::data::SimpleDataView<float>(const_cast<float*>(x), n, d);
    std::vector<size_t> labels(n);
    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto compressed_data = storage_type::reduce(
            data,
            leanvec_matrix,
            threadpool,
            0,
            leanvec_d,
            blocked_alloc_type{});

    svs::threads::parallel_for(
            threadpool,
            svs::threads::StaticPartition(n),
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    labels[i] = ntotal + i;
                }
            });
    ntotal += n;

    svs::index::vamana::VamanaBuildParameters build_parameters{
            alpha,
            graph_max_degree,
            construction_window_size,
            max_candidate_pool_size,
            prune_to,
            use_full_search_history};

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            impl = std::make_unique<svs::DynamicVamana>(
                    svs::DynamicVamana::build<float>(
                            std::move(build_parameters),
                            std::move(compressed_data),
                            std::move(labels),
                            svs::DistanceIP(),
                            std::move(threadpool)));
            break;
        case METRIC_L2:
            impl = std::make_unique<svs::DynamicVamana>(
                    svs::DynamicVamana::build<float>(
                            std::move(build_parameters),
                            std::move(compressed_data),
                            std::move(labels),
                            svs::DistanceL2(),
                            std::move(threadpool)));
            break;
        default:
            FAISS_ASSERT(!"not supported SVS distance");
    }
}

} // namespace faiss
