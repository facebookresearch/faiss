/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSLeanVec.h>
#include <variant>
#include "svs/core/medioid.h"
#include "svs/orchestrators/dynamic_vamana.h"

namespace faiss {

IndexSVSLeanVec::IndexSVSLeanVec(
        idx_t d,
        MetricType metric,
        size_t leanvec_dims,
        LeanVecLevel leanvec_level)
        : IndexSVS(d, metric), leanvec_level{leanvec_level} {
    leanvec_d = leanvec_dims == 0 ? d / 2 : leanvec_d;
    is_trained = false;
}

void IndexSVSLeanVec::train(idx_t n, const float* x) {
    const auto data =
            svs::data::SimpleDataView<float>(const_cast<float*>(x), n, d);
    auto threadpool = svs::threads::as_threadpool(num_threads);
    auto means = svs::utils::compute_medioid(data, threadpool);
    auto matrix = svs::leanvec::compute_leanvec_matrix<svs::Dynamic, svs::Dynamic>(data, means, threadpool, svs::lib::MaybeStatic<svs::Dynamic>{leanvec_d});
    leanvec_matrix = svs::leanvec::LeanVecMatrices<svs::Dynamic>(matrix, matrix);
    is_trained = true;
}

void IndexSVSLeanVec::reset() {
    is_trained = false;
    IndexSVS::reset();
}

void IndexSVSLeanVec::init_impl(idx_t n, const float* x) {
    // TODO: support ConstSimpleDataView in SVS shared/static lib
    const auto data =
            svs::data::SimpleDataView<float>(const_cast<float*>(x), n, d);
    std::vector<size_t> labels(n);
    auto threadpool = svs::threads::as_threadpool(num_threads);

    std::variant<
            std::monostate,
            storage_type_4x4,
            storage_type_4x8,
            storage_type_8x8>
            compressed_data;

    switch (leanvec_level) {
        case LeanVecLevel::LeanVec_4x4:
            compressed_data = storage_type_4x4::reduce(
                    data,
                    leanvec_matrix,
                    threadpool,
                    0,
                    svs::lib::MaybeStatic<svs::Dynamic>(leanvec_d),
                    blocked_alloc_type{});
            break;
        case LeanVecLevel::LeanVec_4x8:
            compressed_data = storage_type_4x8::reduce(
                    data,
                    leanvec_matrix,
                    threadpool,
                    0,
                    svs::lib::MaybeStatic<svs::Dynamic>(leanvec_d),
                    blocked_alloc_type{});
            break;
        case LeanVecLevel::LeanVec_8x8:
            compressed_data = storage_type_8x8::reduce(
                    data,
                    leanvec_matrix,
                    threadpool,
                    0,
                    svs::lib::MaybeStatic<svs::Dynamic>(leanvec_d),
                    blocked_alloc_type{});
            break;
        default:
            FAISS_ASSERT(!"not supported SVS LeanVec level");
    }

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

    std::visit(
            [&](auto&& storage) {
                if constexpr (std::is_same_v<
                                      std::decay_t<decltype(storage)>,
                                      std::monostate>) {
                    FAISS_ASSERT(!"SVS LeanVec data is not initialized.");
                } else {
                    switch (metric_type) {
                        case METRIC_INNER_PRODUCT:
                            impl = new svs::DynamicVamana(
                                    svs::DynamicVamana::build<float>(
                                            std::move(build_parameters),
                                            std::forward<decltype(storage)>(
                                                    storage),
                                            std::move(labels),
                                            svs::DistanceIP(),
                                            std::move(threadpool)));
                            break;
                        case METRIC_L2:
                            impl = new svs::DynamicVamana(
                                    svs::DynamicVamana::build<float>(
                                            std::move(build_parameters),
                                            std::forward<decltype(storage)>(
                                                    storage),
                                            std::move(labels),
                                            svs::DistanceL2(),
                                            std::move(threadpool)));
                            break;
                        default:
                            FAISS_ASSERT(!"not supported SVS distance");
                    }
                }
            },
            compressed_data);
}

} // namespace faiss
