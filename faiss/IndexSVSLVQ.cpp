/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSLVQ.h>

#include <variant>

#include <svs/orchestrators/dynamic_vamana.h>
namespace faiss {

IndexSVSLVQ::IndexSVSLVQ(idx_t d, MetricType metric, LVQLevel lvq_level)
        : IndexSVS(d, metric), lvq_level{lvq_level} {}

void IndexSVSLVQ::init_impl(idx_t n, const float* x) {
    // TODO: support ConstSimpleDataView in SVS shared/static lib
    const auto data =
            svs::data::SimpleDataView<float>(const_cast<float*>(x), n, d);
    std::vector<size_t> labels(n);
    auto threadpool = svs::threads::ThreadPoolHandle(
            svs::threads::OMPThreadPool(omp_get_max_threads()));

    std::variant<
            std::monostate,
            storage_type_4x0,
            storage_type_4x4,
            storage_type_4x8>
            compressed_data;

    switch (lvq_level) {
        case LVQLevel::LVQ_4x0:
            compressed_data = storage_type_4x0::compress(
                    data, threadpool, 0, blocked_alloc_type{});
            break;
        case LVQLevel::LVQ_4x4:
            compressed_data = storage_type_4x4::compress(
                    data, threadpool, 0, blocked_alloc_type{});
            break;
        case LVQLevel::LVQ_4x8:
            compressed_data = storage_type_4x8::compress(
                    data, threadpool, 0, blocked_alloc_type{});
            break;
        default:
            FAISS_ASSERT(!"not supported SVS LVQ level");
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
                    FAISS_ASSERT(!"SVS LVQ data is not initialized.");
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

void IndexSVSLVQ::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(
            impl, "Cannot deserialize: SVS index already initialized.");

    // Write stream to files that can be read by DynamicVamana::assemble()
    svs_io::SVSTempDirectory tmp;
    tmp.write_stream_to_files(in);
    auto threadpool = svs::threads::ThreadPoolHandle(
            svs::threads::OMPThreadPool(omp_get_max_threads()));

    std::variant<svs::DistanceIP, svs::DistanceL2> svs_distance;
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            svs_distance = svs::DistanceIP();
            break;
        case METRIC_L2:
            svs_distance = svs::DistanceL2();
            break;
        default:
            FAISS_ASSERT(!"not supported SVS distance");
    }

    std::visit(
            [&](auto&& svs_distance) {
                switch (lvq_level) {
                    case LVQLevel::LVQ_4x0:
                        impl = new svs::DynamicVamana(
                                svs::DynamicVamana::assemble<float>(
                                        tmp.config.string(),
                                        svs::GraphLoader(tmp.graph.string()),
                                        svs::lib::load_from_disk<
                                                storage_type_4x0>(
                                                tmp.data.string()),
                                        svs_distance,
                                        std::move(threadpool)));
                        break;
                    case LVQLevel::LVQ_4x4:
                        impl = new svs::DynamicVamana(
                                svs::DynamicVamana::assemble<float>(
                                        tmp.config.string(),
                                        svs::GraphLoader(tmp.graph.string()),
                                        svs::lib::load_from_disk<
                                                storage_type_4x4>(
                                                tmp.data.string()),
                                        svs_distance,
                                        std::move(threadpool)));
                        break;
                    case LVQLevel::LVQ_4x8:
                        impl = new svs::DynamicVamana(
                                svs::DynamicVamana::assemble<float>(
                                        tmp.config.string(),
                                        svs::GraphLoader(tmp.graph.string()),
                                        svs::lib::load_from_disk<
                                                storage_type_4x8>(
                                                tmp.data.string()),
                                        svs_distance,
                                        std::move(threadpool)));
                        break;
                    default:
                        FAISS_ASSERT(!"not supported SVS LVQ level");
                }
            },
            svs_distance);
}

} // namespace faiss
