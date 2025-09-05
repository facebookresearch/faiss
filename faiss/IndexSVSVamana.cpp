/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSVamana.h>

#include <svs/core/data.h>
#include <svs/orchestrators/dynamic_vamana.h>

#include <faiss/MetricType.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

IndexSVSVamana::IndexSVSVamana() : Index{} {}

IndexSVSVamana::IndexSVSVamana(idx_t d, size_t degree, MetricType metric)
        : Index(d, metric), graph_max_degree{degree} {
    prune_to = graph_max_degree < 4 ? graph_max_degree : graph_max_degree - 4;
    alpha = metric == METRIC_L2 ? 1.2f : 0.95f;
}

IndexSVSVamana::~IndexSVSVamana() {
    if (impl) {
        delete impl;
        impl = nullptr;
    }
}

void IndexSVSVamana::add(idx_t n, const float* x) {
    if (!impl) {
        init_impl(n, x);
        return;
    }

    // construct sequential labels
    std::vector<size_t> labels(n);

    svs::threads::parallel_for(
            impl->get_threadpool_handle(),
            svs::threads::StaticPartition(n),
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    labels[i] = ntotal + i;
                }
            });
    ntotal += n;

    auto data = svs::data::ConstSimpleDataView<float>(x, n, d);
    impl->add_points(data, labels);
}

void IndexSVSVamana::reset() {
    if (impl) {
        delete impl;
        impl = nullptr;
    }
    ntotal = 0;
    ntotal_soft_deleted = 0;
}

void IndexSVSVamana::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(impl);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    auto queries = svs::data::ConstSimpleDataView<float>(x, n, d);

    // TODO: use params for SVS search parameters
    auto sp = impl->get_search_parameters();
    sp.buffer_config({search_window_size, search_buffer_capacity});
    // TODO: faiss use int64_t as label whereas SVS uses size_t?
    auto results = svs::QueryResultView<size_t>{
            svs::MatrixView<size_t>{
                    svs::make_dims(n, k),
                    static_cast<size_t*>(static_cast<void*>(labels))},
            svs::MatrixView<float>{svs::make_dims(n, k), distances}};
    impl->search(results, queries, sp);
}

size_t IndexSVSVamana::remove_ids(const IDSelector& sel) {
    std::vector<size_t> ids;
    for (idx_t i = 0; i < ntotal; ++i) {
        if (sel.is_member(i)) {
            ids.emplace_back(i);
        }
    }

    // SVS deletion is a soft deletion, meaning the corresponding vectors are
    // marked as deleted but still present in both the dataset and the graph,
    // and will be navigated through during search.
    // Actual cleanup happens once a large enough number of soft deleted vectors
    // are collected.
    impl->delete_points(ids);
    ntotal -= ids.size();
    ntotal_soft_deleted += ids.size();

    const float cleanup_threshold = .5f;
    if (ntotal == 0 ||
        (float)ntotal_soft_deleted / ntotal > cleanup_threshold) {
        impl->consolidate();
        impl->compact();
        ntotal_soft_deleted = 0;
    }
    return ids.size();
}

void IndexSVSVamana::init_impl(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(!impl);
    FAISS_THROW_IF_NOT(ntotal == 0);

    std::vector<size_t> labels(n);
    auto data = svs::data::SimpleData<float>(n, d);
    auto threadpool = svs::threads::ThreadPoolHandle(
            svs::threads::OMPThreadPool(omp_get_max_threads()));

    svs::threads::parallel_for(
            threadpool,
            svs::threads::StaticPartition(n),
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    data.set_datum(i, std::span<const float>(x + i * d, d));
                    labels[i] = i;
                }
            });
    ntotal = n;

    svs::index::vamana::VamanaBuildParameters build_parameters{
            alpha,
            graph_max_degree,
            construction_window_size,
            max_candidate_pool_size,
            prune_to,
            use_full_search_history};

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            impl = new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                    std::move(build_parameters),
                    std::move(data),
                    std::move(labels),
                    svs::DistanceIP(),
                    std::move(threadpool)));
            break;
        case METRIC_L2:
            impl = new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                    std::move(build_parameters),
                    std::move(data),
                    std::move(labels),
                    svs::DistanceL2(),
                    std::move(threadpool)));
            break;
        default:
            FAISS_ASSERT(!"not supported SVS distance");
    }
}

void IndexSVSVamana::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS index not initialized.");

    // Write index to temporary files and concatenate the contents
    svs_io::SVSTempDirectory tmp;
    impl->save(tmp.config, tmp.graph, tmp.data);
    tmp.write_files_to_stream(out);
}

void IndexSVSVamana::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(
            impl, "Cannot deserialize: SVS index already initialized.");

    // Write stream to files that can be read by DynamicVamana::assemble()
    svs_io::SVSTempDirectory tmp;
    tmp.write_stream_to_files(in);
    auto threadpool = svs::threads::ThreadPoolHandle(
            svs::threads::OMPThreadPool(omp_get_max_threads()));

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            impl = new svs::DynamicVamana(svs::DynamicVamana::assemble<float>(
                    tmp.config.string(),
                    svs::GraphLoader(tmp.graph.string()),
                    svs::VectorDataLoader<float>(tmp.data.string()),
                    svs::distance::DistanceIP(),
                    std::move(threadpool)));
            break;
        case METRIC_L2:
            impl = new svs::DynamicVamana(svs::DynamicVamana::assemble<float>(
                    tmp.config.string(),
                    svs::GraphLoader(tmp.graph.string()),
                    svs::VectorDataLoader<float>(tmp.data.string()),
                    svs::distance::DistanceL2(),
                    std::move(threadpool)));
            break;
        default:
            FAISS_ASSERT(!"not supported SVS distance");
    }
}

} // namespace faiss
