/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSFlat.h>

#include <faiss/impl/FaissAssert.h>
#include <svs/core/data.h>
#include <svs/core/query_result.h>

#include <svs/orchestrators/exhaustive.h>

namespace faiss {

IndexSVSFlat::IndexSVSFlat(idx_t d, MetricType metric) : Index(d, metric) {}

void IndexSVSFlat::add(idx_t n, const float* x) {
    if (!impl) {
        init_impl(n, x);
        return;
    }

    FAISS_THROW_MSG(
            "IndexSVSFlat does not support adding points after initialization");
}

void IndexSVSFlat::reset() {
    if (impl) {
        delete impl;
        impl = nullptr;
    }
    ntotal = 0;
}

IndexSVSFlat::~IndexSVSFlat() {}

void IndexSVSFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(impl);
    FAISS_THROW_IF_NOT(k > 0);

    auto queries = svs::data::ConstSimpleDataView<float>(x, n, d);

    auto results =
            svs::QueryResult<size_t>{queries.size(), static_cast<size_t>(k)};
    // TODO: Tuable Flat parameters
    impl->search(results.view(), queries, {});

    svs::threads::parallel_for(
            impl->get_threadpool_handle(),
            svs::threads::StaticPartition(n),
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    for (idx_t j = 0; j < k; ++j) {
                        labels[j + i * k] = results.index(i, j);
                        distances[j + i * k] = results.distance(i, j);
                    }
                }
            });
}

void IndexSVSFlat::init_impl(idx_t n, const float* x) {
    auto data = svs::data::SimpleData<float>(n, d);
    auto threadpool = svs::threads::ThreadPoolHandle(
            svs::threads::OMPThreadPool(omp_get_max_threads()));
    ntotal = n;

    svs::threads::parallel_for(
            threadpool,
            svs::threads::StaticPartition(n),
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    data.set_datum(i, std::span<const float>(x + i * d, d));
                }
            });

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            impl = new svs::Flat(svs::Flat::assemble<float>(
                    std::move(data), svs::DistanceIP(), std::move(threadpool)));
            break;
        case METRIC_L2:
            impl = new svs::Flat(svs::Flat::assemble<float>(
                    std::move(data), svs::DistanceL2(), std::move(threadpool)));
            break;
        default:
            FAISS_ASSERT(!"not supported SVS distance");
    }
}

void IndexSVSFlat::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS index not initialized.");

    // Write index to temporary files and concatenate the contents
    svs_io::SVSTempDirectory tmp;
    impl->save(tmp.data);
    tmp.write_files_to_stream(out);
}

void IndexSVSFlat::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(
            impl, "Cannot deserialize: SVS index already initialized.");

    // Write stream to files that can be read by Flat::assemble()
    svs_io::SVSTempDirectory tmp;
    tmp.write_stream_to_files(in);

    auto threadpool = svs::threads::ThreadPoolHandle(
            svs::threads::OMPThreadPool(omp_get_max_threads()));

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            impl = new svs::Flat(svs::Flat::assemble<float>(
                    svs::VectorDataLoader<float>(tmp.data.string()),
                    svs::DistanceIP(),
                    std::move(threadpool)));
            break;
        case METRIC_L2:
            impl = new svs::Flat(svs::Flat::assemble<float>(
                    svs::VectorDataLoader<float>(tmp.data.string()),
                    svs::DistanceL2(),
                    std::move(threadpool)));
            break;
        default:
            FAISS_ASSERT(!"not supported SVS distance");
    }
}

} // namespace faiss
