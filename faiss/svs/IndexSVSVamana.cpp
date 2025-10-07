/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright 2025 Intel Corporation
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

#include <faiss/svs/IndexSVSVamana.h>

#include <svs/core/data.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/orchestrators/dynamic_vamana.h>

#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {
namespace {
std::variant<svs::DistanceIP, svs::DistanceL2> get_svs_distance(
        faiss::MetricType metric) {
    switch (metric) {
        case METRIC_INNER_PRODUCT:
            return svs::DistanceIP();
        case METRIC_L2:
            return svs::DistanceL2();
        default:
            FAISS_ASSERT(!"not supported SVS distance");
    }
}

std::variant<float, svs::Float16, std::int8_t> get_storage_variant(
        IndexSVSVamana::StorageKind kind) {
    switch (kind) {
        case IndexSVSVamana::StorageKind::FP32:
            return float{};
        case IndexSVSVamana::StorageKind::FP16:
            return svs::Float16{};
        case IndexSVSVamana::StorageKind::SQI8:
            return std::int8_t{};
        default:
            FAISS_ASSERT(!"not supported SVS storage kind");
    }
}

svs::index::vamana::VamanaBuildParameters get_build_parameters(
        const IndexSVSVamana& index) {
    return svs::index::vamana::VamanaBuildParameters{
            index.alpha,
            index.graph_max_degree,
            index.construction_window_size,
            index.max_candidate_pool_size,
            index.prune_to,
            index.use_full_search_history};
}

template <
        typename T,
        typename Alloc = svs::data::Blocked<svs::lib::Allocator<T>>,
        svs::data::ImmutableMemoryDataset Dataset,
        svs::threads::ThreadPool Pool>
requires std::is_floating_point_v<T> || std::is_same_v<T, svs::Float16>
        svs::data::SimpleData<T, svs::Dynamic, Alloc> make_storage(
                const Dataset& data,
                Pool& pool) {
    svs::data::SimpleData<T, svs::Dynamic, Alloc> result(
            data.size(), data.dimensions(), Alloc{});
    svs::threads::parallel_for(
            pool,
            svs::threads::StaticPartition(result.size()),
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    result.set_datum(i, data.get_datum(i));
                }
            });
    return result;
}

template <
        typename T,
        typename Alloc = svs::data::Blocked<svs::lib::Allocator<T>>,
        svs::data::ImmutableMemoryDataset Dataset,
        svs::threads::ThreadPool Pool>
requires std::is_integral_v<T> svs::quantization::scalar::
        SQDataset<T, svs::Dynamic, Alloc>
        make_storage(const Dataset& data, Pool& pool) {
    return svs::quantization::scalar::SQDataset<T, svs::Dynamic, Alloc>::
            compress(data, pool, Alloc{});
}

template <typename ElementType>
svs::DynamicVamana* init_impl_t(
        IndexSVSVamana* index,
        faiss::MetricType metric,
        idx_t n,
        const float* x) {
    auto threadpool = svs::threads::ThreadPoolHandle(
            svs::threads::OMPThreadPool(omp_get_max_threads()));

    auto data = make_storage<ElementType>(
            svs::data::ConstSimpleDataView<float>(x, n, index->d), threadpool);

    std::vector<size_t> labels(data.size());
    std::iota(labels.begin(), labels.end(), 0);

    return std::visit(
            [&](auto&& distance) {
                return new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                        std::move(get_build_parameters(*index)),
                        std::move(data),
                        std::move(labels),
                        std::move(distance),
                        std::move(threadpool)));
            },
            get_svs_distance(metric));
}

template <
        typename T,
        typename Alloc = svs::data::Blocked<svs::lib::Allocator<T>>,
        typename Enabler = void>
struct storage_type;

template <typename T, typename Alloc>
struct storage_type<
        T,
        Alloc,
        std::enable_if_t<
                std::is_floating_point_v<T> ||
                std::is_same_v<T, svs::Float16>>> {
    using type = svs::data::SimpleData<T, svs::Dynamic, Alloc>;
};

template <typename T, typename Alloc>
struct storage_type<T, Alloc, std::enable_if_t<std::is_integral_v<T>>> {
    using type = svs::quantization::scalar::SQDataset<T, svs::Dynamic, Alloc>;
};

template <
        typename T,
        typename Alloc = svs::data::Blocked<svs::lib::Allocator<T>>>
using storage_type_t = typename storage_type<T, Alloc>::type;

template <typename ElementType>
svs::DynamicVamana* deserialize_impl_t(
        std::istream& stream,
        faiss::MetricType metric) {
    auto threadpool = svs::threads::ThreadPoolHandle(
            svs::threads::OMPThreadPool(omp_get_max_threads()));

    return std::visit(
            [&](auto&& distance) {
                return new svs::DynamicVamana(
                        svs::DynamicVamana::
                                assemble<float, storage_type_t<ElementType>>(
                                        stream,
                                        std::move(distance),
                                        std::move(threadpool)));
            },
            get_svs_distance(metric));
}

svs::index::vamana::VamanaSearchParameters make_search_parameters(
        const IndexSVSVamana& index,
        const SearchParameters* params) {
    FAISS_THROW_IF_NOT(index.impl);

    auto search_window_size = index.search_window_size;
    auto search_buffer_capacity = index.search_buffer_capacity;

    if (auto svs_params =
                dynamic_cast<const SearchParametersSVSVamana*>(params)) {
        if (svs_params->search_window_size > 0)
            search_window_size = svs_params->search_window_size;
        if (svs_params->search_buffer_capacity > 0)
            search_buffer_capacity = svs_params->search_buffer_capacity;
    }

    return index.impl->get_search_parameters().buffer_config(
            {search_window_size, search_buffer_capacity});
}
} // namespace

IndexSVSVamana::IndexSVSVamana() : Index{} {}

IndexSVSVamana::IndexSVSVamana(
        idx_t d,
        size_t degree,
        MetricType metric,
        StorageKind storage)
        : Index(d, metric), graph_max_degree{degree}, storage_kind{storage} {
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

    std::iota(labels.begin(), labels.end(), ntotal);
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
    if (!impl) {
        for (idx_t i = 0; i < n; ++i) {
            distances[i] = std::numeric_limits<float>::infinity();
            labels[i] = -1;
        }
        return;
    }
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    auto sp = make_search_parameters(*this, params);

    // Simple search
    if (params == nullptr || params->sel == nullptr) {
        auto queries = svs::data::ConstSimpleDataView<float>(x, n, d);

        // TODO: faiss use int64_t as label whereas SVS uses size_t?
        auto results = svs::QueryResultView<size_t>{
                svs::MatrixView<size_t>{
                        svs::make_dims(n, k),
                        static_cast<size_t*>(static_cast<void*>(labels))},
                svs::MatrixView<float>{svs::make_dims(n, k), distances}};
        impl->search(results, queries, sp);
        return;
    }

    // Selective search with IDSelector
    auto old_sp = impl->get_search_parameters();
    impl->set_search_parameters(sp);

    auto search_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
        for (auto i : range) {
            // For every query
            auto query = std::span(x + i * d, d);
            auto curr_distances = std::span(distances + i * k, k);
            auto curr_labels = std::span(labels + i * k, k);

            auto iterator = impl->batch_iterator(query);
            idx_t found = 0;
            do {
                iterator.next(k);
                for (auto& neighbor : iterator.results()) {
                    if (params->sel->is_member(neighbor.id())) {
                        curr_distances[found] = neighbor.distance();
                        curr_labels[found] = neighbor.id();
                        found++;
                        if (found == k) {
                            break;
                        }
                    }
                }
            } while (found < k && !iterator.done());
            // Pad with -1s
            for (; found < k; ++found) {
                curr_distances[found] = -1;
                curr_labels[found] = -1;
            }
        }
    };

    auto threadpool = svs::threads::OMPThreadPool(
            std::min(n, idx_t(omp_get_max_threads())));

    svs::threads::parallel_for(
            threadpool, svs::threads::StaticPartition{n}, search_closure);

    impl->set_search_parameters(old_sp);
}

void IndexSVSVamana::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(impl);
    FAISS_THROW_IF_NOT(radius > 0);
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(result->nq == n);

    auto sp = make_search_parameters(*this, params);
    auto old_sp = impl->get_search_parameters();
    impl->set_search_parameters(sp);

    // Using ResultHandler makes no sense due to it's complexity, overhead and
    // missed features; e.g. add_result() does not indicate whether result added
    // or not - we have to manually manage threshold comparison and id
    // selection.

    // Prepare output buffers
    std::vector<std::vector<svs::Neighbor<size_t>>> all_results(n);
    // Reserve space for allocation to avoid multiple reallocations
    // Use search_buffer_capacity as a heuristic
    const auto result_capacity = sp.buffer_config_.get_total_capacity();
    for (auto& res : all_results) {
        res.reserve(result_capacity);
    }

    std::function<bool(float, float)> compare = std::visit(
            [](auto&& dist) {
                return std::function<bool(float, float)>{
                        svs::distance::comparator(dist)};
            },
            get_svs_distance(metric_type));

    std::function<bool(size_t)> select = [](size_t) { return true; };
    if (params != nullptr && params->sel != nullptr) {
        select = [&](size_t id) { return params->sel->is_member(id); };
    }

    // Set iterator batch size to search window size
    auto batch_size = sp.buffer_config_.get_search_window_size();

    auto range_search_closure = [&](const auto& range,
                                    uint64_t SVS_UNUSED(tid)) {
        for (auto i : range) {
            // For every query
            auto query = std::span(x + i * d, d);

            auto iterator = impl->batch_iterator(query);
            bool in_range = true;

            do {
                iterator.next(batch_size);
                for (auto& neighbor : iterator.results()) {
                    // SVS comparator functor returns true if the first distance
                    // is 'closer' than the second one
                    in_range = compare(neighbor.distance(), radius);
                    if (in_range) {
                        // Selective search with IDSelector
                        if (select(neighbor.id())) {
                            all_results[i].push_back(neighbor);
                        }
                    } else {
                        // Since iterator.results() are ordered by distance, we
                        // can stop processing
                        break;
                    }
                }
            } while (in_range && !iterator.done());
        }
    };

    auto threadpool = svs::threads::OMPThreadPool(
            std::min(n, idx_t(omp_get_max_threads())));

    svs::threads::parallel_for(
            threadpool, svs::threads::StaticPartition{n}, range_search_closure);

    // RangeSearchResult .ctor() allows unallocated lims
    if (result->lims == nullptr) {
        result->lims = new size_t[result->nq + 1];
    }

    std::transform(
            all_results.begin(),
            all_results.end(),
            result->lims,
            [](const auto& res) { return res.size(); });

    result->do_allocation();

    for (size_t q = 0; q < n; ++q) {
        size_t ofs = result->lims[q];
        for (const auto& [id, distance] : all_results[q]) {
            result->labels[ofs] = id;
            result->distances[ofs] = distance;
            ofs++;
        }
    }

    impl->set_search_parameters(old_sp);
    return;
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

    ntotal = n;
    impl = std::visit(
            [&](auto element) {
                using ElementType = std::decay_t<decltype(element)>;
                return init_impl_t<ElementType>(this, metric_type, n, x);
            },
            get_storage_variant(storage_kind));
}

void IndexSVSVamana::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS index not initialized.");

    impl->save(out);
}

void IndexSVSVamana::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(
            impl, "Cannot deserialize: SVS index already initialized.");

    impl = std::visit(
            [&](auto element) {
                using ElementType = std::decay_t<decltype(element)>;
                return deserialize_impl_t<ElementType>(in, metric_type);
            },
            get_storage_variant(storage_kind));
}

} // namespace faiss
