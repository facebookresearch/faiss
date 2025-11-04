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

#include <faiss/svs/IndexSVSFaissUtils.h>
#include <faiss/svs/IndexSVSVamana.h>

#include <faiss/Index.h>

#include <svs/runtime/IndexSVSTrainingInfo.h>

#include <cstddef>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace faiss {
namespace {
svs::runtime::IndexSVSVamanaImpl::SearchParams make_search_parameters(
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

    return {search_window_size, search_buffer_capacity};
}
} // namespace

IndexSVSVamana::IndexSVSVamana() = default;

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
    svs::runtime::IndexSVSVamanaImpl::destroy(impl);
    impl = nullptr;
}

void IndexSVSVamana::add(idx_t n, const float* x) {
    if (!impl) {
        create_impl();
    }

    auto status = impl->add(n, x);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
    ntotal += n;
}

void IndexSVSVamana::reset() {
    if (impl) {
        impl->reset();
    }
    is_trained = false;
    ntotal = 0;
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
    auto id_filter = make_faiss_id_filter(params);
    auto status = impl->search(
            static_cast<size_t>(n),
            x,
            static_cast<size_t>(k),
            distances,
            convert_output_buffer<size_t>(labels, static_cast<size_t>(n * k)),
            &sp,
            id_filter.get());

    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
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
    FAISS_THROW_IF_NOT(result->nq == static_cast<size_t>(n));

    auto sp = make_search_parameters(*this, params);
    auto id_filter = make_faiss_id_filter(params);
    auto status = impl->range_search(
            static_cast<size_t>(n),
            x,
            radius,
            FaissResultsAllocator{result},
            &sp,
            id_filter.get());
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
}

size_t IndexSVSVamana::remove_ids(const IDSelector& sel) {
    FAISS_THROW_IF_NOT(impl);
    auto id_filter = FaissIDFilter{sel};
    size_t removed = impl->remove_ids(id_filter);
    ntotal -= removed;
    return removed;
}

void IndexSVSVamana::create_impl() {
    FAISS_THROW_IF_NOT(!impl);
    ntotal = 0;
    auto svs_metric = to_svs_metric(metric_type);
    svs::runtime::IndexSVSVamanaImpl::BuildParams build_params;
    build_params.storage_kind = storage_kind;
    build_params.graph_max_degree = graph_max_degree;
    build_params.prune_to = prune_to;
    build_params.alpha = alpha;
    build_params.construction_window_size = construction_window_size;
    build_params.max_candidate_pool_size = max_candidate_pool_size;
    impl = svs::runtime::IndexSVSVamanaImpl::build(d, svs_metric, build_params);
    FAISS_THROW_IF_NOT(impl);
}

void IndexSVSVamana::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS index not initialized.");

    auto status = impl->serialize_impl(out);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
}

void IndexSVSVamana::deserialize_impl(std::istream& in) {
    if (!impl) {
        create_impl();
    }

    auto status = impl->deserialize_impl(in);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message);
    }
}

} // namespace faiss
