/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2025 Intel Corporation
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

#include <svs/runtime/api_defs.h>
#include <svs/runtime/dynamic_vamana_index.h>
#include <svs/runtime/vamana_index.h>

#include <cstddef>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace faiss {
namespace {
svs_runtime::VamanaIndex::SearchParams make_search_parameters(
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
        SVSStorageKind storage)
        : Index(d, metric), graph_max_degree{degree}, storage_kind{storage} {
    prune_to = graph_max_degree < 4 ? graph_max_degree : graph_max_degree - 4;
    alpha = metric == METRIC_L2 ? 1.2f : 0.95f;

    // Validate the requested storage kind is available in current runtime.
    // NB: LVQ/LeanVec are only available on Intel(R) hardware AND when using
    //     a build based on LVQ/LeanVec-enabled SVS.
    auto svs_storage = to_svs_storage_kind(storage_kind);
    auto status =
            svs_runtime::DynamicVamanaIndex::check_storage_kind(svs_storage);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

bool IndexSVSVamana::is_lvq_leanvec_enabled() {
    auto lvq = to_svs_storage_kind(SVS_LVQ4x0);
    auto status = svs_runtime::DynamicVamanaIndex::check_storage_kind(lvq);
    if (!status.ok()) {
        return false;
    }
    auto leanvec = to_svs_storage_kind(SVS_LeanVec4x4);
    status = svs_runtime::DynamicVamanaIndex::check_storage_kind(leanvec);
    if (!status.ok()) {
        return false;
    }
    return true;
}

IndexSVSVamana::~IndexSVSVamana() {
    if (impl) {
        auto status = svs_runtime::DynamicVamanaIndex::destroy(impl);
        FAISS_ASSERT(status.ok());
        impl = nullptr;
    }
}

void IndexSVSVamana::add(idx_t n, const float* x) {
    if (!impl) {
        create_impl();
    }

    std::vector<size_t> labels(n);
    std::iota(labels.begin(), labels.end(), ntotal);

    auto status = impl->add(n, labels.data(), x);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
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
        FAISS_THROW_MSG(status.message());
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
        FAISS_THROW_MSG(status.message());
    }
}

size_t IndexSVSVamana::remove_ids(const IDSelector& sel) {
    FAISS_THROW_IF_NOT(impl);
    auto id_filter = FaissIDFilter{sel};
    size_t removed = 0;
    auto Status = impl->remove_selected(&removed, id_filter);
    ntotal -= removed;
    return removed;
}

void IndexSVSVamana::create_impl() {
    FAISS_THROW_IF_NOT(!impl);
    ntotal = 0;
    auto svs_metric = to_svs_metric(metric_type);
    auto svs_storage_kind = to_svs_storage_kind(storage_kind);
    auto build_params = svs_runtime::VamanaIndex::BuildParams{
            .graph_max_degree = graph_max_degree,
            .prune_to = prune_to,
            .alpha = alpha,
            .construction_window_size = construction_window_size,
            .max_candidate_pool_size = max_candidate_pool_size,
            .use_full_search_history = use_full_search_history,
    };
    auto search_params = svs_runtime::VamanaIndex::SearchParams{
            .search_window_size = search_window_size,
            .search_buffer_capacity = search_buffer_capacity,
    };
    auto Status = svs_runtime::DynamicVamanaIndex::build(
            &impl,
            d,
            svs_metric,
            svs_storage_kind,
            build_params,
            search_params);
    if (!Status.ok()) {
        FAISS_THROW_MSG(Status.message());
    }
    FAISS_THROW_IF_NOT(impl);
}

void IndexSVSVamana::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS index not initialized.");

    auto status = impl->save(out);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

void IndexSVSVamana::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(impl, "Cannot deserialize: SVS index already loaded.");
    auto svs_metric = to_svs_metric(metric_type);
    auto svs_storage_kind = to_svs_storage_kind(storage_kind);
    auto status = impl->load(&impl, in, svs_metric, svs_storage_kind);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

} // namespace faiss
