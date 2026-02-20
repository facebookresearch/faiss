/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2026 Intel Corporation
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
#include <faiss/svs/IndexSVSIVF.h>

#include <faiss/Index.h>

#include <svs/runtime/api_defs.h>
#include <svs/runtime/dynamic_ivf_index.h>
#include <svs/runtime/ivf_index.h>

#include <cstddef>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

namespace faiss {
namespace {
svs_runtime::IVFIndex::SearchParams make_ivf_search_parameters(
        const IndexSVSIVF& index,
        const SearchParameters* params) {
    FAISS_THROW_IF_NOT(index.impl);

    auto n_probes = index.n_probes;
    auto k_reorder = index.k_reorder;

    if (auto svs_params =
                dynamic_cast<const SearchParametersSVSIVF*>(params)) {
        if (svs_params->n_probes > 0)
            n_probes = svs_params->n_probes;
        if (svs_params->k_reorder > 0)
            k_reorder = svs_params->k_reorder;
    }

    return {n_probes, k_reorder};
}
} // namespace

IndexSVSIVF::IndexSVSIVF() = default;

IndexSVSIVF::IndexSVSIVF(
        idx_t d,
        size_t nlist,
        MetricType metric,
        SVSStorageKind storage)
        : Index(d, metric), num_centroids{nlist}, storage_kind{storage} {
    is_trained = false;

    // Validate the requested storage kind is available in current runtime.
    auto svs_storage = to_svs_storage_kind(storage_kind);
    auto status =
            svs_runtime::DynamicIVFIndex::check_storage_kind(svs_storage);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

bool IndexSVSIVF::is_lvq_leanvec_enabled() {
    auto lvq = to_svs_storage_kind(SVS_LVQ4x0);
    auto status = svs_runtime::DynamicIVFIndex::check_storage_kind(lvq);
    if (!status.ok()) {
        return false;
    }
    auto leanvec = to_svs_storage_kind(SVS_LeanVec4x4);
    status = svs_runtime::DynamicIVFIndex::check_storage_kind(leanvec);
    if (!status.ok()) {
        return false;
    }
    return true;
}

IndexSVSIVF::~IndexSVSIVF() {
    if (impl) {
        auto status = svs_runtime::DynamicIVFIndex::destroy(impl);
        FAISS_ASSERT(status.ok());
        impl = nullptr;
    }
}

void IndexSVSIVF::train(idx_t n, const float* x) {
    FAISS_THROW_IF_MSG(impl, "Index already trained or contains data.");
    create_impl(n, x);
    is_trained = true;
}

void IndexSVSIVF::add(idx_t n, const float* x) {
    FAISS_THROW_IF_MSG(
            !is_trained, "Index not trained: call train() before add().");
    FAISS_THROW_IF_NOT(impl);

    std::vector<size_t> labels(n);
    std::iota(labels.begin(), labels.end(), ntotal);

    auto status = impl->add(n, labels.data(), x);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    ntotal += n;
}

void IndexSVSIVF::reset() {
    if (impl) {
        auto status = svs_runtime::DynamicIVFIndex::destroy(impl);
        FAISS_ASSERT(status.ok());
        impl = nullptr;
    }
    is_trained = false;
    ntotal = 0;
}

void IndexSVSIVF::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (!impl) {
        for (idx_t i = 0; i < n * k; ++i) {
            distances[i] = std::numeric_limits<float>::infinity();
            labels[i] = -1;
        }
        return;
    }
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    // Sync thread settings to the SVS backend before searching
    {
        size_t current_threads = 0;
        impl->get_num_threads(&current_threads);
        if (num_threads > 0 && current_threads != num_threads) {
            auto st = impl->set_num_threads(num_threads);
            if (!st.ok()) {
                FAISS_THROW_MSG(st.message());
            }
        }
    }

    auto sp = make_ivf_search_parameters(*this, params);
    auto status = impl->search(
            static_cast<size_t>(n),
            x,
            static_cast<size_t>(k),
            distances,
            convert_output_buffer<size_t>(labels, static_cast<size_t>(n * k)),
            &sp);

    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

size_t IndexSVSIVF::remove_ids(const IDSelector& sel) {
    FAISS_THROW_IF_NOT(impl);
    auto id_filter = FaissIDFilter{sel};
    size_t removed = 0;
    auto status = impl->remove_selected(&removed, id_filter);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    ntotal -= removed;
    return removed;
}

void IndexSVSIVF::create_impl(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(!impl);
    ntotal = 0;
    auto svs_metric = to_svs_metric(metric_type);
    auto svs_storage_kind = to_svs_storage_kind(storage_kind);
    auto build_params = svs_runtime::IVFIndex::BuildParams{
            .num_centroids = num_centroids,
            .minibatch_size = minibatch_size,
            .num_iterations = num_iterations,
            .is_hierarchical = is_hierarchical,
            .training_fraction = training_fraction,
            .hierarchical_level1_clusters = hierarchical_level1_clusters,
            .seed = seed,
    };
    auto search_params = svs_runtime::IVFIndex::SearchParams{
            .n_probes = n_probes,
            .k_reorder = k_reorder,
    };

    std::vector<size_t> labels(n);
    std::iota(labels.begin(), labels.end(), 0);

    auto status = svs_runtime::DynamicIVFIndex::build(
            &impl,
            d,
            svs_metric,
            svs_storage_kind,
            static_cast<size_t>(n),
            x,
            labels.data(),
            build_params,
            search_params,
            num_threads,
            intra_query_threads);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT(impl);
    ntotal = n;
}

void IndexSVSIVF::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS IVF index not initialized.");

    auto status = impl->save(out);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

void IndexSVSIVF::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(
            impl, "Cannot deserialize: SVS IVF index already loaded.");
    auto svs_metric = to_svs_metric(metric_type);
    auto svs_storage_kind = to_svs_storage_kind(storage_kind);
    auto status = svs_runtime::DynamicIVFIndex::load(
            &impl,
            in,
            svs_metric,
            svs_storage_kind,
            num_threads,
            intra_query_threads);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

} // namespace faiss
