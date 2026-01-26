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
#include <faiss/svs/IndexSVSVamanaLeanVec.h>

#include <svs/runtime/dynamic_vamana_index.h>
#include <svs/runtime/training.h>
#include <svs/runtime/vamana_index.h>

#include <memory>
#include <span>
#include "faiss/svs/IndexSVSVamana.h"

namespace faiss {

IndexSVSVamanaLeanVec::IndexSVSVamanaLeanVec() : IndexSVSVamana() {
    is_trained = false;
    storage_kind = SVSStorageKind::SVS_LeanVec4x4;
}

IndexSVSVamanaLeanVec::IndexSVSVamanaLeanVec(
        idx_t d,
        size_t degree,
        MetricType metric,
        size_t leanvec_dims,
        SVSStorageKind storage_kind)
        : IndexSVSVamana(d, degree, metric, storage_kind) {
    is_trained = false;
    leanvec_d = leanvec_dims == 0 ? d / 2 : leanvec_dims;
}

IndexSVSVamanaLeanVec::~IndexSVSVamanaLeanVec() {
    if (training_data) {
        auto status = svs_runtime::LeanVecTrainingData::destroy(training_data);
        FAISS_ASSERT(status.ok());
        training_data = nullptr;
    }
    IndexSVSVamana::~IndexSVSVamana();
}

void IndexSVSVamanaLeanVec::add(idx_t n, const float* x) {
    FAISS_THROW_IF_MSG(
            !is_trained, "Index not trained: call train() before add().");
    IndexSVSVamana::add(n, x);
}

void IndexSVSVamanaLeanVec::train(idx_t n, const float* x) {
    FAISS_THROW_IF_MSG(
            training_data || impl, "Index already trained or contains data.");

    FAISS_THROW_IF_NOT_MSG(
            IndexSVSVamana::is_lvq_leanvec_enabled(),
            "LVQ/LeanVec support not available on this platform or build");

    auto status = svs_runtime::LeanVecTrainingData::build(
            &training_data, d, n, x, leanvec_d);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT_MSG(
            training_data, "Failed to build leanvec training info.");
    is_trained = true;
}

void IndexSVSVamanaLeanVec::serialize_training_data(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            training_data, "Cannot serialize: Training data not initialized.");

    auto status = training_data->save(out);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

void IndexSVSVamanaLeanVec::deserialize_training_data(std::istream& in) {
    svs_runtime::LeanVecTrainingData* tdata = nullptr;
    auto status = svs_runtime::LeanVecTrainingData::load(&tdata, in);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT_MSG(tdata, "Failed to load leanvec training data.");
    training_data = tdata;
}

void IndexSVSVamanaLeanVec::create_impl() {
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
    auto status = svs_runtime::Status_Ok;
    if (training_data) {
        status = svs_runtime::DynamicVamanaIndexLeanVec::build(
                &impl,
                d,
                svs_metric,
                svs_storage_kind,
                training_data,
                build_params,
                search_params);
    } else {
        status = svs_runtime::DynamicVamanaIndexLeanVec::build(
                &impl,
                d,
                svs_metric,
                svs_storage_kind,
                leanvec_d,
                build_params,
                search_params);
    }

    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT(impl);
}

} // namespace faiss
