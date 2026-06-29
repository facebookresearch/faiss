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
#include <faiss/svs/IndexSVSIVFLeanVec.h>

#include <svs/runtime/dynamic_ivf_index.h>
#include <svs/runtime/ivf_index.h>
#include <svs/runtime/training.h>

#include <memory>
#include <numeric>

namespace faiss {

IndexSVSIVFLeanVec::IndexSVSIVFLeanVec() : IndexSVSIVF() {
    is_trained = false;
    storage_kind = SVSStorageKind::SVS_LeanVec4x4;
}

IndexSVSIVFLeanVec::IndexSVSIVFLeanVec(
        idx_t d,
        size_t nlist,
        MetricType metric,
        size_t leanvec_dims,
        SVSStorageKind storage_kind,
        bool is_static)
        : IndexSVSIVF(d, nlist, metric, storage_kind, is_static) {
    is_trained = false;
    leanvec_d = leanvec_dims == 0 ? d / 2 : leanvec_dims;
}

IndexSVSIVFLeanVec::~IndexSVSIVFLeanVec() {
    if (training_data) {
        auto status = svs_runtime::LeanVecTrainingData::destroy(training_data);
        FAISS_ASSERT(status.ok());
        training_data = nullptr;
    }
    // Base class destructor handles impl cleanup
}

void IndexSVSIVFLeanVec::train(idx_t n, const float* x) {
    train_with_queries(n, x, 0, nullptr);
}

void IndexSVSIVFLeanVec::train_with_queries(
        idx_t n,
        const float* x,
        idx_t n_train_q,
        const float* xq_train) {
    FAISS_THROW_IF_MSG(
            training_data || impl, "Index already trained or contains data.");

    FAISS_THROW_IF_NOT_MSG(
            IndexSVSIVF::is_lvq_leanvec_enabled(),
            "LVQ/LeanVec support not available on this platform or build");

    // Build LeanVec training data
    auto status = svs_runtime::LeanVecTrainingData::build(
            &training_data, d, n, x, n_train_q, xq_train, leanvec_d);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT_MSG(
            training_data, "Failed to build leanvec training info.");

    // Now build the IVF index with the training data
    create_impl(n, x);
    is_trained = true;
}

void IndexSVSIVFLeanVec::reset() {
    if (training_data) {
        auto status = svs_runtime::LeanVecTrainingData::destroy(training_data);
        FAISS_ASSERT(status.ok());
        training_data = nullptr;
    }
    IndexSVSIVF::reset();
}

void IndexSVSIVFLeanVec::serialize_training_data(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            training_data, "Cannot serialize: Training data not initialized.");

    auto status = training_data->save(out);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
}

void IndexSVSIVFLeanVec::deserialize_training_data(std::istream& in) {
    svs_runtime::LeanVecTrainingData* tdata = nullptr;
    auto status = svs_runtime::LeanVecTrainingData::load(&tdata, in);
    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT_MSG(tdata, "Failed to load leanvec training data.");
    training_data = tdata;
}

void IndexSVSIVFLeanVec::create_impl(idx_t n, const float* x) {
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

    auto status = svs_runtime::Status_Ok;
    if (is_static) {
        if (training_data) {
            status = svs_runtime::IVFIndexLeanVec::build(
                    &impl,
                    d,
                    svs_metric,
                    svs_storage_kind,
                    static_cast<size_t>(n),
                    x,
                    training_data,
                    build_params,
                    search_params,
                    num_threads,
                    intra_query_threads);
        } else {
            status = svs_runtime::IVFIndexLeanVec::build(
                    &impl,
                    d,
                    svs_metric,
                    svs_storage_kind,
                    static_cast<size_t>(n),
                    x,
                    leanvec_d,
                    build_params,
                    search_params,
                    num_threads,
                    intra_query_threads);
        }
    } else {
        std::vector<size_t> labels(n);
        std::iota(labels.begin(), labels.end(), 0);

        svs_runtime::DynamicIVFIndex* dyn_impl = nullptr;
        if (training_data) {
            status = svs_runtime::DynamicIVFIndexLeanVec::build(
                    &dyn_impl,
                    d,
                    svs_metric,
                    svs_storage_kind,
                    static_cast<size_t>(n),
                    x,
                    labels.data(),
                    training_data,
                    build_params,
                    search_params,
                    num_threads,
                    intra_query_threads);
        } else {
            status = svs_runtime::DynamicIVFIndexLeanVec::build(
                    &dyn_impl,
                    d,
                    svs_metric,
                    svs_storage_kind,
                    static_cast<size_t>(n),
                    x,
                    labels.data(),
                    leanvec_d,
                    build_params,
                    search_params,
                    num_threads,
                    intra_query_threads);
        }
        impl = dyn_impl;
    }

    if (!status.ok()) {
        FAISS_THROW_MSG(status.message());
    }
    FAISS_THROW_IF_NOT(impl);

    // Force reinitialization of SVS search infrastructure after build.
    // Same workaround as in IndexSVSIVF::create_impl — the move chain through
    // the orchestrator wrappers can leave matmul_results_ inconsistent
    // when threads > centroids.
    size_t current_threads = 0;
    impl->get_num_threads(&current_threads);
    auto st = impl->set_num_threads(current_threads);
    if (!st.ok()) {
        FAISS_THROW_MSG(st.message());
    }

    ntotal = n;
}

} // namespace faiss
