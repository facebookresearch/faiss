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

#pragma once

#include <faiss/Index.h>
#include <faiss/svs/IndexSVSFaissUtils.h>

#include <svs/runtime/api_defs.h>

#include <iostream>

namespace faiss {

struct SearchParametersSVSVamana : public SearchParameters {
    size_t search_window_size = 0;
    size_t search_buffer_capacity = 0;
};

// redefinition for swig export
enum SVSStorageKind {
    SVS_FP32,
    SVS_FP16,
    SVS_SQI8,
    SVS_LVQ4x0,
    SVS_LVQ4x4,
    SVS_LVQ4x8,
    SVS_LeanVec4x4,
    SVS_LeanVec4x8,
    SVS_LeanVec8x8,
};

inline svs_runtime::StorageKind to_svs_storage_kind(SVSStorageKind kind) {
    switch (kind) {
        case SVS_FP32:
            return svs_runtime::StorageKind::FP32;
        case SVS_FP16:
            return svs_runtime::StorageKind::FP16;
        case SVS_SQI8:
            return svs_runtime::StorageKind::SQI8;
        case SVS_LVQ4x0:
            return svs_runtime::StorageKind::LVQ4x0;
        case SVS_LVQ4x4:
            return svs_runtime::StorageKind::LVQ4x4;
        case SVS_LVQ4x8:
            return svs_runtime::StorageKind::LVQ4x8;
        case SVS_LeanVec4x4:
            return svs_runtime::StorageKind::LeanVec4x4;
        case SVS_LeanVec4x8:
            return svs_runtime::StorageKind::LeanVec4x8;
        case SVS_LeanVec8x8:
            return svs_runtime::StorageKind::LeanVec8x8;
        default:
            FAISS_ASSERT(!"not supported SVS storage kind");
    }
}

struct IndexSVSVamana : Index {
    size_t graph_max_degree;
    size_t prune_to;
    float alpha = 1.2;
    size_t search_window_size = 10;
    size_t search_buffer_capacity = 10;
    size_t construction_window_size = 40;
    size_t max_candidate_pool_size = 200;
    bool use_full_search_history = true;

    SVSStorageKind storage_kind;

    IndexSVSVamana();

    IndexSVSVamana(
            idx_t d,
            size_t degree,
            MetricType metric = METRIC_L2,
            SVSStorageKind storage = SVSStorageKind::SVS_FP32);

    ~IndexSVSVamana() override;

    // static member that exposes whether or not LVQ/LeanVec are enabled for
    // this build and runtime.
    static bool is_lvq_leanvec_enabled();

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    size_t remove_ids(const IDSelector& sel) override;

    void reset() override;

    /* Serialization and deserialization helpers */
    void serialize_impl(std::ostream& out) const;
    virtual void deserialize_impl(std::istream& in);

    /* The actual SVS implementation */
    svs_runtime::DynamicVamanaIndex* impl{nullptr};

   protected:
    /* Initializes the implementation*/
    virtual void create_impl();
};

} // namespace faiss
