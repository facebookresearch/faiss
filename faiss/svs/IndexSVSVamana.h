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

#pragma once

#include "IndexSVSFaissUtils.h"

#include <faiss/Index.h>

#include <svs/runtime/IndexSVSVamanaImpl.h>

#include <iostream>

namespace faiss {

struct SearchParametersSVSVamana : public SearchParameters {
    size_t search_window_size = 0;
    size_t search_buffer_capacity = 0;
};

struct IndexSVSVamana : Index {
    using StorageKind = svs::runtime::IndexSVSVamanaImpl::StorageKind;

    size_t graph_max_degree;
    size_t prune_to;
    float alpha = 1.2;
    size_t search_window_size = 10;
    size_t search_buffer_capacity = 10;
    size_t construction_window_size = 40;
    size_t max_candidate_pool_size = 200;
    bool use_full_search_history = true;

    StorageKind storage_kind;

    IndexSVSVamana();

    IndexSVSVamana(
            idx_t d,
            size_t degree,
            MetricType metric = METRIC_L2,
            StorageKind storage = StorageKind::FP32);

    ~IndexSVSVamana() override;

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
    svs::runtime::IndexSVSVamanaImpl* impl{nullptr};

   protected:
    /* Initializes the implementation*/
    virtual void create_impl();
};

} // namespace faiss
