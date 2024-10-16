// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once

#include <faiss/Index.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// This index is useful for overriding a certain base functionality
//   on the fly.
struct IndexWrapper : Index {
    // a non-owning pointer
    Index* index = nullptr;

    explicit IndexWrapper(Index* underlying_index);

    virtual ~IndexWrapper();

    void train(idx_t n, const float* x) override;

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

    void reset() override;

    void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    DistanceComputer* get_distance_computer() const override;
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
