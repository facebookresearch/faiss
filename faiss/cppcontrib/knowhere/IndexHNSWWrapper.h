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

#include <cstddef>
#include <cstdint>

#include <faiss/IndexHNSW.h>

#include <faiss/cppcontrib/knowhere/IndexWrapper.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// Custom parameters for IndexHNSW.
struct SearchParametersHNSWWrapper : public SearchParametersHNSW {
    // Stats will be updated if the object pointer is provided.
    HNSWStats* hnsw_stats = nullptr;
    // filtering parameter. A floating point value within [0.0f, 1.0f range]
    float kAlpha = 0.0f;

    inline ~SearchParametersHNSWWrapper() {}
};

// TODO:
// Please note that this particular searcher is int32_t based, so won't
//   work correctly for 2B+ samples. This can be easily changed, if needed.

// override a search() procedure for IndexHNSW.
struct IndexHNSWWrapper : IndexWrapper {
    IndexHNSWWrapper(IndexHNSW* underlying_index);

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// entry point for range search
    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params) const override;
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
