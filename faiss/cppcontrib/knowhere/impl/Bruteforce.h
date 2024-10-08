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
#include <memory>
#include <type_traits>
#include <utility>

#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/Heap.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// C is CMax<> or CMin<>
template <typename C, typename DistanceComputerT, typename FilterT>
void brute_force_search_impl(
        const idx_t ntotal,
        DistanceComputerT& __restrict qdis,
        const FilterT& __restrict filter,
        const idx_t k,
        float* __restrict distances,
        idx_t* __restrict labels) {
    static_assert(std::is_same_v<typename C::T, float>);
    static_assert(std::is_same_v<typename C::TI, idx_t>);

    auto max_heap = std::make_unique<std::pair<float, idx_t>[]>(k);
    idx_t n_added = 0;
    for (idx_t idx = 0; idx < ntotal; ++idx) {
        if (filter.is_member(idx)) {
            const float distance = qdis(idx);
            if (n_added < k) {
                n_added += 1;
                heap_push<C>(n_added, max_heap.get(), distance, idx);
            } else if (C::cmp(max_heap[0].first, distance)) {
                heap_replace_top<C>(k, max_heap.get(), distance, idx);
            }
        }
    }

    const idx_t len = std::min(n_added, idx_t(k));
    for (idx_t i = 0; i < len; i++) {
        labels[len - i - 1] = max_heap[0].second;
        distances[len - i - 1] = max_heap[0].first;

        heap_pop<C>(len - i, max_heap.get());
    }

    // fill leftovers
    if (len < k) {
        for (idx_t idx = len; idx < k; idx++) {
            labels[idx] = -1;
            distances[idx] = C::neutral();
        }
    }
}

// C is CMax<> or CMin<>
template <typename ResultHandlerT, typename DistanceComputerT, typename FilterT>
void brute_force_range_search_impl(
        const idx_t ntotal,
        DistanceComputerT& __restrict qdis,
        const FilterT& __restrict filter,
        ResultHandlerT& __restrict rres) {
    for (idx_t idx = 0; idx < ntotal; ++idx) {
        if (filter.is_member(idx)) {
            const float distance = qdis(idx);
            rres.add_result(distance, idx);
        }
    }
}

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
