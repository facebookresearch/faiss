/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL must be defined before including pq_scan_impl.h"
#endif

#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>
#include <faiss/utils/Heap.h>

namespace faiss {
namespace pq_code_distance {

namespace {

template <class C>
void pq_scan_8bit_at_level(
        size_t M,
        const float* dis_table,
        const uint8_t* codes,
        size_t ncodes,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids) {
    const size_t code_size = M;

    size_t j = 0;
    for (; j + 3 < ncodes; j += 4) {
        float d0, d1, d2, d3;
        pq_code_distance_8bit_four_impl<THE_SIMD_LEVEL>( // NOLINT(facebook-modularize-issue-check)
                M,
                dis_table,
                codes + j * code_size,
                codes + (j + 1) * code_size,
                codes + (j + 2) * code_size,
                codes + (j + 3) * code_size,
                d0,
                d1,
                d2,
                d3);

        if (C::cmp(heap_dis[0], d0)) {
            heap_replace_top<C>(k, heap_dis, heap_ids, d0, j);
        }
        if (C::cmp(heap_dis[0], d1)) {
            heap_replace_top<C>(k, heap_dis, heap_ids, d1, j + 1);
        }
        if (C::cmp(heap_dis[0], d2)) {
            heap_replace_top<C>(k, heap_dis, heap_ids, d2, j + 2);
        }
        if (C::cmp(heap_dis[0], d3)) {
            heap_replace_top<C>(k, heap_dis, heap_ids, d3, j + 3);
        }
    }

    for (; j < ncodes; j++) {
        float dis = pq_code_distance_8bit_single_impl<THE_SIMD_LEVEL>( // NOLINT(facebook-modularize-issue-check)
                M, dis_table, codes + j * code_size);
        if (C::cmp(heap_dis[0], dis)) {
            heap_replace_top<C>(k, heap_dis, heap_ids, dis, j);
        }
    }
}

} // anonymous namespace

template <SIMDLevel SL>
void pq_scan_8bit_impl(
        size_t M,
        const float* dis_table,
        const uint8_t* codes,
        size_t ncodes,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids,
        bool max_heap);

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization,facebook-modularize-issue-check)
template <>
void pq_scan_8bit_impl<THE_SIMD_LEVEL>(
        size_t M,
        const float* dis_table,
        const uint8_t* codes,
        size_t ncodes,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids,
        bool max_heap) {
    if (max_heap) {
        pq_scan_8bit_at_level<CMax<float, int64_t>>(
                M, dis_table, codes, ncodes, k, heap_dis, heap_ids);
    } else {
        pq_scan_8bit_at_level<CMin<float, int64_t>>(
                M, dis_table, codes, ncodes, k, heap_dis, heap_ids);
    }
}

} // namespace pq_code_distance
} // namespace faiss
