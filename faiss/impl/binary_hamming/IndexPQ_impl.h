/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Per-ISA implementation of polysemous Hamming inner loop for IndexPQ.
 * Included once per SIMD TU with THE_SIMD_LEVEL set to the desired
 * SIMDLevel.
 */

#pragma once

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL must be defined before including this file"
#endif

#include <faiss/utils/hamming_distance/hamming_computer.h>

#include <faiss/IndexPQ.h>
#include <faiss/impl/binary_hamming/dispatch.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/hamming.h>

namespace faiss {

namespace {

template <class HammingComputer>
size_t polysemous_inner_loop(
        const IndexPQ* index,
        const float* dis_table_qi,
        const uint8_t* q_code,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids,
        int ht) {
    size_t M = index->pq.M;
    size_t code_size = index->pq.code_size;
    size_t ksub = index->pq.ksub;
    size_t ntotal = index->ntotal;

    const uint8_t* b_code = index->codes.data();

    size_t n_pass_i = 0;

    HammingComputer hc(q_code, static_cast<int>(code_size));

    for (int64_t bi = 0; bi < static_cast<int64_t>(ntotal); bi++) {
        int hd = hc.hamming(b_code);

        if (hd < ht) {
            n_pass_i++;

            float dis = 0;
            const float* dis_table = dis_table_qi;
            for (size_t m = 0; m < M; m++) {
                dis += dis_table[b_code[m]];
                dis_table += ksub;
            }

            if (dis < heap_dis[0]) {
                maxheap_replace_top(k, heap_dis, heap_ids, dis, bi);
            }
        }
        b_code += code_size;
    }
    return n_pass_i;
}

} // anonymous namespace

template <>
size_t polysemous_inner_loop_fixSL<THE_SIMD_LEVEL>(
        int code_size,
        const IndexPQ* index,
        const float* dis_table_qi,
        const uint8_t* q_code,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids,
        int ht) {
    return with_HammingComputer<THE_SIMD_LEVEL>(
            code_size, [&]<class HammingComputer>() -> size_t {
                return polysemous_inner_loop<HammingComputer>(
                        index, dis_table_qi, q_code, k, heap_dis, heap_ids, ht);
            });
}

} // namespace faiss
