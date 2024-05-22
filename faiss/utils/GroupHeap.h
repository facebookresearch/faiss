/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <climits>
#include <cmath>
#include <cstring>

#include <stdint.h>
#include <cassert>
#include <cstdio>

#include <limits>
#include <unordered_map>

#include <faiss/MetricType.h>
#include <faiss/utils/ordered_key_value.h>

namespace faiss {

/**
 * From start_index, it compare its value with parent node's and swap if needed.
 * Continue until either there is no swap or it reaches the top node.
 */
template <class C>
static inline void group_up_heap(
        typename C::T* heap_dis,
        typename C::TI* heap_ids,
        typename C::TI* heap_group_ids,
        std::unordered_map<typename C::TI, size_t>* group_id_to_index_in_heap,
        size_t start_index) {
    heap_dis--; /* Use 1-based indexing for easier node->child translation */
    heap_ids--;
    heap_group_ids--;
    size_t i = start_index + 1, i_father;
    typename C::T target_dis = heap_dis[i];
    typename C::TI target_id = heap_ids[i];
    typename C::TI target_group_id = heap_group_ids[i];

    while (i > 1) {
        i_father = i >> 1;
        if (!C::cmp2(
                    target_dis,
                    heap_dis[i_father],
                    target_id,
                    heap_ids[i_father])) {
            /* the heap structure is ok */
            break;
        }
        heap_dis[i] = heap_dis[i_father];
        heap_ids[i] = heap_ids[i_father];
        heap_group_ids[i] = heap_group_ids[i_father];
        (*group_id_to_index_in_heap)[heap_group_ids[i]] = i - 1;
        i = i_father;
    }
    heap_dis[i] = target_dis;
    heap_ids[i] = target_id;
    heap_group_ids[i] = target_group_id;
    (*group_id_to_index_in_heap)[heap_group_ids[i]] = i - 1;
}

/**
 * From start_index, it compare its value with child node's and swap if needed.
 * Continue until either there is no swap or it reaches the leaf node.
 */
template <class C>
static inline void group_down_heap(
        size_t k,
        typename C::T* heap_dis,
        typename C::TI* heap_ids,
        typename C::TI* heap_group_ids,
        std::unordered_map<typename C::TI, size_t>* group_id_to_index_in_heap,
        size_t start_index) {
    heap_dis--; /* Use 1-based indexing for easier node->child translation */
    heap_ids--;
    heap_group_ids--;
    size_t i = start_index + 1, i1, i2;
    typename C::T target_dis = heap_dis[i];
    typename C::TI target_id = heap_ids[i];
    typename C::TI target_group_id = heap_group_ids[i];

    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;
        if (i1 > k) {
            break;
        }

        // Note that C::cmp2() is a bool function answering
        // `(a1 > b1) || ((a1 == b1) && (a2 > b2))` for max
        // heap and same with the `<` sign for min heap.
        if ((i2 == k + 1) ||
            C::cmp2(heap_dis[i1], heap_dis[i2], heap_ids[i1], heap_ids[i2])) {
            if (C::cmp2(target_dis, heap_dis[i1], target_id, heap_ids[i1])) {
                break;
            }
            heap_dis[i] = heap_dis[i1];
            heap_ids[i] = heap_ids[i1];
            heap_group_ids[i] = heap_group_ids[i1];
            (*group_id_to_index_in_heap)[heap_group_ids[i]] = i - 1;
            i = i1;
        } else {
            if (C::cmp2(target_dis, heap_dis[i2], target_id, heap_ids[i2])) {
                break;
            }
            heap_dis[i] = heap_dis[i2];
            heap_ids[i] = heap_ids[i2];
            heap_group_ids[i] = heap_group_ids[i2];
            (*group_id_to_index_in_heap)[heap_group_ids[i]] = i - 1;
            i = i2;
        }
    }
    heap_dis[i] = target_dis;
    heap_ids[i] = target_id;
    heap_group_ids[i] = target_group_id;
    (*group_id_to_index_in_heap)[heap_group_ids[i]] = i - 1;
}

template <class C>
static inline void group_heap_replace_top(
        size_t k,
        typename C::T* heap_dis,
        typename C::TI* heap_ids,
        typename C::TI* heap_group_ids,
        typename C::T dis,
        typename C::TI id,
        typename C::TI group_id,
        std::unordered_map<typename C::TI, size_t>* group_id_to_index_in_heap) {
    assert(group_id_to_index_in_heap->find(group_id) ==
                   group_id_to_index_in_heap->end() &&
           "group id should not exist in the binary heap");

    group_id_to_index_in_heap->erase(heap_group_ids[0]);
    heap_group_ids[0] = group_id;
    heap_dis[0] = dis;
    heap_ids[0] = id;
    (*group_id_to_index_in_heap)[group_id] = 0;
    group_down_heap<C>(
            k,
            heap_dis,
            heap_ids,
            heap_group_ids,
            group_id_to_index_in_heap,
            0);
}

template <class C>
static inline void group_heap_replace_at(
        size_t pos,
        size_t k,
        typename C::T* heap_dis,
        typename C::TI* heap_ids,
        typename C::TI* heap_group_ids,
        typename C::T dis,
        typename C::TI id,
        typename C::TI group_id,
        std::unordered_map<typename C::TI, size_t>* group_id_to_index_in_heap) {
    assert(group_id_to_index_in_heap->find(group_id) !=
                   group_id_to_index_in_heap->end() &&
           "group id should exist in the binary heap");
    assert(group_id_to_index_in_heap->find(group_id)->second == pos &&
           "index of group id in the heap should be same as pos");

    heap_dis[pos] = dis;
    heap_ids[pos] = id;
    group_up_heap<C>(
            heap_dis, heap_ids, heap_group_ids, group_id_to_index_in_heap, pos);
    group_down_heap<C>(
            k,
            heap_dis,
            heap_ids,
            heap_group_ids,
            group_id_to_index_in_heap,
            pos);
}

} // namespace faiss