/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <faiss/utils/GroupHeap.h>
#include <faiss/utils/Heap.h>
#include <gtest/gtest.h>
#include <algorithm>

using namespace faiss;

TEST(GroupHeap, group_heap_replace_top) {
    using C = CMax<float, int64_t>;
    const int k = 100;
    float binary_heap_values[k];
    int64_t binary_heap_ids[k];
    heap_heapify<C>(k, binary_heap_values, binary_heap_ids);
    int64_t binary_heap_group_ids[k];
    for (size_t i = 0; i < k; i++) {
        binary_heap_group_ids[i] = -1;
    }
    std::unordered_map<typename C::TI, size_t> group_id_to_index_in_heap;
    for (int i = 1000; i > 0; i--) {
        group_heap_replace_top<C>(
                k,
                binary_heap_values,
                binary_heap_ids,
                binary_heap_group_ids,
                i * 10.0,
                i,
                i,
                &group_id_to_index_in_heap);
    }

    heap_reorder<C>(k, binary_heap_values, binary_heap_ids);

    for (int i = 0; i < k; i++) {
        ASSERT_EQ((i + 1) * 10.0, binary_heap_values[i]);
        ASSERT_EQ(i + 1, binary_heap_ids[i]);
    }
}

TEST(GroupHeap, group_heap_replace_at) {
    using C = CMax<float, int64_t>;
    const int k = 10;
    float binary_heap_values[k];
    int64_t binary_heap_ids[k];
    heap_heapify<C>(k, binary_heap_values, binary_heap_ids);
    int64_t binary_heap_group_ids[k];
    for (size_t i = 0; i < k; i++) {
        binary_heap_group_ids[i] = -1;
    }
    std::unordered_map<typename C::TI, size_t> group_id_to_index_in_heap;

    std::unordered_map<int64_t, int64_t> group_id_to_id;
    for (int i = 1000; i > 0; i--) {
        int64_t group_id = rand() % 100;
        group_id_to_id[group_id] = i;
        if (group_id_to_index_in_heap.find(group_id) ==
            group_id_to_index_in_heap.end()) {
            group_heap_replace_top<C>(
                    k,
                    binary_heap_values,
                    binary_heap_ids,
                    binary_heap_group_ids,
                    i * 10.0,
                    i,
                    group_id,
                    &group_id_to_index_in_heap);
        } else {
            group_heap_replace_at<C>(
                    group_id_to_index_in_heap.at(group_id),
                    k,
                    binary_heap_values,
                    binary_heap_ids,
                    binary_heap_group_ids,
                    i * 10.0,
                    i,
                    group_id,
                    &group_id_to_index_in_heap);
        }
    }

    heap_reorder<C>(k, binary_heap_values, binary_heap_ids);

    std::vector<int> sorted_ids;
    for (const auto& pair : group_id_to_id) {
        sorted_ids.push_back(pair.second);
    }
    std::sort(sorted_ids.begin(), sorted_ids.end());

    for (int i = 0; i < k && binary_heap_ids[i] != -1; i++) {
        ASSERT_EQ(sorted_ids[i] * 10.0, binary_heap_values[i]);
        ASSERT_EQ(sorted_ids[i], binary_heap_ids[i]);
    }
}
