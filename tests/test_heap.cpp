/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/Heap.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>

using namespace faiss;

// Verify that Top1BlockResultHandler::begin_multiple initializes ids_tab to -1.
// The single-query path (SingleResultHandler::begin) explicitly sets
// min_idx=-1; begin_multiple must be consistent so that
// begin_multiple/end_multiple without an intervening add_results produces the
// "no result found" sentinel -1 in ids_tab rather than leaving it
// uninitialized.
TEST(Top1BlockResultHandler, BeginMultipleInitializesIds) {
    constexpr size_t nq = 16;
    std::vector<float> dis_tab(nq, 12345.0f);
    std::vector<int64_t> ids_tab(nq, 42);

    Top1BlockResultHandler<CMax<float, int64_t>> handler(
            nq, dis_tab.data(), ids_tab.data());

    handler.begin_multiple(0, nq);
    handler.end_multiple();

    const std::vector<int64_t> expected_ids(nq, -1);
    EXPECT_EQ(ids_tab, expected_ids);

    const std::vector<float> expected_dis(nq, CMax<float, int64_t>::neutral());
    EXPECT_EQ(dis_tab, expected_dis);
}

TEST(Heap, addn_with_ids) {
    size_t n = 1000;
    size_t k = 1;
    std::vector<int64_t> heap_labels(n, -1);
    std::vector<float> heap_distances(n, 0);
    float_minheap_array_t heaps = {
            n, k, heap_labels.data(), heap_distances.data()};
    heaps.heapify();
    std::vector<int64_t> labels(n, 1);
    std::vector<float> distances(n, 0.0f);
    std::vector<int64_t> subset(n);
    std::iota(subset.begin(), subset.end(), 0);
    heaps.addn_with_ids(1, distances.data(), labels.data(), 1);
    heaps.reorder();
    EXPECT_TRUE(
            std::all_of(heap_labels.begin(), heap_labels.end(), [](int64_t i) {
                return i == 1;
            }));
}

TEST(Heap, addn_query_subset_with_ids) {
    size_t n = 20000000; // more than 2^24
    size_t k = 1;
    std::vector<int64_t> heap_labels(n, -1);
    std::vector<float> heap_distances(n, 0);
    float_minheap_array_t heaps = {
            n, k, heap_labels.data(), heap_distances.data()};
    heaps.heapify();
    std::vector<int64_t> labels(n, 1);
    std::vector<float> distances(n, 0.0f);
    std::vector<int64_t> subset(n);
    std::iota(subset.begin(), subset.end(), 0);
    heaps.addn_query_subset_with_ids(
            n, subset.data(), 1, distances.data(), labels.data(), 1);
    heaps.reorder();
    EXPECT_TRUE(
            std::all_of(heap_labels.begin(), heap_labels.end(), [](int64_t i) {
                return i == 1;
            }));
}
