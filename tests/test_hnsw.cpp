/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/impl/HNSW.h>

int reference_pop_min(faiss::HNSW::MinimaxHeap& heap, float* vmin_out) {
    assert(heap.k > 0);
    // returns min. This is an O(n) operation
    int i = heap.k - 1;
    while (i >= 0) {
        if (heap.ids[i] != -1)
            break;
        i--;
    }
    if (i == -1)
        return -1;
    int imin = i;
    float vmin = heap.dis[i];
    i--;
    while (i >= 0) {
        if (heap.ids[i] != -1 && heap.dis[i] < vmin) {
            vmin = heap.dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out)
        *vmin_out = vmin;
    int ret = heap.ids[imin];
    heap.ids[imin] = -1;
    --heap.nvalid;

    return ret;
}

void test_popmin(int heap_size, int amount_to_put) {
    // create a heap
    faiss::HNSW::MinimaxHeap mm_heap(heap_size);

    using storage_idx_t = faiss::HNSW::storage_idx_t;

    std::default_random_engine rng(123 + heap_size * amount_to_put);
    std::uniform_int_distribution<storage_idx_t> u(0, 65536);
    std::uniform_real_distribution<float> uf(0, 1);

    // generate random unique indices
    std::unordered_set<storage_idx_t> indices;
    while (indices.size() < amount_to_put) {
        const storage_idx_t index = u(rng);
        indices.insert(index);
    }

    // put ones into the heap
    for (const auto index : indices) {
        float distance = uf(rng);
        if (distance >= 0.7f) {
            // add infinity values from time to time
            distance = std::numeric_limits<float>::infinity();
        }
        mm_heap.push(index, distance);
    }

    // clone the heap
    faiss::HNSW::MinimaxHeap cloned_mm_heap = mm_heap;

    // takes ones out one by one
    while (mm_heap.size() > 0) {
        // compare heaps
        ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
        ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
        ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
        ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
        ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);

        // use the reference pop_min for the cloned heap
        float cloned_vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t cloned_vmin_idx =
                reference_pop_min(cloned_mm_heap, &cloned_vmin_dis);

        float vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t vmin_idx = mm_heap.pop_min(&vmin_dis);

        // compare returns
        ASSERT_EQ(vmin_dis, cloned_vmin_dis);
        ASSERT_EQ(vmin_idx, cloned_vmin_idx);
    }

    // compare heaps again
    ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
    ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
    ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
    ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
    ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);
}

void test_popmin_identical_distances(
        int heap_size,
        int amount_to_put,
        const float distance) {
    // create a heap
    faiss::HNSW::MinimaxHeap mm_heap(heap_size);

    using storage_idx_t = faiss::HNSW::storage_idx_t;

    std::default_random_engine rng(123 + heap_size * amount_to_put);
    std::uniform_int_distribution<storage_idx_t> u(0, 65536);

    // generate random unique indices
    std::unordered_set<storage_idx_t> indices;
    while (indices.size() < amount_to_put) {
        const storage_idx_t index = u(rng);
        indices.insert(index);
    }

    // put ones into the heap
    for (const auto index : indices) {
        mm_heap.push(index, distance);
    }

    // clone the heap
    faiss::HNSW::MinimaxHeap cloned_mm_heap = mm_heap;

    // takes ones out one by one
    while (mm_heap.size() > 0) {
        // compare heaps
        ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
        ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
        ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
        ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
        ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);

        // use the reference pop_min for the cloned heap
        float cloned_vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t cloned_vmin_idx =
                reference_pop_min(cloned_mm_heap, &cloned_vmin_dis);

        float vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t vmin_idx = mm_heap.pop_min(&vmin_dis);

        // compare returns
        ASSERT_EQ(vmin_dis, cloned_vmin_dis);
        ASSERT_EQ(vmin_idx, cloned_vmin_idx);
    }

    // compare heaps again
    ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
    ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
    ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
    ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
    ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);
}

TEST(HNSW, Test_popmin) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32, 64, 128};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin(size, amount);
        }
    }
}

TEST(HNSW, Test_popmin_identical_distances) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin_identical_distances(size, amount, 1.0f);
        }
    }
}

TEST(HNSW, Test_popmin_infinite_distances) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin_identical_distances(
                    size, amount, std::numeric_limits<float>::infinity());
        }
    }
}
