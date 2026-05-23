/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <faiss/utils/sorting.h>

TEST(TestSorting, argsort_parallel_matches_serial) {
    // n > 1M to exercise the parallel merge path
    size_t n = 2000000;

    std::vector<float> vals(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < n; i++) {
        vals[i] = dist(rng);
    }

    std::vector<size_t> perm_serial(n);
    faiss::fvec_argsort(n, vals.data(), perm_serial.data());

    std::vector<size_t> perm_parallel(n);
    faiss::fvec_argsort_parallel(n, vals.data(), perm_parallel.data());

    // Permutations may differ on ties, but sorted values must match
    for (size_t i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(vals[perm_serial[i]], vals[perm_parallel[i]])
                << "mismatch at position " << i;
    }
}

TEST(TestSorting, hashtable_lookup) {
    int log2_capacity = 12;
    size_t capacity = (size_t)1 << log2_capacity;

    std::vector<int64_t> tab(capacity * 2);
    faiss::hashtable_int64_to_int64_init(log2_capacity, tab.data());

    size_t n = 200;
    std::vector<int64_t> keys(n), vals(n);
    for (size_t i = 0; i < n; i++) {
        keys[i] = static_cast<int64_t>(i * 3);
        vals[i] = static_cast<int64_t>(i + 1);
    }
    faiss::hashtable_int64_to_int64_add(
            log2_capacity, tab.data(), n, keys.data(), vals.data());

    // Interleave present and absent keys
    size_t n_query = n * 2;
    std::vector<int64_t> query_keys(n_query);
    std::vector<int64_t> expected(n_query);
    for (size_t i = 0; i < n; i++) {
        query_keys[2 * i] = keys[i];
        expected[2 * i] = vals[i];
        query_keys[2 * i + 1] =
                keys[i] + 1; // not a multiple of 3, never inserted
        expected[2 * i + 1] = -1;
    }

    std::vector<int64_t> result(n_query);
    faiss::hashtable_int64_to_int64_lookup(
            log2_capacity,
            tab.data(),
            n_query,
            query_keys.data(),
            result.data());

    for (size_t i = 0; i < n_query; i++) {
        ASSERT_EQ(result[i], expected[i])
                << "query key " << query_keys[i] << " at index " << i;
    }
}
