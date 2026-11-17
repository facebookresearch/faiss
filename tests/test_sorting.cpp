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
    size_t n = 100000;

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
