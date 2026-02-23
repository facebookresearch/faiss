/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "partition.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

TEST(ArgsortTest, SortsCorrectly) {
    std::mt19937 rng(12345);

    for (int trial = 0; trial < 100; trial++) {
        // Generate random size between 1 and 1000
        std::uniform_int_distribution<size_t> size_dist(1, 1000);
        size_t n = size_dist(rng);

        std::vector<float> vals(n);
        std::vector<int32_t> idxs(n);

        // Generate random float values
        std::uniform_real_distribution<float> val_dist(-1e6f, 1e6f);
        for (size_t i = 0; i < n; i++) {
            vals[i] = val_dist(rng);
        }

        // Generate distinct random IDs by shuffling a sequence
        std::iota(idxs.begin(), idxs.end(), 0);
        std::shuffle(idxs.begin(), idxs.end(), rng);

        // Make copies for reference
        std::vector<std::pair<float, int32_t>> ref_pairs(n);
        for (size_t i = 0; i < n; i++) {
            ref_pairs[i] = {vals[i], idxs[i]};
        }

        // Sort using our implementation
        argsort(n, vals.data(), idxs.data());

        // Sort reference using std::sort
        std::sort(
                ref_pairs.begin(),
                ref_pairs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

        // Check that values are sorted correctly
        for (size_t i = 0; i < n; i++) {
            EXPECT_EQ(vals[i], ref_pairs[i].first)
                    << "Trial " << trial << ": Value mismatch at index " << i
                    << " for n=" << n;
            EXPECT_EQ(idxs[i], ref_pairs[i].second)
                    << "Trial " << trial << ": Index mismatch at index " << i
                    << " for n=" << n;
        }
    }
}

TEST(ArgsortTest, SortsWithCollisions) {
    std::mt19937 rng(99999);

    for (int trial = 0; trial < 100; trial++) {
        // Generate random size between 1 and 1000
        std::uniform_int_distribution<size_t> size_dist(1, 1000);
        size_t n = size_dist(rng);

        std::vector<float> vals(n);
        std::vector<int32_t> idxs(n);

        // Generate integer values between 0 and 20 (many collisions)
        std::uniform_int_distribution<int> val_dist(0, 20);
        for (size_t i = 0; i < n; i++) {
            vals[i] = static_cast<float>(val_dist(rng));
        }

        // Generate distinct random IDs by shuffling a sequence
        std::iota(idxs.begin(), idxs.end(), 0);
        std::shuffle(idxs.begin(), idxs.end(), rng);

        // Save original idx array for validation
        std::vector<int32_t> original_idxs = idxs;

        // Sort using our implementation
        argsort(n, vals.data(), idxs.data());

        // Check that values are sorted in ascending order
        for (size_t i = 1; i < n; i++) {
            EXPECT_LE(vals[i - 1], vals[i])
                    << "Trial " << trial << ": Values not sorted at index " << i
                    << " for n=" << n << " (vals[" << i - 1
                    << "]=" << vals[i - 1] << ", vals[" << i << "]=" << vals[i]
                    << ")";
        }

        // Check that all original IDs are present (idx array is a permutation)
        std::vector<int32_t> sorted_original = original_idxs;
        std::vector<int32_t> sorted_result = idxs;
        std::sort(sorted_original.begin(), sorted_original.end());
        std::sort(sorted_result.begin(), sorted_result.end());
        EXPECT_EQ(sorted_result, sorted_original)
                << "Trial " << trial
                << ": Result IDs don't match original IDs for n=" << n;
    }
}

TEST(ArgpartitionTest, PartitionsCorrectly) {
    std::mt19937 rng(54321);

    for (int trial = 0; trial < 100; trial++) {
        // Generate random size between 1 and 1000
        std::uniform_int_distribution<size_t> size_dist(1, 1000);
        size_t n = size_dist(rng);

        // Generate random k between 1 and n
        std::uniform_int_distribution<size_t> k_dist(1, n);
        size_t k = k_dist(rng);

        std::vector<float> vals(n);
        std::vector<int32_t> idxs(n);

        // Generate random float values
        std::uniform_real_distribution<float> val_dist(-1e6f, 1e6f);
        for (size_t i = 0; i < n; i++) {
            vals[i] = val_dist(rng);
        }

        // Generate distinct random IDs
        std::iota(idxs.begin(), idxs.end(), 0);
        std::shuffle(idxs.begin(), idxs.end(), rng);

        // Make copies for reference
        std::vector<std::pair<float, int32_t>> ref_pairs(n);
        for (size_t i = 0; i < n; i++) {
            ref_pairs[i] = {vals[i], idxs[i]};
        }

        // Partition using our implementation
        argpartition(n, vals.data(), idxs.data(), k);

        // Use std::nth_element on reference
        std::nth_element(
                ref_pairs.begin(),
                ref_pairs.begin() + (n - k),
                ref_pairs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

        // Collect the k largest values from our result (sorted)
        std::vector<float> our_topk(vals.begin() + (n - k), vals.end());
        std::sort(our_topk.begin(), our_topk.end());

        // Collect the k largest values from reference (sorted)
        std::vector<float> ref_topk;
        for (size_t i = n - k; i < n; i++) {
            ref_topk.push_back(ref_pairs[i].first);
        }
        std::sort(ref_topk.begin(), ref_topk.end());

        // Check that we have the same k largest values
        ASSERT_EQ(our_topk.size(), ref_topk.size());
        for (size_t i = 0; i < k; i++) {
            EXPECT_EQ(our_topk[i], ref_topk[i])
                    << "Trial " << trial << ": Top-k mismatch at index " << i
                    << " for n=" << n << ", k=" << k;
        }
    }
}
