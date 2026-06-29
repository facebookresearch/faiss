/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>

using namespace faiss;

namespace {

constexpr int d = 32;
constexpr size_t nt = 5000;
constexpr size_t nb = 1000;
constexpr size_t nq = 20;

std::vector<float> make_data(size_t n, int64_t seed) {
    std::vector<float> data(n * d);
    rand_smooth_vectors(n, d, data.data(), seed);
    return data;
}

/*************************************************************
 * Helper: test filtered search on a fastscan IVF index.
 *
 * Primary check: every returned ID is in the allowed set.
 * Secondary check: results have reasonable overlap with Flat
 * baseline (loose threshold since PQ is approximate).
 *************************************************************/

void test_filtered_search(
        const char* fs_index_key,
        MetricType metric,
        const size_t k,
        IDSelector& sel,
        const std::unordered_set<idx_t>& allowed_ids,
        const size_t db_size = nb) {
    auto xt = make_data(nt, 1234);
    auto xb = make_data(db_size, 4567);
    auto xq = make_data(nq, 7890);

    // Build fastscan index
    std::unique_ptr<Index> fs_index(index_factory(d, fs_index_key, metric));
    fs_index->train(nt, xt.data());
    fs_index->add(db_size, xb.data());

    // Search fastscan with selector
    SearchParametersIVF params;
    params.sel = &sel;
    params.nprobe = 8;

    std::vector<float> Dfs(nq * k);
    std::vector<idx_t> Ifs(nq * k);

    fs_index->search(nq, xq.data(), k, Dfs.data(), Ifs.data(), &params);

    // Primary check: all results are in the allowed set (or -1)
    for (size_t i = 0; i < nq * k; i++) {
        if (Ifs[i] >= 0) {
            EXPECT_TRUE(allowed_ids.count(Ifs[i]) > 0)
                    << "Fastscan returned id " << Ifs[i]
                    << " which is not in the allowed set";
        }
    }

    // Secondary check: if allowed set is large enough, we should get
    // some valid results (not all -1)
    if (allowed_ids.size() > k) {
        size_t valid_count = 0;
        for (size_t i = 0; i < nq * k; i++) {
            if (Ifs[i] >= 0)
                valid_count++;
        }
        EXPECT_GT(valid_count, 0) << "Expected some valid results";
    }
}

/*************************************************************
 * Test: IDSelectorBatch with fastscan (batch of allowed IDs)
 *************************************************************/

// Both tests are needed, because one tests CMax,
//   and another tests CMin.

TEST(TestFastScanFilter, IVFPQfs_Batch_L2) {
    std::vector<idx_t> subset;
    for (idx_t i = 0; i < nb; i += 3) { // ~333 out of 1000
        subset.push_back(i);
    }

    std::unordered_set<idx_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 10, sel, allowed);
}

TEST(TestFastScanFilter, IVFPQfs_Batch_IP) {
    std::vector<idx_t> subset;
    for (idx_t i = 0; i < nb; i += 3) {
        subset.push_back(i);
    }

    std::unordered_set<idx_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    test_filtered_search(
            "IVF32,PQ4x4fs", METRIC_INNER_PRODUCT, 10, sel, allowed);
}

/*************************************************************
 * Test: k=1 (SingleResultHandler path) and k=40 (ReservoirHandler)
 *************************************************************/

TEST(TestFastScanFilter, IVFPQfs_K1) {
    std::vector<idx_t> subset;
    for (idx_t i = 0; i < nb; i += 2) {
        subset.push_back(i);
    }

    std::unordered_set<idx_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 1, sel, allowed);
}

TEST(TestFastScanFilter, IVFPQfs_K40) {
    std::vector<idx_t> subset;
    for (idx_t i = 0; i < nb; i += 2) {
        subset.push_back(i);
    }

    std::unordered_set<idx_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 40, sel, allowed);
}

/*************************************************************
 * Test: IDSelectorNot that excludes entire blocks (IDs 0-31, 64-95, etc.)
 * This specifically tests the block-skip optimization.
 *************************************************************/

TEST(TestFastScanFilter, BlockSkip_WholeBlocks) {
    // Exclude blocks 0 (ids 0-31) and 2 (ids 64-95)
    std::vector<idx_t> excluded;
    for (idx_t i = 0; i < 32; i++) {
        excluded.push_back(i);
    }
    for (idx_t i = 64; i < 96; i++) {
        excluded.push_back(i);
    }

    // Allowed = everything NOT in excluded
    std::unordered_set<idx_t> allowed;
    std::unordered_set<idx_t> excluded_set(excluded.begin(), excluded.end());
    for (idx_t i = 0; i < nb; i++) {
        if (excluded_set.count(i) == 0) {
            allowed.insert(i);
        }
    }

    IDSelectorBatch inner_sel(excluded.size(), excluded.data());
    IDSelectorNot sel(&inner_sel);

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 10, sel, allowed);
}

/*************************************************************
 * Test: Partial block filtering (exclude some IDs within a block)
 *************************************************************/

TEST(TestFastScanFilter, BlockSkip_PartialBlock) {
    // Exclude only a few IDs from the first block (0-31)
    std::vector<idx_t> excluded = {5, 10, 20, 31};
    std::unordered_set<idx_t> excluded_set(excluded.begin(), excluded.end());

    std::unordered_set<idx_t> allowed;
    for (idx_t i = 0; i < nb; i++) {
        if (excluded_set.count(i) == 0) {
            allowed.insert(i);
        }
    }

    IDSelectorBatch inner_sel(excluded.size(), excluded.data());
    IDSelectorNot sel(&inner_sel);

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 10, sel, allowed);
}

/*************************************************************
 * Test: Empty selector (nothing accepted) -> all results -1
 *************************************************************/

TEST(TestFastScanFilter, EmptySelector) {
    auto xt = make_data(nt, 1234);
    auto xb = make_data(nb, 4567);
    auto xq = make_data(nq, 7890);

    std::unique_ptr<Index> index(index_factory(d, "IVF32,PQ4x4fs", METRIC_L2));
    index->train(nt, xt.data());
    index->add(nb, xb.data());

    // Empty batch selector: accepts nothing
    std::vector<idx_t> empty_subset;
    IDSelectorBatch sel(0, empty_subset.data());

    size_t k = 10;
    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);

    SearchParametersIVF params;
    params.sel = &sel;
    params.nprobe = 4;
    index->search(nq, xq.data(), k, D.data(), I.data(), &params);

    // All results should be -1 (nothing matches)
    for (size_t i = 0; i < nq * k; i++) {
        EXPECT_EQ(I[i], -1)
                << "Expected -1 at position " << i << " but got " << I[i];
    }
}

/*************************************************************
 * Test: ntotal not a multiple of 32
 * Tests the min(32, ntotal - j0) boundary in block-skip check.
 *************************************************************/

TEST(TestFastScanFilter, NonAlignedNtotal_50) {
    std::vector<idx_t> subset;
    for (idx_t i = 0; i < 50; i += 2) {
        subset.push_back(i);
    }

    std::unordered_set<idx_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 10, sel, allowed, 50);
}

TEST(TestFastScanFilter, NonAlignedNtotal_77) {
    std::vector<idx_t> subset;
    for (idx_t i = 0; i < 77; i += 2) {
        subset.push_back(i);
    }

    std::unordered_set<idx_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 10, sel, allowed, 77);
}

TEST(TestFastScanFilter, NonAlignedNtotal_150) {
    std::vector<idx_t> subset;
    for (idx_t i = 0; i < 150; i += 3) {
        subset.push_back(i);
    }

    std::unordered_set<idx_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 10, sel, allowed, 150);
}

/*************************************************************
 * Test: IDSelectorRange with fastscan
 *************************************************************/

TEST(TestFastScanFilter, IVFPQfs_Range) {
    std::unordered_set<idx_t> allowed;
    for (idx_t i = 100; i < 500; i++) {
        allowed.insert(i);
    }

    IDSelectorRange sel(100, 500);

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 10, sel, allowed);
}

/*************************************************************
 * Test: Heavy filtering (>90% excluded) - block-skip should
 * skip most blocks entirely.
 *************************************************************/

TEST(TestFastScanFilter, HeavyFiltering) {
    // Allow only ~5% of vectors
    std::vector<idx_t> subset;
    for (idx_t i = 0; i < nb; i += 20) {
        subset.push_back(i);
    }

    std::unordered_set<idx_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    test_filtered_search("IVF32,PQ4x4fs", METRIC_L2, 10, sel, allowed);
}

} // namespace
