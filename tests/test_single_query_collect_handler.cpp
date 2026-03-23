/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Tests for SingleQueryResultCollectHandler.
 *
 * Drives the handler through the actual fast_scan accumulate loop
 * by building an IndexPQFastScan, computing its quantized LUT,
 * and calling pq4_accumulate_loop_qbs_fixed_scaler_256 directly.
 */

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexPQFastScan.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/fast_scan/accumulate_loops.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>
#include <faiss/utils/random.h>

using namespace faiss;
using namespace faiss::simd_result_handlers;

namespace {

constexpr int d = 32;
constexpr size_t nt = 5000;
constexpr size_t nb = 200;

std::vector<float> make_data(size_t n, int64_t seed) {
    std::vector<float> data(n * d);
    rand_smooth_vectors(n, d, data.data(), seed);
    return data;
}

/// Build, train and populate an IndexPQFastScan with bbs=32.
std::unique_ptr<IndexPQFastScan> make_index(size_t db_size = nb) {
    auto index = std::make_unique<IndexPQFastScan>(d, 4, 4, METRIC_L2, 32);
    auto xt = make_data(nt, 1234);
    index->train(nt, xt.data());
    auto xb = make_data(db_size, 4567);
    index->add(db_size, xb.data());
    return index;
}

/// The following code relies on 1 query, because
///   SingleQueryResultCollectHandler is hardcoded to use
///   only 1 query.

/// Run the collect handler through the real accumulate loop for one query.
/// Returns collected (id, distance) pairs.
std::vector<std::pair<int64_t, float>> run_collect_handler(
        const IndexPQFastScan& index,
        const float* query,
        const IDSelector* sel) {
    // CMax = L2 metric (keep the largest threshold, collect distances below it)
    using C = CMax<uint16_t, int64_t>;

    // 1. Compute quantized LUT + normalizers for 1 query
    size_t dim12 = index.ksub * index.M2;
    AlignedTable<uint8_t> quantized_dis_tables(dim12);
    float normalizers[2];

    FastScanDistancePostProcessing context{};
    index.compute_quantized_LUT(
            1, query, quantized_dis_tables.get(), normalizers, context);

    // 2. Pack the LUT
    int qbs = pq4_preferred_qbs(1);
    AlignedTable<uint8_t> LUT(dim12);
    pq4_pack_LUT_qbs(qbs, index.M2, quantized_dis_tables.get(), LUT.get());

    // 3. Create the handler
    std::vector<std::pair<int64_t, float>> results;
    SingleQueryResultCollectHandler<C, false> handler(
            results, index.ntotal, sel);

    // 4. Begin + set normalizers
    handler.begin(normalizers);

    // 5. Run the accumulate loop
    DummyScaler<> scaler;
    pq4_accumulate_loop_qbs_fixed_scaler_256(
            qbs,
            index.ntotal2,
            index.M2,
            index.codes.get(),
            LUT.get(),
            handler,
            scaler,
            index.get_block_stride());

    // 6. End (applies normalizer scaling)
    handler.end();

    return results;
}

/*************************************************************
 * Test: handler collects results and count matches expectation
 *************************************************************/

TEST(TestSingleQueryCollectHandler, BasicResultCount) {
    auto index = make_index();
    auto xq = make_data(1, 7890);

    auto results = run_collect_handler(*index, xq.data(), nullptr);

    // Without any selector, the handler should collect all nb vectors
    // (CMax::neutral() = uint16_max, so every distance passes)
    EXPECT_EQ(results.size(), nb)
            << "Expected " << nb << " results, got " << results.size();

    // Each result should have a valid ID in [0, nb)
    for (auto& [id, dist] : results) {
        EXPECT_GE(id, 0);
        EXPECT_LT(id, static_cast<int64_t>(nb));
    }

    // IDs should be unique
    std::unordered_set<int64_t> id_set;
    for (auto& [id, dist] : results) {
        id_set.insert(id);
    }
    EXPECT_EQ(id_set.size(), nb) << "Expected unique IDs, got duplicates";
}

/*************************************************************
 * Test: handler with IDSelector only returns allowed IDs
 *************************************************************/

TEST(TestSingleQueryCollectHandler, WithIDSelector) {
    auto index = make_index();
    auto xq = make_data(1, 7890);

    // Allow only even IDs
    std::vector<idx_t> subset;
    for (idx_t i = 0; i < static_cast<idx_t>(nb); i += 2) {
        subset.push_back(i);
    }
    std::unordered_set<int64_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    auto results = run_collect_handler(*index, xq.data(), &sel);

    // Should collect exactly the allowed IDs (nb/2)
    EXPECT_EQ(results.size(), nb / 2)
            << "Expected " << nb / 2 << " results, got " << results.size();

    // Every result must be in the allowed set
    for (auto& [id, dist] : results) {
        EXPECT_TRUE(allowed.count(id) > 0)
                << "Got id " << id << " which is not in the allowed set";
    }
}

/*************************************************************
 * Test: empty selector -> no results
 *************************************************************/

TEST(TestSingleQueryCollectHandler, EmptySelector) {
    auto index = make_index();
    auto xq = make_data(1, 7890);

    // Empty batch = accept nothing
    IDSelectorBatch sel(0, nullptr);

    auto results = run_collect_handler(*index, xq.data(), &sel);

    EXPECT_EQ(results.size(), 0)
            << "Expected 0 results with empty selector, got " << results.size();
}

/*************************************************************
 * Test: results are consistent with normal search
 *
 * The top-k from a normal search should be a subset of the
 * collected results (since the handler collects everything).
 *************************************************************/

TEST(TestSingleQueryCollectHandler, ConsistentWithSearch) {
    auto index = make_index();
    auto xq = make_data(1, 7890);

    // Normal top-10 search
    int k = 10;
    std::vector<float> Dref(k);
    std::vector<idx_t> Iref(k);
    index->search(1, xq.data(), k, Dref.data(), Iref.data());

    // Collect all results
    auto results = run_collect_handler(*index, xq.data(), nullptr);

    // Build a map of id -> distance from collected results
    std::unordered_map<int64_t, float> collected;
    for (auto& [id, dist] : results) {
        collected[id] = dist;
    }

    // Every reference result should appear in the collected set
    for (int j = 0; j < k; j++) {
        if (Iref[j] >= 0) {
            EXPECT_TRUE(collected.count(Iref[j]) > 0)
                    << "Top-" << k << " result id " << Iref[j]
                    << " not found in collected results";
        }
    }
}

/*************************************************************
 * Test: ntotal not a multiple of 32
 *************************************************************/

TEST(TestSingleQueryCollectHandler, NonAlignedNtotal) {
    size_t db_size = 77; // not a multiple of 32
    auto index = make_index(db_size);
    auto xq = make_data(1, 7890);

    auto results = run_collect_handler(*index, xq.data(), nullptr);

    // Should collect exactly db_size results
    EXPECT_EQ(results.size(), db_size)
            << "Expected " << db_size << " results, got " << results.size();

    // All IDs should be in [0, db_size)
    for (auto& [id, dist] : results) {
        EXPECT_GE(id, 0);
        EXPECT_LT(id, static_cast<int64_t>(db_size));
    }
}

/*************************************************************
 * Test: IDSelector that excludes entire 32-vector blocks
 * (exercises the block-skip optimization)
 *************************************************************/

TEST(TestSingleQueryCollectHandler, BlockSkipWithSelector) {
    auto index = make_index();
    auto xq = make_data(1, 7890);

    // Exclude the first 32 IDs (one full block)
    std::vector<idx_t> excluded;
    for (idx_t i = 0; i < 32; i++) {
        excluded.push_back(i);
    }
    std::unordered_set<int64_t> excluded_set(excluded.begin(), excluded.end());

    IDSelectorBatch inner_sel(excluded.size(), excluded.data());
    IDSelectorNot sel(&inner_sel);

    auto results = run_collect_handler(*index, xq.data(), &sel);

    // Should collect nb - 32 results
    EXPECT_EQ(results.size(), nb - 32)
            << "Expected " << nb - 32 << " results, got " << results.size();

    // No excluded ID should appear
    for (auto& [id, dist] : results) {
        EXPECT_TRUE(excluded_set.count(id) == 0) << "Got excluded id " << id;
    }
}

} // namespace
