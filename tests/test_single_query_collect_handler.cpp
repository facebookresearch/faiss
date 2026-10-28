/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Tests for SingleQueryResultCollectHandler.
 *
 * Drives the handler through the actual fast_scan accumulate loop
 * by building an IndexPQFastScan, computing its quantized LUT,
 * and calling pq4_accumulate_loop_qbs_fixed_scaler_simd directly.
 */

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
std::unique_ptr<IndexPQFastScan> make_index() {
    auto index = std::make_unique<IndexPQFastScan>(d, 4, 4, METRIC_L2, 32);
    auto xt = make_data(nt, 1234);
    index->train(nt, xt.data());
    auto xb = make_data(nb, 4567);
    index->add(nb, xb.data());
    return index;
}

/// SingleQueryResultCollectHandler supports only 1 query.
/// Run the collect handler through the real accumulate loop for one query.
/// Returns collected (id, distance) pairs.
std::vector<std::pair<int64_t, float>> run_collect_handler(
        const IndexPQFastScan& index,
        const float* query,
        const IDSelector* sel) {
    // CMax = L2 metric (keep the largest threshold, collect distances below it)
    using C = CMax<uint16_t, int64_t>;

    size_t dim12 = index.ksub * index.M2;
    AlignedTable<uint8_t> quantized_dis_tables(dim12);
    float normalizers[2];

    FastScanDistancePostProcessing context{};
    index.compute_quantized_LUT(
            1, query, quantized_dis_tables.get(), normalizers, context);

    int qbs = pq4_preferred_qbs(1);
    AlignedTable<uint8_t> LUT(dim12);
    pq4_pack_LUT_qbs(qbs, index.M2, quantized_dis_tables.get(), LUT.get());

    std::vector<std::pair<int64_t, float>> results;
    SingleQueryResultCollectHandler<C, false> handler(
            results, index.ntotal, sel);

    handler.begin(normalizers);

    DummyScaler<> scaler;
    pq4_accumulate_loop_qbs_fixed_scaler_simd(
            qbs,
            index.ntotal2,
            index.M2,
            index.codes.get(),
            LUT.get(),
            handler,
            scaler,
            index.get_block_stride());

    handler.end();

    return results;
}

TEST(TestSingleQueryCollectHandler, BasicResultCount) {
    auto index = make_index();
    auto xq = make_data(1, 7890);

    auto results = run_collect_handler(*index, xq.data(), nullptr);

    // Without any selector, the handler should collect all nb vectors
    // (CMax::neutral() = uint16_max, so every distance passes)
    EXPECT_EQ(results.size(), nb)
            << "Expected " << nb << " results, got " << results.size();

    for (auto& [id, dist] : results) {
        EXPECT_GE(id, 0);
        EXPECT_LT(id, static_cast<int64_t>(nb));
    }

    std::unordered_set<int64_t> id_set;
    for (auto& [id, dist] : results) {
        id_set.insert(id);
    }
    EXPECT_EQ(id_set.size(), nb) << "Expected unique IDs, got duplicates";
}

TEST(TestSingleQueryCollectHandler, WithIDSelector) {
    auto index = make_index();
    auto xq = make_data(1, 7890);

    std::vector<idx_t> subset;
    for (idx_t i = 0; i < static_cast<idx_t>(nb); i += 2) {
        subset.push_back(i);
    }
    std::unordered_set<int64_t> allowed(subset.begin(), subset.end());
    IDSelectorBatch sel(subset.size(), subset.data());

    auto results = run_collect_handler(*index, xq.data(), &sel);

    EXPECT_EQ(results.size(), nb / 2)
            << "Expected " << nb / 2 << " results, got " << results.size();

    for (auto& [id, dist] : results) {
        EXPECT_TRUE(allowed.count(id) > 0)
                << "Got id " << id << " which is not in the allowed set";
    }
}

TEST(TestSingleQueryCollectHandler, EmptySelector) {
    auto index = make_index();
    auto xq = make_data(1, 7890);

    IDSelectorBatch sel(0, nullptr);

    auto results = run_collect_handler(*index, xq.data(), &sel);

    EXPECT_EQ(results.size(), 0)
            << "Expected 0 results with empty selector, got " << results.size();
}

// Excludes a whole 32-vector block (block skip in for_each_block).
TEST(TestSingleQueryCollectHandler, BlockSkipWithSelector) {
    auto index = make_index();
    auto xq = make_data(1, 7890);

    std::vector<idx_t> excluded;
    for (idx_t i = 0; i < 32; i++) {
        excluded.push_back(i);
    }
    std::unordered_set<int64_t> excluded_set(excluded.begin(), excluded.end());

    IDSelectorBatch inner_sel(excluded.size(), excluded.data());
    IDSelectorNot sel(&inner_sel);

    auto results = run_collect_handler(*index, xq.data(), &sel);

    EXPECT_EQ(results.size(), nb - 32)
            << "Expected " << nb - 32 << " results, got " << results.size();

    for (auto& [id, dist] : results) {
        EXPECT_TRUE(excluded_set.count(id) == 0) << "Got excluded id " << id;
    }
}

} // namespace
