/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests for RangeResultHandler::add_result return values.

#include <set>

#include <gtest/gtest.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/ResultHandler.h>

// RangeResultHandler::add_result must return true when a result is added
// (distance passes the threshold) so that callers in expanded_scanners.h
// correctly track nheap_updates.
TEST(IVFEarlyTermination, RangeResultHandlerAddResultReturnValue) {
    using C = faiss::CMax<float, faiss::idx_t>;

    faiss::RangeSearchResult rsr(1);
    faiss::RangeSearchPartialResult pres(&rsr);
    auto& qr = pres.new_result(0);

    faiss::RangeResultHandler<C> handler(&qr, /*threshold=*/5.0f);

    // CMax::cmp(5.0, 3.0) is true: distance within radius, should add
    EXPECT_TRUE(handler.add_result(3.0f, 42));

    // CMax::cmp(5.0, 5.0) is false: at boundary, should not add
    EXPECT_FALSE(handler.add_result(5.0f, 43));

    // CMax::cmp(5.0, 10.0) is false: distance exceeds radius
    EXPECT_FALSE(handler.add_result(10.0f, 44));

    EXPECT_TRUE(handler.add_result(1.0f, 45));

    pres.finalize();
    ASSERT_EQ(rsr.lims[1], size_t(2));
    std::set<faiss::idx_t> ids(rsr.labels, rsr.labels + 2);
    EXPECT_EQ(ids, (std::set<faiss::idx_t>{42, 45}));
    std::set<float> dists(rsr.distances, rsr.distances + 2);
    EXPECT_EQ(dists, (std::set<float>{1.0f, 3.0f}));
}

// Same test with CMin (inner product semantics): adds when threshold < dis.
TEST(IVFEarlyTermination, RangeResultHandlerAddResultCMin) {
    using C = faiss::CMin<float, faiss::idx_t>;

    faiss::RangeSearchResult rsr(1);
    faiss::RangeSearchPartialResult pres(&rsr);
    auto& qr = pres.new_result(0);

    faiss::RangeResultHandler<C> handler(&qr, /*threshold=*/5.0f);

    // CMin::cmp(5.0, 10.0) is true: dis above threshold, should add
    EXPECT_TRUE(handler.add_result(10.0f, 100));

    // CMin::cmp(5.0, 5.0) is false: at boundary, should not add
    EXPECT_FALSE(handler.add_result(5.0f, 101));

    // CMin::cmp(5.0, 3.0) is false: dis below threshold
    EXPECT_FALSE(handler.add_result(3.0f, 102));

    EXPECT_TRUE(handler.add_result(7.0f, 103));

    pres.finalize();
    ASSERT_EQ(rsr.lims[1], size_t(2));
    std::set<faiss::idx_t> ids(rsr.labels, rsr.labels + 2);
    EXPECT_EQ(ids, (std::set<faiss::idx_t>{100, 103}));
    std::set<float> dists(rsr.distances, rsr.distances + 2);
    EXPECT_EQ(dists, (std::set<float>{7.0f, 10.0f}));
}
