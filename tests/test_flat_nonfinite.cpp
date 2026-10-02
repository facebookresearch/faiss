/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>

// Issue 5684: a NaN or Inf component is rejected at add time instead of
// being dropped from every search result later.
TEST(IndexFlat, AddRejectsNonFinite) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (float bad : {nan, inf, -inf}) {
        faiss::IndexFlatL2 index(2);
        std::vector<float> xb = {0.5f, 1.0f, bad, 1.0f};
        EXPECT_THROW(index.add(2, xb.data()), faiss::FaissException);
        EXPECT_EQ(index.ntotal, 0);
    }
    faiss::IndexFlatIP ip(2);
    std::vector<float> xb = {nan, 1.0f};
    EXPECT_THROW(ip.add(1, xb.data()), faiss::FaissException);
}

// Finite data still adds and returns min(k, ntotal) results.
TEST(IndexFlat, FiniteSearchReturnsAllRows) {
    faiss::IndexFlatL2 index(2);
    std::vector<float> xb = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 1.0f};
    index.add(3, xb.data());
    std::vector<float> q = {0.5f, 1.0f};
    std::vector<float> D(5);
    std::vector<faiss::idx_t> I(5);
    index.search(1, q.data(), 5, D.data(), I.data());
    EXPECT_EQ(I[0], 2);
    EXPECT_GE(I[1], 0);
    EXPECT_GE(I[2], 0);
    EXPECT_EQ(I[3], -1);
    EXPECT_EQ(I[4], -1);
}

// Issue 5683: finite inputs whose squared distance overflows float32 used
// to come back as -1 slots with no error.
TEST(IndexFlat, L2OverflowIsReported) {
    for (float a : {2.5e19f, 3e19f}) {
        faiss::IndexFlatL2 index(2);
        std::vector<float> xb = {a, 0.0f, 0.0f, a};
        index.add(2, xb.data());
        std::vector<float> q = {0.6f * a, 0.8f * a};
        std::vector<float> D(2);
        std::vector<faiss::idx_t> I(2);
        EXPECT_THROW(
                index.search(1, q.data(), 2, D.data(), I.data()),
                faiss::FaissException);
    }
}

// A selector may legitimately return fewer than k results.
TEST(IndexFlat, SelectorMayReturnFewer) {
    faiss::IndexFlatL2 index(2);
    std::vector<float> xb = {1.0f, 0.0f, 0.0f, 1.0f};
    index.add(2, xb.data());
    faiss::IDSelectorRange sel(0, 1);
    faiss::SearchParameters params;
    params.sel = &sel;
    std::vector<float> q = {0.6f, 0.8f};
    std::vector<float> D(2);
    std::vector<faiss::idx_t> I(2);
    index.search(1, q.data(), 2, D.data(), I.data(), &params);
    EXPECT_EQ(I[0], 0);
    EXPECT_EQ(I[1], -1);
}
