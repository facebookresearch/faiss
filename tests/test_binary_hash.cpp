/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexBinaryHash.h>

// These use a single 2-byte vector (d=16, code_size=2) so that the
// resulting index data is not an even multiple of sizeof(uint64_t),
// testing that the uint64_t sized hash key extraction at the end of
// the allocation boundary doesn't trigger a heap-buffer-overflow under
// ASAN. They are in C++ because Python does not control the allocation
// size exactly, so they would not fail deterministically.

TEST(BinaryHash, SmallCodeSizeRoundTrip) {
    int d = 16;
    int b = 5;
    faiss::IndexBinaryHash idx(d, b);
    idx.nflip = 1;

    int n = 1;
    std::vector<uint8_t> data(n * idx.code_size);
    data[0] = 0xAA;
    data[1] = 0x55;
    idx.add(n, data.data());
    EXPECT_EQ(idx.ntotal, n);

    int k = 1;
    std::vector<int32_t> distances(k);
    std::vector<faiss::idx_t> labels(k);
    idx.search(1, data.data(), k, distances.data(), labels.data());
    EXPECT_EQ(distances[0], 0);
    EXPECT_EQ(labels[0], 0);
}

TEST(BinaryHash, MultiHashSmallCodeSizeRoundTrip) {
    int d = 16;
    int nhash = 2;
    int b = 4; // 2 hashes * 4 bits = 8 bits <= d=16
    faiss::IndexBinaryMultiHash idx(d, nhash, b);
    idx.nflip = 1;

    int n = 1;
    std::vector<uint8_t> data(n * idx.code_size);
    data[0] = 0xAA;
    data[1] = 0x55;
    idx.add(n, data.data());
    EXPECT_EQ(idx.ntotal, n);

    int k = 1;
    std::vector<int32_t> distances(k);
    std::vector<faiss::idx_t> labels(k);
    idx.search(1, data.data(), k, distances.data(), labels.data());
    EXPECT_EQ(distances[0], 0);
    EXPECT_EQ(labels[0], 0);
}

TEST(BinaryHash, MultiHashResetClearsMaps) {
    int d = 16;
    int nhash = 2;
    int b = 4;
    faiss::IndexBinaryMultiHash idx(d, nhash, b);
    idx.nflip = 0;

    int n = 1;
    std::vector<uint8_t> data(n * idx.code_size);
    data[0] = 0xAA;
    data[1] = 0x55;
    idx.add(n, data.data());
    EXPECT_EQ(idx.ntotal, 1);
    EXPECT_GT(idx.hashtable_size(), 0u);

    idx.reset();
    EXPECT_EQ(idx.ntotal, 0);
    EXPECT_EQ(idx.hashtable_size(), 0u);

    int k = 1;
    std::vector<int32_t> distances(k);
    std::vector<faiss::idx_t> labels(k);
    idx.search(1, data.data(), k, distances.data(), labels.data());
    EXPECT_EQ(labels[0], -1);

    std::vector<uint8_t> data2(n * idx.code_size);
    data2[0] = 0x55;
    data2[1] = 0xAA;
    idx.add(n, data2.data());

    idx.search(1, data2.data(), k, distances.data(), labels.data());
    EXPECT_EQ(distances[0], 0);
    EXPECT_EQ(labels[0], 0);
}
