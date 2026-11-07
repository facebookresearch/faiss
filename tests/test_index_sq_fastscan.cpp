/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <vector>

#include <faiss/IndexSQFastScan.h>
#include <faiss/impl/idselector.h>

// See https://github.com/facebookresearch/faiss/issues/5831

TEST(TestSQFastScan, RemoveIdsCompactsLowBits) {
    const int d = 8;
    const int n = 100; // more than one bbs block

    std::vector<float> x(size_t(n) * d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            x[size_t(i) * d + j] = float((i * 7 + j * 13) % 256);
        }
    }

    faiss::IndexSQFastScan index(d, faiss::ScalarQuantizer::QT_8bit);
    index.train(n, x.data());
    index.add(n, x.data());
    ASSERT_EQ(index.ntotal, n);
    ASSERT_FALSE(index.lo_codes.empty());
    const size_t stride = index.lo_codes.size() / size_t(index.ntotal);

    // snapshot the low bits before the removal
    const std::vector<uint8_t> lo_before = index.lo_codes;

    // widen the shortlist so every query's own vector is reranked
    index.rerank_factor = 50;

    faiss::IDSelectorRange sel(10, 40);
    index.remove_ids(sel);
    ASSERT_EQ(index.ntotal, n - 30);
    ASSERT_EQ(index.lo_codes.size(), size_t(index.ntotal) * stride);

    // remaining original ids 0..9 and 40..99 are compacted to positions
    // 0..9 and 10..69. Every kept row of lo_codes must be the row of the
    // original vector it now represents.
    for (int j = 0; j < index.ntotal; j++) {
        const int orig = j < 10 ? j : j + 30;
        for (size_t t = 0; t < stride; t++) {
            ASSERT_EQ(
                    index.lo_codes[size_t(j) * stride + t],
                    lo_before[size_t(orig) * stride + t])
                    << "position " << j << " byte " << t;
        }
    }

    // smoke check: every remaining vector still finds itself
    std::vector<faiss::idx_t> labels(1);
    std::vector<float> distances(1);
    for (int i = 0; i < n; i++) {
        if (i >= 10 && i < 40) {
            continue;
        }
        const faiss::idx_t expected = i < 40 ? i : i - 30;
        index.search(
                1, x.data() + size_t(i) * d, 1, distances.data(), labels.data());
        ASSERT_EQ(labels[0], expected) << "vector " << i;
        ASSERT_LT(distances[0], 4.0f) << "vector " << i;
    }
}
