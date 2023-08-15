/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/utils/partitioning.h>
#include <faiss/utils/AlignedTable.h>

using namespace faiss;

typedef AlignedTable<uint16_t> AlignedTableUint16;

TEST(TEST_PARTITIONING, TestPartitioning) {
    auto n = 1024;
    AlignedTableUint16 tab(n);
    for (auto i = 0; i < n; i++) {
        tab[i] = i * 64;
    }
    int32_t hist[16] {};
    simd_histogram_16(tab.get(), n, 0, 12, hist);
    for (auto i = 0; i < 16; i++) {
        ASSERT_EQ(hist[i], 64);
    }
}
