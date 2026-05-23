/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/partitioning.h>

using namespace faiss;

using AlignedTableUint16 = AlignedTable<uint16_t>;

// GCC 12 miscompiles the AVX2 SIMD histogram. The conda packages
// now use GCC 12.4, so we fall back to scalar in partitioning.cpp.
TEST(TestPartitioning, TestPartitioningBigRange) {
    auto n = 1024;
    AlignedTableUint16 tab(n);
    for (auto i = 0; i < n; i++) {
        tab[i] = i * 64;
    }
    int32_t hist[16]{};
    simd_histogram_16(tab.get(), n, 0, 12, hist);
    for (auto i = 0; i < 16; i++) {
        ASSERT_EQ(hist[i], 64);
    }
}
