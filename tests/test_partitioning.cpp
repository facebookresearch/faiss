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

typedef AlignedTable<uint16_t> AlignedTableUint16;

// TODO: This test fails when Faiss is compiled with
// GCC 13.2 from conda-forge with AVX2 enabled. This may be
// a GCC bug that needs to be investigated further.
// As of 16-AUG-2023 the Faiss conda packages are built
// with GCC 11.2, so the published binaries are not affected.
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
