/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/Index.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/utils.h>

TEST(TestUtils, get_version) {
    std::string version = std::to_string(FAISS_VERSION_MAJOR) + "." +
            std::to_string(FAISS_VERSION_MINOR) + "." +
            std::to_string(FAISS_VERSION_PATCH);

    EXPECT_EQ(version, faiss::get_version());
}

// IndexIVFFastScan sizes per-thread LUT tables for the largest slice and
// relies on resizes to slightly smaller slices keeping the same buffer.
TEST(TestUtils, AlignedTableResizeWithinCapacityKeepsBuffer) {
    faiss::AlignedTable<uint8_t> table;
    table.resize(1000 * 512);
    const uint8_t* ptr = table.get();
    table.resize(999 * 512);
    EXPECT_EQ(ptr, table.get());
    table.resize(1000 * 512);
    EXPECT_EQ(ptr, table.get());
}
