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

// IndexIVFFastScan sizes per-thread LUTs for the largest slice; a smaller
// slice whose size rounds to a smaller capacity must keep the same buffer.
TEST(TestUtils, AlignedTableKeepCapacity) {
    faiss::AlignedTable<uint8_t> table;
    table.keep_capacity = true;
    table.resize(3 * 32768);
    const uint8_t* ptr = table.get();
    table.resize(2 * 32768);
    EXPECT_EQ(ptr, table.get());
    EXPECT_EQ(2 * 32768, table.size());
}
