/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/Index.h>
#include <faiss/utils/utils.h>

TEST(TestUtils, get_version) {
    std::string version = std::to_string(FAISS_VERSION_MAJOR) + "." +
            std::to_string(FAISS_VERSION_MINOR) + "." +
            std::to_string(FAISS_VERSION_PATCH);

    EXPECT_EQ(version, faiss::get_version());
}
