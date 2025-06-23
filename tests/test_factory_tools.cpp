/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/factory_tools.h>
#include <faiss/index_factory.h>
#include <gtest/gtest.h>

namespace faiss {

TEST(TestFactoryTools, TestReverseIndexFactory) {
    for (const char* factory : {
                 "Flat",
                 "IMI2x5,PQ8x8",
                 "IVF32_HNSW32,SQ8",
                 "IVF8,Flat",
                 "IVF8,SQ4",
                 "IVF8,PQ4x8",
                 "LSHrt",
                 "PQ4x8",
                 "HNSW32",
                 "SQ8",
                 "SQfp16",
                 "NSG24,Flat",
                 "NSG16,SQ8",
         }) {
        std::unique_ptr<Index> index{index_factory(64, factory)};
        ASSERT_TRUE(index);
        EXPECT_EQ(factory, reverse_index_factory(index.get()));
    }
    using Case = std::pair<const char*, const char*>;
    for (auto [src, dst] : {
                 Case{"SQ8,RFlat", "SQ8,Refine(Flat)"},
                 Case{"NSG", "NSG32,Flat"},
                 Case{"NSG,PQ8", "NSG32,PQ8x8"},
         }) {
        std::unique_ptr<Index> index{index_factory(64, src)};
        ASSERT_TRUE(index);
        EXPECT_EQ(dst, reverse_index_factory(index.get()));
    }
}

} // namespace faiss
