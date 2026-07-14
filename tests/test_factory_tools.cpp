/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIDMap.h>
#include <faiss/factory_tools.h>
#include <faiss/index_factory.h>
#include <gtest/gtest.h>

namespace faiss {

TEST(TestFactoryTools, TestReverseIndexFactory) {
    for (const char* factory : {
                 "Flat",      "IMI2x5,PQ8x8", "IVF32_HNSW32,SQ8",
                 "IVF8,Flat", "IVF8,SQ4",     "IVF8,PQ4x8",
                 "LSHrt",     "PQ4x8",        "HNSW32",
                 "SQ8",       "SQtqmse1",     "SQtqmse2",
                 "SQtqmse3",  "SQtqmse4",     "SQtqmse8",
                 "SQfp16",    "NSG24,Flat",   "NSG16,SQ8",
                 "RaBitQ",    "IVF8,RaBitQ",  "IVF8,SQtqmse8",
                 "EDEN",      "EDEN4",        "EDEN4BIASED",
                 "IVF8,EDEN", "IVF8,EDEN4",   "IVF8,EDEN4BIASED",
                 "IVF8,SQ0",
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

// IndexIDMap2 inherits IndexIDMap; verify the subclass is distinguished.
TEST(TestFactoryTools, TestReverseIndexFactoryIDMap) {
    std::unique_ptr<Index> base{index_factory(64, "Flat")};
    ASSERT_TRUE(base);

    IndexIDMap idmap(base.get());
    EXPECT_EQ("IDMap,Flat", reverse_index_factory(&idmap));

    IndexIDMap2 idmap2(base.get());
    EXPECT_EQ("IDMap2,Flat", reverse_index_factory(&idmap2));
}

} // namespace faiss
