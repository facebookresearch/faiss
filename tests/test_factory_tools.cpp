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
                 "Flat",
                 "IMI2x5,PQ8x8",
                 "IVF32_HNSW32,SQ8",
                 "IVF8,Flat",
                 "IVF8,SQ4",
                 "IVF8,PQ4x8",
                 "IVF8,PQ4+4",
                 "LSHrt",
                 "PQ4x8",
                 "HNSW32",
                 "HNSW32,SQ8",
                 "SQ8",
                 "SQtqmse1",
                 "SQtqmse2",
                 "SQtqmse3",
                 "SQtqmse4",
                 "SQtqmse8",
                 "SQfp16",
                 "NSG24,Flat",
                 "NSG16,SQ8",
                 "RaBitQ",
                 "RaBitQ2",
                 "RaBitQ4",
                 "IVF8,RaBitQ",
                 "IVF8,RaBitQ2",
                 "IVF8,RaBitQ4",
                 "IVF8,SQtqmse8",
                 "IVF8,SQ0",
                 "EDEN",
                 "EDEN4",
                 "EDEN4BIASED",
                 "IVF8,EDEN",
                 "IVF8,EDEN4",
                 "IVF8,EDEN4BIASED",
         }) {
        std::unique_ptr<Index> index{index_factory(64, factory)};
        EXPECT_EQ(factory, reverse_index_factory(index.get()));
    }
    using Case = std::pair<const char*, const char*>;
    for (auto [src, dst] : {
                 Case{"SQ8,RFlat", "SQ8,Refine(Flat)"},
                 Case{"NSG", "NSG32,Flat"},
                 Case{"NSG,PQ8", "NSG32,PQ8x8"},
                 Case{"HNSW32,PQ16", "HNSW32,PQ16x8"},
         }) {
        std::unique_ptr<Index> index{index_factory(64, src)};
        EXPECT_EQ(dst, reverse_index_factory(index.get()));
    }
}

TEST(TestFactoryTools, TestReverseIndexFactoryPiPNN) {
    using Case = std::pair<const char*, const char*>;
    for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT}) {
        for (auto [src, expected] : {
                     Case{"PiPNN", "PiPNN64,Flat"},
                     Case{"PiPNN32,Flat", "PiPNN32,Flat"},
                     Case{"PiPNN16,SQ8", "PiPNN16,SQ8"},
                     Case{"IDMap2,PiPNN32,SQ4", "IDMap2,PiPNN32,SQ4"},
                     Case{"IVF16384(PiPNN32,Flat),SQ4",
                          "IVF16384(PiPNN32,Flat),SQ4"},
                     Case{"IVF16384_PiPNN32,SQ4", "IVF16384(PiPNN32,Flat),SQ4"},
             }) {
            SCOPED_TRACE(src);
            std::unique_ptr<Index> index{index_factory(128, src, metric)};
            const auto factory = reverse_index_factory(index.get());
            ASSERT_EQ(expected, factory);
            std::unique_ptr<Index> rebuilt{
                    index_factory(128, factory.c_str(), metric)};
            EXPECT_EQ(index->d, rebuilt->d);
            EXPECT_EQ(index->metric_type, rebuilt->metric_type);
            EXPECT_EQ(factory, reverse_index_factory(rebuilt.get()));
        }
    }
}

// IndexIDMap2 inherits IndexIDMap; verify the subclass is distinguished.
TEST(TestFactoryTools, TestReverseIndexFactoryIDMap) {
    std::unique_ptr<Index> base{index_factory(64, "Flat")};

    IndexIDMap idmap(base.get());
    EXPECT_EQ("IDMap,Flat", reverse_index_factory(&idmap));

    IndexIDMap2 idmap2(base.get());
    EXPECT_EQ("IDMap2,Flat", reverse_index_factory(&idmap2));
}

} // namespace faiss
