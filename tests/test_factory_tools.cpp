/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/factory_tools.h>
#include <faiss/index_factory.h>
#include <gtest/gtest.h>

using namespace faiss;

TEST(TestFactoryTools, TestReverseIndexFactory) {
    auto factory_string = "Flat";
    auto index = faiss::index_factory(64, factory_string);
    EXPECT_EQ(factory_string, reverse_index_factory(index));
    delete index;

    factory_string = "IMI2x5,PQ8x8";
    index = faiss::index_factory(32, factory_string);
    EXPECT_EQ(factory_string, reverse_index_factory(index));
    delete index;

    factory_string = "IVF32_HNSW32,SQ8";
    index = faiss::index_factory(64, factory_string);
    EXPECT_EQ(factory_string, reverse_index_factory(index));
    delete index;

    factory_string = "IVF8,Flat";
    index = faiss::index_factory(64, factory_string);
    EXPECT_EQ(factory_string, reverse_index_factory(index));
    delete index;

    factory_string = "IVF8,SQ4";
    index = faiss::index_factory(64, factory_string);
    EXPECT_EQ(factory_string, reverse_index_factory(index));
    delete index;

    factory_string = "IVF8,PQ4x8";
    index = faiss::index_factory(64, factory_string);
    EXPECT_EQ(factory_string, reverse_index_factory(index));
    delete index;

    factory_string = "LSHrt";
    index = faiss::index_factory(64, factory_string);
    EXPECT_EQ(factory_string, reverse_index_factory(index));
    delete index;

    factory_string = "PQ4x8";
    index = faiss::index_factory(64, factory_string);
    EXPECT_EQ(factory_string, reverse_index_factory(index));
    delete index;

    factory_string = "HNSW32";
    index = faiss::index_factory(64, factory_string);
    EXPECT_EQ(factory_string, reverse_index_factory(index));
    delete index;
}
