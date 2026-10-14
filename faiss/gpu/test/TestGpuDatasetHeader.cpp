/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Keep this translation unit ordinary C++ without CUDA, RAFT, or cuVS headers.
#include <faiss/gpu/Dataset.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissException.h>
#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace {
using faiss::gpu::Dataset;

static_assert(!std::is_copy_constructible_v<Dataset>);
static_assert(!std::is_copy_assignable_v<Dataset>);
static_assert(std::is_nothrow_move_constructible_v<Dataset>);
static_assert(std::is_nothrow_move_assignable_v<Dataset>);
static_assert(std::is_same_v<
              decltype(std::declval<const Dataset&>().getData()),
              const void*>);
static_assert(std::is_same_v<
              decltype(std::declval<const Dataset&>().borrow()),
              Dataset>);
static_assert(std::is_same_v<
              decltype(Dataset::viewHost(nullptr, 0, 4, faiss::Float32, 4)),
              Dataset>);

TEST(TestGpuDatasetHeader, CpuFloatIndexRejectsDatasetButKeepsPointerApis) {
    const float data[]{1, 2, 3, 4};
    auto dataset = Dataset::viewHost(data, 1, 4, faiss::Float32, 4);
    EXPECT_EQ(dataset.getData(), data);
    EXPECT_FALSE(dataset.ownsData());
    faiss::IndexFlatL2 index(4);
    faiss::Index* base = &index;
    EXPECT_THROW(base->train(dataset), faiss::FaissException);
    EXPECT_THROW(base->add(dataset), faiss::FaissException);
    EXPECT_NO_THROW(base->train(1, data));
    EXPECT_NO_THROW(base->add(1, data));
    EXPECT_EQ(base->ntotal, 1);
}

TEST(TestGpuDatasetHeader, CpuBinaryIndexRejectsDatasetButKeepsPointerApis) {
    const uint8_t data[]{1, 2, 3, 4};
    auto dataset = Dataset::viewHost(data, 1, 4, faiss::UInt8, 4);
    faiss::IndexBinaryFlat index(32);
    faiss::IndexBinary* base = &index;
    EXPECT_THROW(base->train(dataset), faiss::FaissException);
    EXPECT_THROW(base->add(dataset), faiss::FaissException);
    EXPECT_NO_THROW(base->train(1, data));
    EXPECT_NO_THROW(base->add(1, data));
    EXPECT_EQ(base->ntotal, 1);
}
} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
