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

// Detect calls, including overload resolution against deleted rvalue overloads.
struct ViewDataset {
    template <typename Data>
    auto operator()(Data&& data) const
            -> decltype(std::forward<Data>(data).view());
};

struct TrainDataset {
    template <typename Index, typename Data>
    auto operator()(Index& index, Data&& data) const
            -> decltype(index.train(std::forward<Data>(data)));
};

struct AddDataset {
    template <typename Index, typename Data>
    auto operator()(Index& index, Data&& data) const
            -> decltype(index.add(std::forward<Data>(data)));
};

template <typename Operation, typename Index>
constexpr bool acceptsDatasetLvaluesOnly =
        std::is_invocable_v<Operation, Index&, Dataset&> &&
        std::is_invocable_v<Operation, Index&, const Dataset&> &&
        !std::is_invocable_v<Operation, Index&, Dataset&&> &&
        !std::is_invocable_v<Operation, Index&, const Dataset&&>;

static_assert(std::is_invocable_v<ViewDataset, Dataset&>);
static_assert(std::is_invocable_v<ViewDataset, const Dataset&>);
static_assert(!std::is_invocable_v<ViewDataset, Dataset&&>);
static_assert(!std::is_invocable_v<ViewDataset, const Dataset&&>);
static_assert(acceptsDatasetLvaluesOnly<TrainDataset, faiss::Index>);
static_assert(acceptsDatasetLvaluesOnly<AddDataset, faiss::Index>);
static_assert(acceptsDatasetLvaluesOnly<TrainDataset, faiss::IndexBinary>);
static_assert(acceptsDatasetLvaluesOnly<AddDataset, faiss::IndexBinary>);

static_assert(!std::is_copy_constructible_v<Dataset>);
static_assert(!std::is_copy_assignable_v<Dataset>);
static_assert(std::is_nothrow_move_constructible_v<Dataset>);
static_assert(std::is_nothrow_move_assignable_v<Dataset>);
static_assert(std::is_same_v<
              decltype(std::declval<const Dataset&>().getData()),
              const void*>);
static_assert(std::is_same_v<
              decltype(std::declval<const Dataset&>().view()),
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
