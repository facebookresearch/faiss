/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/Dataset.h>

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/impl/FaissException.h>
#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <raft/core/error.hpp>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace {

using faiss::NumericType;
using faiss::gpu::Dataset;
using faiss::gpu::StandardGpuResources;

TEST(TestGpuDataset, HostStridesCopyLogicalColumnsWithoutChangingInput) {
    std::vector<int8_t> storage{-128, 0, 127, 42, 42, -1, 1, 2, 43, 43};
    const auto original = storage;
    auto dataset =
            Dataset::viewHost(storage.data(), 2, 3, NumericType::Int8, 5);
    EXPECT_FALSE(dataset.ownsData());
    EXPECT_FALSE(dataset.isDevice());
    EXPECT_FALSE(dataset.isCagraCompatible());
    EXPECT_EQ(dataset.getDevice(), -1);
    EXPECT_EQ(dataset.getData(), storage.data());
    EXPECT_EQ(dataset.getLayout(), Dataset::Layout::Standard);
    std::vector<int8_t> packed(6);
    dataset.copyToHost(packed.data());
    EXPECT_EQ(packed, (std::vector<int8_t>{-128, 0, 127, -1, 1, 2}));
    EXPECT_EQ(storage, original);
}

TEST(TestGpuDataset, InvalidMetadataIsRejected) {
    std::vector<float> storage(32);
    EXPECT_THROW(
            Dataset::viewHost(nullptr, 2, 4, NumericType::Float32, 4),
            faiss::FaissException);
    EXPECT_THROW(
            Dataset::viewHost(storage.data(), -1, 4, NumericType::Float32, 4),
            faiss::FaissException);
    EXPECT_THROW(
            Dataset::viewHost(storage.data(), 2, 0, NumericType::Float32, 4),
            faiss::FaissException);
    EXPECT_THROW(
            Dataset::viewHost(storage.data(), 2, 4, NumericType::Float32, 3),
            faiss::FaissException);
    EXPECT_THROW(
            Dataset::viewHost(
                    storage.data(), 2, 4, static_cast<NumericType>(99), 4),
            faiss::FaissException);
    EXPECT_THROW(
            Dataset::viewDevice(
                    storage.data(), 2, 4, NumericType::Float32, 4, 0),
            faiss::FaissException);
    EXPECT_THROW(
            Dataset::viewDevice(
                    storage.data(), 2, 4, NumericType::Float32, 4, -1),
            faiss::FaissException);
}

template <typename T>
void testPaddedCopy(NumericType type) {
    constexpr int dim = 5;
    constexpr int sourceStride = 9;
    constexpr int numRows = 3;
    std::vector<T> storage(numRows * sourceStride, T(42));
    std::vector<T> expected(numRows * dim);
    for (int row = 0; row < numRows; ++row) {
        for (int col = 0; col < dim; ++col) {
            expected[row * dim + col] = T(row * dim + col + 1);
            storage[row * sourceStride + col] = expected[row * dim + col];
        }
    }
    const auto original = storage;
    auto resources = std::make_unique<StandardGpuResources>();
    resources->noTempMemory();
    auto input =
            Dataset::viewHost(storage.data(), numRows, dim, type, sourceStride);
    auto output = Dataset::makeDevicePadded(resources.get(), input, 0);
    EXPECT_TRUE(output.ownsData());
    EXPECT_TRUE(output.isDevice());
    EXPECT_TRUE(output.isCagraCompatible());
    EXPECT_EQ(output.getLayout(), Dataset::Layout::Padded);
    EXPECT_EQ(output.getNumericType(), type);
    EXPECT_EQ(output.getNumRows(), numRows);
    EXPECT_EQ(output.getDim(), dim);
    EXPECT_EQ(
            output.getRowStride(),
            ((dim * sizeof(T) + 15) / 16) * 16 / sizeof(T));
    std::vector<T> actual(expected.size());
    output.copyToHost(actual.data());
    EXPECT_EQ(
            std::memcmp(
                    actual.data(), expected.data(), actual.size() * sizeof(T)),
            0);
    EXPECT_EQ(
            std::memcmp(
                    storage.data(),
                    original.data(),
                    storage.size() * sizeof(T)),
            0);

    std::vector<T> physical(numRows * output.getRowStride());
    ASSERT_EQ(
            cudaMemcpy(
                    physical.data(),
                    output.getData(),
                    physical.size() * sizeof(T),
                    cudaMemcpyDeviceToHost),
            cudaSuccess);
    for (int row = 0; row < numRows; ++row) {
        for (int col = dim; col < output.getRowStride(); ++col) {
            EXPECT_EQ(
                    static_cast<float>(
                            physical[row * output.getRowStride() + col]),
                    0);
        }
    }

    // After the copy completes, resource providers and host inputs can go away.
    resources.reset();
    storage.clear();
    output.copyToHost(actual.data());
    EXPECT_EQ(
            std::memcmp(
                    actual.data(), expected.data(), actual.size() * sizeof(T)),
            0);
}

TEST(TestGpuDataset, PaddedCopiesAllDtypesAndRetainsResources) {
    testPaddedCopy<float>(NumericType::Float32);
    testPaddedCopy<half>(NumericType::Float16);
    testPaddedCopy<int8_t>(NumericType::Int8);
    testPaddedCopy<uint8_t>(NumericType::UInt8);
}

TEST(TestGpuDataset, AlignedStandardInputCanBeBorrowedForSearch) {
    // 12 floats are 48 bytes, satisfying alignment without dim % 16 == 0.
    std::vector<float> storage(24, 1.0f);
    StandardGpuResources resources;
    resources.noTempMemory();
    auto host =
            Dataset::viewHost(storage.data(), 2, 12, NumericType::Float32, 12);
    auto owner = Dataset::makeDevicePadded(&resources, host, 0);
    auto standard = Dataset::viewDevice(
            owner.getData(), 2, 12, NumericType::Float32, 12, 0);
    EXPECT_EQ(standard.getLayout(), Dataset::Layout::Standard);
    EXPECT_TRUE(standard.isCagraCompatible());
    EXPECT_FALSE(standard.ownsData());
    EXPECT_EQ(standard.getData(), owner.getData());

    auto padded = Dataset::viewDevicePadded(
            owner.getData(), 2, 12, NumericType::Float32, 12, 0);
    EXPECT_EQ(padded.getData(), owner.getData());
    EXPECT_TRUE(padded.isCagraCompatible());
    EXPECT_THROW(
            Dataset::viewDevicePadded(
                    owner.getData(), 1, 4, NumericType::Float32, 8, 0),
            faiss::FaissException);
    EXPECT_THROW(
            Dataset::viewDevicePadded(
                    static_cast<const float*>(owner.getData()) + 1,
                    1,
                    4,
                    NumericType::Float32,
                    4,
                    0),
            faiss::FaissException);
    EXPECT_THROW(
            Dataset::viewHost(owner.getData(), 2, 12, NumericType::Float32, 12),
            faiss::FaissException);
}

TEST(TestGpuDataset, PreparingCompatibleDeviceInputIsRejected) {
    StandardGpuResources resources;
    resources.noTempMemory();
    std::vector<float> values(10, 1.0f);
    auto host = Dataset::viewHost(values.data(), 2, 5, NumericType::Float32, 5);
    auto owner = Dataset::makeDevicePadded(&resources, host, 0);
    resources.getResources()->syncDefaultStream(0);
    EXPECT_THROW(
            Dataset::makeDevicePadded(&resources, owner, 0), raft::logic_error);
    auto standard = Dataset::viewDevice(
            owner.getData(),
            2,
            5,
            NumericType::Float32,
            owner.getRowStride(),
            0);
    EXPECT_THROW(
            Dataset::makeDevicePadded(&resources, standard, 0),
            raft::logic_error);
}

TEST(TestGpuDataset, MovingOwnerPreservesViewAndViewDoesNotOwnStorage) {
    StandardGpuResources resources;
    resources.noTempMemory();
    std::vector<float> expected(10, 2.0f);
    auto host =
            Dataset::viewHost(expected.data(), 2, 5, NumericType::Float32, 5);
    auto owner = Dataset::makeDevicePadded(&resources, host, 0);
    auto borrowed = owner.view();
    const void* allocation = owner.getData();
    auto movedOwner = std::move(owner);
    EXPECT_TRUE(movedOwner.ownsData());
    EXPECT_FALSE(borrowed.ownsData());
    EXPECT_EQ(movedOwner.getData(), allocation);
    EXPECT_EQ(borrowed.getData(), allocation);
    std::vector<float> actual(expected.size());
    borrowed.copyToHost(actual.data());
    EXPECT_EQ(actual, expected);
    EXPECT_THROW(owner.getData(), faiss::FaissException);
    {
        auto anotherBorrow = movedOwner.view();
        EXPECT_FALSE(anotherBorrow.ownsData());
    }
    movedOwner.copyToHost(actual.data());
    EXPECT_EQ(actual, expected);
}

} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
