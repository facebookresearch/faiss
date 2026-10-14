/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/Dataset.h>

#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissException.h>
#include <gtest/gtest.h>

#include <cuda_fp16.h>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using faiss::NumericType;
using faiss::gpu::Dataset;
using faiss::gpu::StandardGpuResources;

static_assert(!std::is_copy_constructible_v<Dataset>);
static_assert(!std::is_copy_assignable_v<Dataset>);
static_assert(std::is_nothrow_move_constructible_v<Dataset>);
static_assert(std::is_nothrow_move_assignable_v<Dataset>);

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

    // Resource providers and host inputs can go away after preparation.
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

TEST(TestGpuDataset, PreparingAlignedDeviceInputAlwaysMakesIndependentStorage) {
    StandardGpuResources resources;
    resources.noTempMemory();
    std::vector<float> expected(24, 3.0f);
    auto second = [&]() {
        auto host = Dataset::viewHost(
                expected.data(), 2, 12, NumericType::Float32, 12);
        auto first = Dataset::makeDevicePadded(&resources, host, 0);
        auto copy = Dataset::makeDevicePadded(&resources, first, 0);
        EXPECT_NE(copy.getData(), first.getData());
        EXPECT_TRUE(copy.ownsData());
        return copy;
    }();
    std::vector<float> actual(expected.size());
    second.copyToHost(actual.data());
    EXPECT_EQ(actual, expected);
}

TEST(TestGpuDataset, MovingOwnerPreservesBorrowAndBorrowDoesNotOwnStorage) {
    StandardGpuResources resources;
    resources.noTempMemory();
    std::vector<float> expected(10, 2.0f);
    auto host =
            Dataset::viewHost(expected.data(), 2, 5, NumericType::Float32, 5);
    auto owner = Dataset::makeDevicePadded(&resources, host, 0);
    auto borrowed = owner.borrow();
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
        auto anotherBorrow = movedOwner.borrow();
        EXPECT_FALSE(anotherBorrow.ownsData());
    }
    movedOwner.copyToHost(actual.data());
    EXPECT_EQ(actual, expected);
}

TEST(TestGpuDataset, SignedInt8HnswPreparationDecodesWithoutChangingCpuCodes) {
    constexpr int dim = 5;
    const std::vector<float> values{-128, -127, -1, 0, 127, 126, 1, 2, 3, 4};
    const std::vector<int8_t> expected{-128, -127, -1, 0, 127, 126, 1, 2, 3, 4};
    StandardGpuResources resources;
    resources.noTempMemory();
    auto dataset = [&]() {
        faiss::IndexHNSWCagra source(dim, 4);
        delete source.storage;
        auto* storage = new faiss::IndexScalarQuantizer(
                dim, faiss::ScalarQuantizer::QT_8bit_direct_signed);
        source.storage = storage;
        source.set_numeric_type(NumericType::Int8);
        source.add(2, values.data());
        const std::vector<uint8_t> original(
                storage->codes.begin(), storage->codes.end());
        auto prepared =
                Dataset::makeDevicePaddedFromIndex(&resources, source, 0);
        EXPECT_EQ(
                std::vector<uint8_t>(
                        storage->codes.begin(), storage->codes.end()),
                original);
        return prepared;
    }();
    EXPECT_EQ(dataset.getNumericType(), NumericType::Int8);
    std::vector<int8_t> actual(expected.size());
    dataset.copyToHost(actual.data());
    EXPECT_EQ(actual, expected);
}

TEST(TestGpuDataset, BinaryHnswPreparationIsIndependentOfCpuIndex) {
    const std::vector<uint8_t> expected{0, 1, 255, 2, 3, 254};
    StandardGpuResources resources;
    resources.noTempMemory();
    auto dataset = [&]() {
        faiss::IndexBinaryHNSWCagra source(24, 4);
        source.add(2, expected.data());
        return Dataset::makeDevicePaddedFromIndex(&resources, source, 0);
    }();
    EXPECT_EQ(dataset.getNumericType(), NumericType::UInt8);
    EXPECT_EQ(dataset.getDim(), 3);
    EXPECT_EQ(dataset.getRowStride(), 16);
    std::vector<uint8_t> actual(expected.size());
    dataset.copyToHost(actual.data());
    EXPECT_EQ(actual, expected);
}

TEST(TestGpuDataset, DevicePreparationSupportsDifferentDevices) {
    if (faiss::gpu::getNumDevices() < 2) {
        GTEST_SKIP() << "Requires two CUDA devices";
    }
    StandardGpuResources resources;
    resources.noTempMemory();
    const std::vector<float> expected{1, 2, 3, 4, 5, 6};
    auto host =
            Dataset::viewHost(expected.data(), 2, 3, NumericType::Float32, 3);
    auto first = Dataset::makeDevicePadded(&resources, host, 0);
    EXPECT_THROW(
            Dataset::viewDevice(
                    first.getData(), 2, 3, NumericType::Float32, 4, 1),
            faiss::FaissException);
    auto second = Dataset::makeDevicePadded(&resources, first, 1);
    EXPECT_EQ(second.getDevice(), 1);
    std::vector<float> actual(expected.size());
    second.copyToHost(actual.data());
    EXPECT_EQ(actual, expected);
}

} // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
