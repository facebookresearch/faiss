/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/BinaryDistance.cuh>
#include <faiss/gpu/impl/BinaryFlatIndex.cuh>

namespace faiss {
namespace gpu {

constexpr int kBitsPerByte = 8;

BinaryFlatIndex::BinaryFlatIndex(GpuResources* res, int dim, MemorySpace space)
        : resources_(res),
          dim_(dim),
          num_(0),
          rawData_(
                  res,
                  makeSpaceAlloc(
                          AllocType::FlatData,
                          space,
                          res->getDefaultStreamCurrentDevice())) {
    // Like the CPU version, dimensions must be evenly divisible by 8 (fit into
    // an integral number of bytes)
    FAISS_ASSERT(dim % kBitsPerByte == 0);
}

/// Returns the number of vectors we contain
idx_t BinaryFlatIndex::getSize() const {
    return vectors_.getSize(0);
}

idx_t BinaryFlatIndex::getDim() const {
    return vectors_.getSize(1) * kBitsPerByte;
}

void BinaryFlatIndex::reserve(size_t numVecs, cudaStream_t stream) {
    // Like the CPU version, dimensions must be evenly divisible by 8 (fit into
    // an integral number of bytes)
    rawData_.reserve(
            numVecs * (dim_ / kBitsPerByte) * sizeof(unsigned char), stream);
}

Tensor<unsigned char, 2, true>& BinaryFlatIndex::getVectorsRef() {
    return vectors_;
}

void BinaryFlatIndex::query(
        Tensor<unsigned char, 2, true>& input,
        int k,
        Tensor<int, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    runBinaryDistance(vectors_, input, outDistances, outIndices, k, stream);
}

void BinaryFlatIndex::add(
        const unsigned char* data,
        idx_t numVecs,
        cudaStream_t stream) {
    if (numVecs == 0) {
        return;
    }

    rawData_.append(
            (char*)data,
            (size_t)(dim_ / kBitsPerByte) * numVecs * sizeof(unsigned char),
            stream,
            true /* reserve exactly */);

    num_ += numVecs;

    DeviceTensor<unsigned char, 2, true> vectors(
            (unsigned char*)rawData_.data(), {num_, (dim_ / kBitsPerByte)});
    vectors_ = std::move(vectors);
}

void BinaryFlatIndex::reset() {
    rawData_.clear();
    vectors_ = DeviceTensor<unsigned char, 2, true>();
    num_ = 0;
}

} // namespace gpu
} // namespace faiss
