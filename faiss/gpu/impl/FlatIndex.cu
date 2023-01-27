/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Transpose.cuh>

namespace faiss {
namespace gpu {

FlatIndex::FlatIndex(
        GpuResources* res,
        int dim,
        bool useFloat16,
        MemorySpace space)
        : resources_(res),
          dim_(dim),
          useFloat16_(useFloat16),
          space_(space),
          num_(0),
          rawData32_(
                  res,
                  AllocInfo(
                          AllocType::FlatData,
                          getCurrentDevice(),
                          space,
                          res->getDefaultStreamCurrentDevice())),
          rawData16_(
                  res,
                  AllocInfo(
                          AllocType::FlatData,
                          getCurrentDevice(),
                          space,
                          res->getDefaultStreamCurrentDevice())) {}

bool FlatIndex::getUseFloat16() const {
    return useFloat16_;
}

/// Returns the number of vectors we contain
int FlatIndex::getSize() const {
    if (useFloat16_) {
        return vectorsHalf_.getSize(0);
    } else {
        return vectors_.getSize(0);
    }
}

int FlatIndex::getDim() const {
    return dim_;
}

void FlatIndex::reserve(size_t numVecs, cudaStream_t stream) {
    if (useFloat16_) {
        rawData16_.reserve(numVecs * dim_ * sizeof(half), stream);
    } else {
        rawData32_.reserve(numVecs * dim_ * sizeof(float), stream);
    }

    // The above may have caused a reallocation, we need to update the vector
    // types
    if (useFloat16_) {
        DeviceTensor<half, 2, true> vectors16(
                (half*)rawData16_.data(), {num_, dim_});
        vectorsHalf_ = std::move(vectors16);
    } else {
        DeviceTensor<float, 2, true> vectors32(
                (float*)rawData32_.data(), {num_, dim_});
        vectors_ = std::move(vectors32);
    }
}

Tensor<float, 2, true>& FlatIndex::getVectorsFloat32Ref() {
    // Should not call this unless we are in float32 mode
    FAISS_ASSERT(!useFloat16_);

    return vectors_;
}

Tensor<half, 2, true>& FlatIndex::getVectorsFloat16Ref() {
    // Should not call this unless we are in float16 mode
    FAISS_ASSERT(useFloat16_);

    return vectorsHalf_;
}

void FlatIndex::query(
        Tensor<float, 2, true>& input,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices,
        bool exactDistance) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (useFloat16_) {
        // We need to convert the input to float16 for comparison to ourselves
        auto inputHalf = convertTensorTemporary<float, half, 2>(
                resources_, stream, input);

        query(inputHalf,
              k,
              metric,
              metricArg,
              outDistances,
              outIndices,
              exactDistance);
    } else {
        bfKnnOnDevice(
                resources_,
                getCurrentDevice(),
                stream,
                vectors_,
                true, // is vectors row major?
                &norms_,
                input,
                true, // input is row major
                k,
                metric,
                metricArg,
                outDistances,
                outIndices,
                !exactDistance);
    }
}

void FlatIndex::query(
        Tensor<half, 2, true>& input,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices,
        bool exactDistance) {
    FAISS_ASSERT(useFloat16_);

    bfKnnOnDevice(
            resources_,
            getCurrentDevice(),
            resources_->getDefaultStreamCurrentDevice(),
            vectorsHalf_,
            true, // is vectors row major?
            &norms_,
            input,
            true, // input is row major
            k,
            metric,
            metricArg,
            outDistances,
            outIndices,
            !exactDistance);
}

void FlatIndex::computeResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& ids,
        Tensor<float, 2, true>& residuals) {
    if (useFloat16_) {
        runCalcResidual(
                vecs,
                getVectorsFloat16Ref(),
                ids,
                residuals,
                resources_->getDefaultStreamCurrentDevice());
    } else {
        runCalcResidual(
                vecs,
                getVectorsFloat32Ref(),
                ids,
                residuals,
                resources_->getDefaultStreamCurrentDevice());
    }
}

void FlatIndex::reconstruct(
        idx_t start,
        idx_t num,
        Tensor<float, 2, true>& vecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    FAISS_ASSERT(vecs.getSize(0) == num);
    FAISS_ASSERT(vecs.getSize(1) == dim_);

    if (useFloat16_) {
        runReconstruct(start, num, getVectorsFloat16Ref(), vecs, stream);
    } else {
        runReconstruct(start, num, getVectorsFloat32Ref(), vecs, stream);
    }
}

void FlatIndex::reconstruct(
        Tensor<idx_t, 1, true>& ids,
        Tensor<float, 2, true>& vecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    FAISS_ASSERT(vecs.getSize(0) == ids.getSize(0));
    FAISS_ASSERT(vecs.getSize(1) == dim_);

    if (useFloat16_) {
        runReconstruct(ids, getVectorsFloat16Ref(), vecs, stream);
    } else {
        runReconstruct(ids, getVectorsFloat32Ref(), vecs, stream);
    }
}

void FlatIndex::add(const float* data, int numVecs, cudaStream_t stream) {
    if (numVecs == 0) {
        return;
    }

    // convert and add to float16 data if needed
    if (useFloat16_) {
        // Make sure that `data` is on our device; we'll run the
        // conversion on our device
        auto devData = toDeviceTemporary<float, 2>(
                resources_,
                getCurrentDevice(),
                (float*)data,
                stream,
                {numVecs, dim_});

        auto devDataHalf = convertTensorTemporary<float, half, 2>(
                resources_, stream, devData);

        rawData16_.append(
                (char*)devDataHalf.data(),
                devDataHalf.getSizeInBytes(),
                stream,
                true /* reserve exactly */);
    } else {
        // add to float32 data
        rawData32_.append(
                (char*)data,
                (size_t)dim_ * numVecs * sizeof(float),
                stream,
                true /* reserve exactly */);
    }

    num_ += numVecs;

    if (useFloat16_) {
        DeviceTensor<half, 2, true> vectors16(
                (half*)rawData16_.data(), {(int)num_, dim_});
        vectorsHalf_ = std::move(vectors16);
    } else {
        DeviceTensor<float, 2, true> vectors32(
                (float*)rawData32_.data(), {(int)num_, dim_});
        vectors_ = std::move(vectors32);
    }

    // Precompute L2 norms of our database
    if (useFloat16_) {
        DeviceTensor<float, 1, true> norms(
                resources_,
                makeSpaceAlloc(AllocType::FlatData, space_, stream),
                {(int)num_});
        runL2Norm(vectorsHalf_, true, norms, true, stream);
        norms_ = std::move(norms);
    } else {
        DeviceTensor<float, 1, true> norms(
                resources_,
                makeSpaceAlloc(AllocType::FlatData, space_, stream),
                {(int)num_});
        runL2Norm(vectors_, true, norms, true, stream);
        norms_ = std::move(norms);
    }
}

void FlatIndex::reset() {
    rawData32_.clear();
    rawData16_.clear();
    vectors_ = DeviceTensor<float, 2, true>();
    vectorsHalf_ = DeviceTensor<half, 2, true>();
    norms_ = DeviceTensor<float, 1, true>();
    num_ = 0;
}

} // namespace gpu
} // namespace faiss
