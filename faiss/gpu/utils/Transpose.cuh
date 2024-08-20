/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <stdint.h>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

template <typename T>
struct TensorInfo {
    static constexpr int kMaxDims = 8;

    T* data;
    idx_t sizes[kMaxDims];
    idx_t strides[kMaxDims];
    int dims;
};

template <typename T, int Dim>
struct TensorInfoOffset {
    __device__ inline static idx_t get(
            const TensorInfo<T>& info,
            idx_t linearId) {
        idx_t offset = 0;

#pragma unroll
        for (int i = Dim - 1; i >= 0; --i) {
            auto curDimIndex = linearId % info.sizes[i];
            auto curDimOffset = curDimIndex * info.strides[i];

            offset += curDimOffset;

            if (i > 0) {
                linearId /= info.sizes[i];
            }
        }

        return offset;
    }
};

template <typename T>
struct TensorInfoOffset<T, -1> {
    __device__ inline static idx_t get(
            const TensorInfo<T>& info,
            idx_t linearId) {
        return linearId;
    }
};

template <typename T, int Dim>
TensorInfo<T> getTensorInfo(const Tensor<T, Dim, true>& t) {
    TensorInfo<T> info;

    for (int i = 0; i < Dim; ++i) {
        info.sizes[i] = t.getSize(i);
        info.strides[i] = t.getStride(i);
    }

    info.data = t.data();
    info.dims = Dim;

    return info;
}

template <typename T, int DimInput, int DimOutput>
__global__ void transposeAny(
        TensorInfo<T> input,
        TensorInfo<T> output,
        idx_t totalSize) {
    for (idx_t i = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; i < totalSize;
         i += gridDim.x * blockDim.x) {
        auto inputOffset = TensorInfoOffset<T, DimInput>::get(input, i);
        auto outputOffset = TensorInfoOffset<T, DimOutput>::get(output, i);

#if __CUDA_ARCH__ >= 350 || defined(USE_AMD_ROCM)
        output.data[outputOffset] = __ldg(&input.data[inputOffset]);
#else
        output.data[outputOffset] = input.data[inputOffset];
#endif
    }
}

// Transpose contiguous t1 t2 i1 -> t2 t1 i1
template <typename T>
__global__ void transposeOuter(
        const T* in,
        T* out,
        idx_t t1,
        idx_t t2,
        idx_t i1) {
    idx_t gt2 = blockIdx.x;
    for (idx_t gt1 = blockIdx.y; gt1 < t1; gt1 += gridDim.y) {
        auto curIn = in + i1 * (gt1 * t2 + gt2);
        auto curOut = out + i1 * (gt2 * t1 + gt1);

        for (idx_t i = threadIdx.x; i < i1; i += blockDim.x) {
            curOut[i] = curIn[i];
        }
    }
}

/// Performs an out-of-place transposition between any two dimensions.
/// Best performance is if the transposed dimensions are not
/// innermost, since the reads and writes will be coalesced.
/// Could include a shared memory transposition if the dimensions
/// being transposed are innermost, but would require support for
/// arbitrary rectangular matrices.
/// This linearized implementation seems to perform well enough,
/// especially for cases that we care about (outer dimension
/// transpositions).
template <typename T, int Dim>
void runTransposeAny(
        Tensor<T, Dim, true>& in,
        int dim1,
        int dim2,
        Tensor<T, Dim, true>& out,
        cudaStream_t stream) {
    static_assert(Dim <= TensorInfo<T>::kMaxDims, "too many dimensions");

    FAISS_ASSERT(dim1 != dim2);
    FAISS_ASSERT(dim1 < Dim && dim2 < Dim);

    // Rearrange dim1 and dim2 in increasing order in order to see if this is an
    // outer dimension transposition (below)
    if (dim1 > dim2) {
        std::swap(dim1, dim2);
    }

    idx_t outSize[Dim];

    for (int i = 0; i < Dim; ++i) {
        outSize[i] = in.getSize(i);
    }

    std::swap(outSize[dim1], outSize[dim2]);

    for (int i = 0; i < Dim; ++i) {
        FAISS_ASSERT(out.getSize(i) == outSize[i]);
    }

    idx_t maxThreads = getMaxThreadsCurrentDevice();
    auto totalSize = in.numElements();

    // Is this a transposition of the two outer dimensions?
    bool isTransposeOuter = (Dim >= 3) && (dim1 == 0) && (dim2 == 1);
    if (isTransposeOuter) {
        // Outer dimension transposition only (there is a contiguous inner
        // dimension)
        size_t innerSize = 1;
        for (int i = 2; i < Dim; ++i) {
            innerSize *= in.getSize(i);
        }

        // The grid y dimension is more limited; we do a grid loop if necessary
        idx_t maxGridY = getCurrentDeviceProperties().maxGridSize[1];
        auto grid = dim3(in.getSize(1), std::min(in.getSize(0), maxGridY));

        int block = (innerSize < maxThreads) ? innerSize : maxThreads;

        transposeOuter<T><<<grid, block, 0, stream>>>(
                in.data(), out.data(), in.getSize(0), in.getSize(1), innerSize);
    } else {
        idx_t block = (totalSize < maxThreads) ? totalSize : maxThreads;

        auto inInfo = getTensorInfo<T, Dim>(in);
        auto outInfo = getTensorInfo<T, Dim>(out);

        std::swap(inInfo.sizes[dim1], inInfo.sizes[dim2]);
        std::swap(inInfo.strides[dim1], inInfo.strides[dim2]);

        auto grid = std::min(utils::divUp(totalSize, block), (idx_t)4096);

        transposeAny<T, Dim, -1>
                <<<grid, block, 0, stream>>>(inInfo, outInfo, totalSize);
    }

    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
