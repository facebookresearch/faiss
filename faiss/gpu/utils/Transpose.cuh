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

template <typename T, typename IndexT>
struct TensorInfo {
    static constexpr int kMaxDims = 8;

    T* data;
    IndexT sizes[kMaxDims];
    IndexT strides[kMaxDims];
    int dims;
};

template <typename T, typename IndexT, int Dim>
struct TensorInfoOffset {
    __device__ inline static unsigned int get(
            const TensorInfo<T, IndexT>& info,
            IndexT linearId) {
        IndexT offset = 0;

#pragma unroll
        for (int i = Dim - 1; i >= 0; --i) {
            IndexT curDimIndex = linearId % info.sizes[i];
            IndexT curDimOffset = curDimIndex * info.strides[i];

            offset += curDimOffset;

            if (i > 0) {
                linearId /= info.sizes[i];
            }
        }

        return offset;
    }
};

template <typename T, typename IndexT>
struct TensorInfoOffset<T, IndexT, -1> {
    __device__ inline static unsigned int get(
            const TensorInfo<T, IndexT>& info,
            IndexT linearId) {
        return linearId;
    }
};

template <typename T, typename IndexT, int Dim>
TensorInfo<T, IndexT> getTensorInfo(const Tensor<T, Dim, true>& t) {
    TensorInfo<T, IndexT> info;

    for (int i = 0; i < Dim; ++i) {
        info.sizes[i] = (IndexT)t.getSize(i);
        info.strides[i] = (IndexT)t.getStride(i);
    }

    info.data = t.data();
    info.dims = Dim;

    return info;
}

template <typename T, typename IndexT, int DimInput, int DimOutput>
__global__ void transposeAny(
        TensorInfo<T, IndexT> input,
        TensorInfo<T, IndexT> output,
        IndexT totalSize) {
    for (IndexT i = blockIdx.x * blockDim.x + threadIdx.x; i < totalSize;
         i += gridDim.x * blockDim.x) {
        auto inputOffset = TensorInfoOffset<T, IndexT, DimInput>::get(input, i);
        auto outputOffset =
                TensorInfoOffset<T, IndexT, DimOutput>::get(output, i);

#if __CUDA_ARCH__ >= 350
        output.data[outputOffset] = __ldg(&input.data[inputOffset]);
#else
        output.data[outputOffset] = input.data[inputOffset];
#endif
    }
}

// Transpose contiguous t1 t2 i1 -> t2 t1 i1
template <typename T, typename IndexT>
__global__ void transposeOuter(
        const T* in,
        T* out,
        IndexT t1,
        IndexT t2,
        IndexT i1) {
    IndexT gt2 = blockIdx.x;
    for (IndexT gt1 = blockIdx.y; gt1 < t1; gt1 += gridDim.y) {
        auto curIn = in + i1 * (gt1 * t2 + gt2);
        auto curOut = out + i1 * (gt2 * t1 + gt1);

        for (IndexT i = threadIdx.x; i < i1; i += blockDim.x) {
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
    static_assert(
            Dim <= TensorInfo<T, unsigned int>::kMaxDims,
            "too many dimensions");

    FAISS_ASSERT(dim1 != dim2);
    FAISS_ASSERT(dim1 < Dim && dim2 < Dim);

    // Rearrange dim1 and dim2 in increasing order in order to see if this is an
    // outer dimension transposition (below)
    if (dim1 > dim2) {
        std::swap(dim1, dim2);
    }

    int outSize[Dim];

    for (int i = 0; i < Dim; ++i) {
        outSize[i] = in.getSize(i);
    }

    std::swap(outSize[dim1], outSize[dim2]);

    for (int i = 0; i < Dim; ++i) {
        FAISS_ASSERT(out.getSize(i) == outSize[i]);
    }

    auto maxThreads = getMaxThreadsCurrentDevice();
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
        int maxGridY = getCurrentDeviceProperties().maxGridSize[1];
        auto grid = dim3(in.getSize(1), std::min(in.getSize(0), maxGridY));

        int block = (innerSize < maxThreads) ? innerSize : maxThreads;

        if (totalSize <= (size_t)std::numeric_limits<int>::max()) {
            transposeOuter<T, int32_t><<<grid, block, 0, stream>>>(
                    in.data(),
                    out.data(),
                    in.getSize(0),
                    in.getSize(1),
                    innerSize);
        } else {
            transposeOuter<T, int64_t><<<grid, block, 0, stream>>>(
                    in.data(),
                    out.data(),
                    in.getSize(0),
                    in.getSize(1),
                    innerSize);
        }
    } else {
        int block = (totalSize < maxThreads) ? totalSize : maxThreads;

        // Non-outer transposition
        if (totalSize <= (size_t)std::numeric_limits<int>::max()) {
            // General transposition
            // div/mod seems faster with unsigned types
            auto inInfo = getTensorInfo<T, uint32_t, Dim>(in);
            auto outInfo = getTensorInfo<T, uint32_t, Dim>(out);

            std::swap(inInfo.sizes[dim1], inInfo.sizes[dim2]);
            std::swap(inInfo.strides[dim1], inInfo.strides[dim2]);

            auto grid = std::min(utils::divUp(totalSize, block), (size_t)4096);

            transposeAny<T, uint32_t, Dim, -1>
                    <<<grid, block, 0, stream>>>(inInfo, outInfo, totalSize);
        } else {
            auto inInfo = getTensorInfo<T, uint64_t, Dim>(in);
            auto outInfo = getTensorInfo<T, uint64_t, Dim>(out);

            std::swap(inInfo.sizes[dim1], inInfo.sizes[dim2]);
            std::swap(inInfo.strides[dim1], inInfo.strides[dim2]);

            auto grid = std::min(utils::divUp(totalSize, block), (size_t)4096);

            transposeAny<T, uint64_t, Dim, -1>
                    <<<grid, block, 0, stream>>>(inInfo, outInfo, totalSize);
        }
    }

    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
