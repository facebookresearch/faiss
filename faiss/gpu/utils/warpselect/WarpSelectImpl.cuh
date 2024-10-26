/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/WarpSelectKernel.cuh>

#define WARP_SELECT_DECL(TYPE, DIR, WARP_Q)                 \
    extern void runWarpSelect_##TYPE##_##DIR##_##WARP_Q##_( \
            Tensor<TYPE, 2, true>& in,                      \
            Tensor<TYPE, 2, true>& outK,                    \
            Tensor<idx_t, 2, true>& outV,                   \
            bool dir,                                       \
            int k,                                          \
            cudaStream_t stream)

#define WARP_SELECT_IMPL(TYPE, DIR, WARP_Q, THREAD_Q)                          \
    void runWarpSelect_##TYPE##_##DIR##_##WARP_Q##_(                           \
            Tensor<TYPE, 2, true>& in,                                         \
            Tensor<TYPE, 2, true>& outK,                                       \
            Tensor<idx_t, 2, true>& outV,                                      \
            bool dir,                                                          \
            int k,                                                             \
            cudaStream_t stream) {                                             \
        int warpSize = getWarpSizeCurrentDevice();                             \
        constexpr int kWarpSelectNumThreads = 128;                             \
        auto grid = dim3(utils::divUp(                                         \
                in.getSize(0), (kWarpSelectNumThreads / warpSize)));           \
        auto block = dim3(kWarpSelectNumThreads);                              \
                                                                               \
        FAISS_ASSERT(k <= WARP_Q);                                             \
        FAISS_ASSERT(dir == DIR);                                              \
                                                                               \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();    \
        auto vInit = -1;                                                       \
                                                                               \
        warpSelect<TYPE, idx_t, DIR, WARP_Q, THREAD_Q, kWarpSelectNumThreads>  \
                <<<grid, block, 0, stream>>>(in, outK, outV, kInit, vInit, k); \
        CUDA_TEST_ERROR();                                                     \
    }

#define WARP_SELECT_CALL(TYPE, DIR, WARP_Q) \
    runWarpSelect_##TYPE##_##DIR##_##WARP_Q##_(in, outK, outV, dir, k, stream)
