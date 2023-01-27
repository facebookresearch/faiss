/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>

#define IVF_INTERLEAVED_IMPL(THREADS, WARP_Q, THREAD_Q) \
                                                        \
    void ivfInterleavedScanImpl_##WARP_Q##_(            \
            Tensor<float, 2, true>& queries,            \
            Tensor<idx_t, 2, true>& listIds,            \
            DeviceVector<void*>& listData,              \
            DeviceVector<void*>& listIndices,           \
            IndicesOptions indicesOptions,              \
            DeviceVector<int>& listLengths,             \
            int k,                                      \
            faiss::MetricType metric,                   \
            bool useResidual,                           \
            Tensor<float, 3, true>& residualBase,       \
            GpuScalarQuantizer* scalarQ,                \
            Tensor<float, 2, true>& outDistances,       \
            Tensor<idx_t, 2, true>& outIndices,         \
            GpuResources* res) {                        \
        FAISS_ASSERT(k <= WARP_Q);                      \
                                                        \
        IVFINT_METRICS(THREADS, WARP_Q, THREAD_Q);      \
                                                        \
        CUDA_TEST_ERROR();                              \
    }

#define IVF_INTERLEAVED_DECL(WARP_Q)              \
                                                  \
    void ivfInterleavedScanImpl_##WARP_Q##_(      \
            Tensor<float, 2, true>& queries,      \
            Tensor<idx_t, 2, true>& listIds,      \
            DeviceVector<void*>& listData,        \
            DeviceVector<void*>& listIndices,     \
            IndicesOptions indicesOptions,        \
            DeviceVector<int>& listLengths,       \
            int k,                                \
            faiss::MetricType metric,             \
            bool useResidual,                     \
            Tensor<float, 3, true>& residualBase, \
            GpuScalarQuantizer* scalarQ,          \
            Tensor<float, 2, true>& outDistances, \
            Tensor<idx_t, 2, true>& outIndices,   \
            GpuResources* res)

#define IVF_INTERLEAVED_CALL(WARP_Q)    \
    ivfInterleavedScanImpl_##WARP_Q##_( \
            queries,                    \
            listIds,                    \
            listData,                   \
            listIndices,                \
            indicesOptions,             \
            listLengths,                \
            k,                          \
            metric,                     \
            useResidual,                \
            residualBase,               \
            scalarQ,                    \
            outDistances,               \
            outIndices,                 \
            res)

namespace faiss {
namespace gpu {

IVF_INTERLEAVED_DECL(1);
IVF_INTERLEAVED_DECL(32);
IVF_INTERLEAVED_DECL(64);
IVF_INTERLEAVED_DECL(128);
IVF_INTERLEAVED_DECL(256);
IVF_INTERLEAVED_DECL(512);
IVF_INTERLEAVED_DECL(1024);

#if GPU_MAX_SELECTION_K >= 2048
IVF_INTERLEAVED_DECL(2048);
#endif

} // namespace gpu
} // namespace faiss
