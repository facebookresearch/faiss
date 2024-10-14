/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Timer.h>
#include <faiss/impl/FaissAssert.h>
#include <chrono>

namespace faiss {
namespace gpu {

KernelTimer::KernelTimer(cudaStream_t stream)
        : startEvent_(nullptr),
          stopEvent_(nullptr),
          stream_(stream),
          valid_(true) {
    CUDA_VERIFY(cudaEventCreate(&startEvent_));
    CUDA_VERIFY(cudaEventCreate(&stopEvent_));

    CUDA_VERIFY(cudaEventRecord(startEvent_, stream_));
}

KernelTimer::~KernelTimer() {
    CUDA_VERIFY(cudaEventDestroy(startEvent_));
    CUDA_VERIFY(cudaEventDestroy(stopEvent_));
}

float KernelTimer::elapsedMilliseconds() {
    FAISS_ASSERT(valid_);

    CUDA_VERIFY(cudaEventRecord(stopEvent_, stream_));
    CUDA_VERIFY(cudaEventSynchronize(stopEvent_));

    auto time = 0.0f;
    CUDA_VERIFY(cudaEventElapsedTime(&time, startEvent_, stopEvent_));
    valid_ = false;

    return time;
}

CpuTimer::CpuTimer() {
    start_ = std::chrono::steady_clock::now();
}

float CpuTimer::elapsedMilliseconds() {
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start_;

    return duration.count();
}

} // namespace gpu
} // namespace faiss
