/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_profiler_api.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <mutex>
#include <unordered_map>

namespace faiss {
namespace gpu {

int getCurrentDevice() {
    int dev = -1;
    CUDA_VERIFY(cudaGetDevice(&dev));
    FAISS_ASSERT(dev != -1);

    return dev;
}

void setCurrentDevice(int device) {
    CUDA_VERIFY(cudaSetDevice(device));
}

int getNumDevices() {
    int numDev = -1;
    cudaError_t err = cudaGetDeviceCount(&numDev);
    if (cudaErrorNoDevice == err) {
        numDev = 0;
    } else {
        CUDA_VERIFY(err);
    }
    FAISS_ASSERT(numDev != -1);

    return numDev;
}

void profilerStart() {
    CUDA_VERIFY(cudaProfilerStart());
}

void profilerStop() {
    CUDA_VERIFY(cudaProfilerStop());
}

void synchronizeAllDevices() {
    for (int i = 0; i < getNumDevices(); ++i) {
        DeviceScope scope(i);

        CUDA_VERIFY(cudaDeviceSynchronize());
    }
}

const cudaDeviceProp& getDeviceProperties(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaDeviceProp> properties;

    std::lock_guard<std::mutex> guard(mutex);

    auto it = properties.find(device);
    if (it == properties.end()) {
        cudaDeviceProp prop;
        CUDA_VERIFY(cudaGetDeviceProperties(&prop, device));

        properties[device] = prop;
        it = properties.find(device);
    }

    return it->second;
}

const cudaDeviceProp& getCurrentDeviceProperties() {
    return getDeviceProperties(getCurrentDevice());
}

int getMaxThreads(int device) {
    return getDeviceProperties(device).maxThreadsPerBlock;
}

int getMaxThreadsCurrentDevice() {
    return getMaxThreads(getCurrentDevice());
}

dim3 getMaxGrid(int device) {
    auto& prop = getDeviceProperties(device);

    return dim3(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

dim3 getMaxGridCurrentDevice() {
    return getMaxGrid(getCurrentDevice());
}

size_t getMaxSharedMemPerBlock(int device) {
    return getDeviceProperties(device).sharedMemPerBlock;
}

size_t getMaxSharedMemPerBlockCurrentDevice() {
    return getMaxSharedMemPerBlock(getCurrentDevice());
}

int getDeviceForAddress(const void* p) {
    if (!p) {
        return -1;
    }

    cudaPointerAttributes att;
    cudaError_t err = cudaPointerGetAttributes(&att, p);
    FAISS_ASSERT_FMT(
            err == cudaSuccess || err == cudaErrorInvalidValue,
            "unknown error %d",
            (int)err);

    if (err == cudaErrorInvalidValue) {
        // Make sure the current thread error status has been reset
        err = cudaGetLastError();
        FAISS_ASSERT_FMT(
                err == cudaErrorInvalidValue, "unknown error %d", (int)err);
        return -1;
    }

    // memoryType is deprecated for CUDA 10.0+
#if CUDA_VERSION < 10000
    if (att.memoryType == cudaMemoryTypeHost) {
        return -1;
    } else {
        return att.device;
    }
#else
    // FIXME: what to use for managed memory?
    if (att.type == cudaMemoryTypeDevice) {
        return att.device;
    } else {
        return -1;
    }
#endif
}

bool getFullUnifiedMemSupport(int device) {
    const auto& prop = getDeviceProperties(device);
    return (prop.major >= 6);
}

bool getFullUnifiedMemSupportCurrentDevice() {
    return getFullUnifiedMemSupport(getCurrentDevice());
}

bool getTensorCoreSupport(int device) {
    const auto& prop = getDeviceProperties(device);
    return (prop.major >= 7);
}

bool getTensorCoreSupportCurrentDevice() {
    return getTensorCoreSupport(getCurrentDevice());
}

size_t getFreeMemory(int device) {
    DeviceScope scope(device);

    size_t free = 0;
    size_t total = 0;

    CUDA_VERIFY(cudaMemGetInfo(&free, &total));

    return free;
}

size_t getFreeMemoryCurrentDevice() {
    size_t free = 0;
    size_t total = 0;

    CUDA_VERIFY(cudaMemGetInfo(&free, &total));

    return free;
}

DeviceScope::DeviceScope(int device) {
    if (device >= 0) {
        int curDevice = getCurrentDevice();

        if (curDevice != device) {
            prevDevice_ = curDevice;
            setCurrentDevice(device);
            return;
        }
    }

    // Otherwise, we keep the current device
    prevDevice_ = -1;
}

DeviceScope::~DeviceScope() {
    if (prevDevice_ != -1) {
        setCurrentDevice(prevDevice_);
    }
}

CublasHandleScope::CublasHandleScope() {
    auto blasStatus = cublasCreate(&blasHandle_);
    FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
}

CublasHandleScope::~CublasHandleScope() {
    auto blasStatus = cublasDestroy(blasHandle_);
    FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
}

CudaEvent::CudaEvent(cudaStream_t stream, bool timer) : event_(0) {
    CUDA_VERIFY(cudaEventCreateWithFlags(
            &event_, timer ? cudaEventDefault : cudaEventDisableTiming));
    CUDA_VERIFY(cudaEventRecord(event_, stream));
}

CudaEvent::CudaEvent(CudaEvent&& event) noexcept
        : event_(std::move(event.event_)) {
    event.event_ = 0;
}

CudaEvent::~CudaEvent() {
    if (event_) {
        CUDA_VERIFY(cudaEventDestroy(event_));
    }
}

CudaEvent& CudaEvent::operator=(CudaEvent&& event) noexcept {
    event_ = std::move(event.event_);
    event.event_ = 0;

    return *this;
}

void CudaEvent::streamWaitOnEvent(cudaStream_t stream) {
    CUDA_VERIFY(cudaStreamWaitEvent(stream, event_, 0));
}

void CudaEvent::cpuWaitOnEvent() {
    CUDA_VERIFY(cudaEventSynchronize(event_));
}

} // namespace gpu
} // namespace faiss
