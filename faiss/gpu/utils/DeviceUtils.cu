/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_profiler_api.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <cstring>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <mutex>
#include <unordered_map>
#include <vector>

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
    if (cudaErrorNoDevice == err || cudaErrorInsufficientDriver == err) {
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

#if USE_AMD_ROCM
    if (att.type != hipMemoryTypeHost &&
        att.type != hipMemoryTypeUnregistered) {
        return att.device;
    } else {
        return -1;
    }
#else
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
#endif
}

#ifdef USE_AMD_ROCM
namespace {

// ROCm copies pageable host memory of more than 64 KiB by pinning the user's
// pages for the copy. On MI350X that path takes GPU memory access faults when
// the process unmaps other host memory meanwhile, so copy through our own
// pinned buffers instead, as CUDA does internally.
constexpr size_t kMinStagedCopyBytes = 64 * 1024;
constexpr size_t kStagingChunkBytes = 16 * 1024 * 1024;
// Double buffering: the host copy of one chunk overlaps the device copy of the
// other
constexpr int kStagingBuffers = 2;

struct PinnedStaging {
    PinnedStaging() {
        for (int i = 0; i < kStagingBuffers; ++i) {
            auto err = hipHostMalloc(
                    &buf[i], kStagingChunkBytes, hipHostMallocPortable);
            if (err == hipSuccess) {
                err = hipEventCreateWithFlags(&done[i], hipEventDisableTiming);
            }
            if (err != hipSuccess) {
                // A sticky error would fail a later, unrelated launch check
                (void)hipGetLastError();
                releaseAfterFailure();
                FAISS_THROW_FMT(
                        "failed to allocate a %zu byte pinned staging buffer "
                        "for a pageable host copy (error %d %s)",
                        kStagingChunkBytes,
                        (int)err,
                        hipGetErrorString(err));
            }
        }
    }

    void releaseAfterFailure() {
        for (int i = 0; i < kStagingBuffers; ++i) {
            if (done[i]) {
                (void)hipEventDestroy(done[i]);
            }
            if (buf[i]) {
                (void)hipHostFree(buf[i]);
            }
        }
    }

    void wait(int b) {
        if (pending[b]) {
            CUDA_VERIFY(hipEventSynchronize(done[b]));
            pending[b] = false;
        }
    }

    void waitAll() {
        for (int b = 0; b < kStagingBuffers; ++b) {
            wait(b);
        }
    }

    void record(int b, hipStream_t stream) {
        CUDA_VERIFY(hipEventRecord(done[b], stream));
        pending[b] = true;
    }

    void* buf[kStagingBuffers] = {};
    hipEvent_t done[kStagingBuffers] = {};
    bool pending[kStagingBuffers] = {};
};

// Free staging buffers per device (events belong to a device). The pool is
// never destroyed: freeing pinned memory from a static or thread_local
// destructor crashes once the HIP runtime has shut down at exit.
struct StagingPool {
    std::mutex mutex;
    std::unordered_map<int, std::vector<PinnedStaging*>> free;
};

StagingPool& stagingPool() {
    static auto* pool = new StagingPool();
    return *pool;
}

// Takes a staging buffer set for the current device for one copy
class StagingLease {
   public:
    StagingLease() : device_(getCurrentDevice()) {
        auto& pool = stagingPool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        auto& list = pool.free[device_];
        if (list.empty()) {
            staging_ = new PinnedStaging();
        } else {
            staging_ = list.back();
            list.pop_back();
        }
    }

    ~StagingLease() {
        // An event must not be waited on after its stream may be destroyed:
        // hipEventSynchronize dereferences the stream of the last record
        staging_->waitAll();
        auto& pool = stagingPool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.free[device_].push_back(staging_);
    }

    StagingLease(const StagingLease&) = delete;
    StagingLease& operator=(const StagingLease&) = delete;

    PinnedStaging& operator*() const {
        return *staging_;
    }

   private:
    int device_;
    PinnedStaging* staging_;
};

bool isPageableHostMemory(const void* p) {
    hipPointerAttribute_t att;
    if (hipPointerGetAttributes(&att, p) != hipSuccess) {
        (void)hipGetLastError();
        return true;
    }
    return att.type == hipMemoryTypeUnregistered;
}

void stagedHostToDevice(
        char* dst,
        const char* src,
        size_t bytes,
        hipStream_t stream) {
    StagingLease lease;
    auto& s = *lease;
    for (size_t off = 0, i = 0; off < bytes; off += kStagingChunkBytes, ++i) {
        const int b = i % kStagingBuffers;
        const size_t n = std::min(kStagingChunkBytes, bytes - off);
        s.wait(b);
        std::memcpy(s.buf[b], src + off, n);
        CUDA_VERIFY(hipMemcpyAsync(
                dst + off, s.buf[b], n, hipMemcpyHostToDevice, stream));
        s.record(b, stream);
    }
}

void stagedDeviceToHost(
        char* dst,
        const char* src,
        size_t bytes,
        hipStream_t stream) {
    StagingLease lease;
    auto& s = *lease;
    // The host copy of chunk i - 1 overlaps the device copy of chunk i
    auto drain = [&](int b, size_t off, size_t n) {
        s.wait(b);
        std::memcpy(dst + off, s.buf[b], n);
    };
    int prevB = -1;
    size_t prevOff = 0;
    size_t prevN = 0;
    for (size_t off = 0, i = 0; off < bytes; off += kStagingChunkBytes, ++i) {
        const int b = i % kStagingBuffers;
        const size_t n = std::min(kStagingChunkBytes, bytes - off);
        s.wait(b);
        CUDA_VERIFY(hipMemcpyAsync(
                s.buf[b], src + off, n, hipMemcpyDeviceToHost, stream));
        s.record(b, stream);
        if (prevB >= 0) {
            drain(prevB, prevOff, prevN);
        }
        prevB = b;
        prevOff = off;
        prevN = n;
    }
    if (prevB >= 0) {
        drain(prevB, prevOff, prevN);
    }
}

} // namespace
#endif

void memcpyHostDeviceAsync(
        void* dst,
        const void* src,
        size_t bytes,
        cudaMemcpyKind kind,
        cudaStream_t stream) {
#ifdef USE_AMD_ROCM
    if (bytes > kMinStagedCopyBytes) {
        if (kind == hipMemcpyHostToDevice && isPageableHostMemory(src)) {
            stagedHostToDevice(
                    static_cast<char*>(dst),
                    static_cast<const char*>(src),
                    bytes,
                    stream);
            return;
        }
        if (kind == hipMemcpyDeviceToHost && isPageableHostMemory(dst)) {
            stagedDeviceToHost(
                    static_cast<char*>(dst),
                    static_cast<const char*>(src),
                    bytes,
                    stream);
            return;
        }
    }
#endif
    CUDA_VERIFY(cudaMemcpyAsync(dst, src, bytes, kind, stream));
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

int getWarpSize(int device) {
    const auto& prop = getDeviceProperties(device);
    return prop.warpSize;
}

int getWarpSizeCurrentDevice() {
    return getWarpSize(getCurrentDevice());
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
