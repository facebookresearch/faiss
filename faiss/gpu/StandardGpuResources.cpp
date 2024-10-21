// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined USE_NVIDIA_RAFT
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <memory>
#endif

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <iostream>
#include <limits>
#include <sstream>

namespace faiss {
namespace gpu {

namespace {

// How many streams per device we allocate by default (for multi-streaming)
constexpr int kNumStreams = 2;

// Use 256 MiB of pinned memory for async CPU <-> GPU copies by default
constexpr size_t kDefaultPinnedMemoryAllocation = (size_t)256 * 1024 * 1024;

// Default temporary memory allocation for <= 4 GiB memory GPUs
constexpr size_t k4GiBTempMem = (size_t)512 * 1024 * 1024;

// Default temporary memory allocation for <= 8 GiB memory GPUs
constexpr size_t k8GiBTempMem = (size_t)1024 * 1024 * 1024;

// Maximum temporary memory allocation for all GPUs
constexpr size_t kMaxTempMem = (size_t)1536 * 1024 * 1024;

std::string allocsToString(const std::unordered_map<void*, AllocRequest>& map) {
    // Produce a sorted list of all outstanding allocations by type
    std::unordered_map<AllocType, std::pair<int, size_t>> stats;

    for (auto& entry : map) {
        auto& a = entry.second;

        auto it = stats.find(a.type);
        if (it != stats.end()) {
            stats[a.type].first++;
            stats[a.type].second += a.size;
        } else {
            stats[a.type] = std::make_pair(1, a.size);
        }
    }

    std::stringstream ss;
    for (auto& entry : stats) {
        ss << "Alloc type " << allocTypeToString(entry.first) << ": "
           << entry.second.first << " allocations, " << entry.second.second
           << " bytes\n";
    }

    return ss.str();
}

} // namespace

//
// StandardGpuResourcesImpl
//

StandardGpuResourcesImpl::StandardGpuResourcesImpl()
        :
#if defined USE_NVIDIA_RAFT
          mmr_(new rmm::mr::managed_memory_resource),
          pmr_(new rmm::mr::pinned_memory_resource),
#endif
          pinnedMemAlloc_(nullptr),
          pinnedMemAllocSize_(0),
          // let the adjustment function determine the memory size for us by
          // passing in a huge value that will then be adjusted
          tempMemSize_(getDefaultTempMemForGPU(
                  -1,
                  std::numeric_limits<size_t>::max())),
          pinnedMemSize_(kDefaultPinnedMemoryAllocation),
          allocLogging_(false) {
}

StandardGpuResourcesImpl::~StandardGpuResourcesImpl() {
    // The temporary memory allocator has allocated memory through us, so clean
    // that up before we finish fully de-initializing ourselves
    tempMemory_.clear();

    // Make sure all allocations have been freed
    bool allocError = false;

    for (auto& entry : allocs_) {
        auto& map = entry.second;

        if (!map.empty()) {
            std::cerr
                    << "StandardGpuResources destroyed with allocations outstanding:\n"
                    << "Device " << entry.first
                    << " outstanding allocations:\n";
            std::cerr << allocsToString(map);
            allocError = true;
        }
    }

    FAISS_ASSERT_MSG(
            !allocError, "GPU memory allocations not properly cleaned up");

    for (auto& entry : defaultStreams_) {
        DeviceScope scope(entry.first);

        // We created these streams, so are responsible for destroying them
        CUDA_VERIFY(cudaStreamDestroy(entry.second));
    }

    for (auto& entry : alternateStreams_) {
        DeviceScope scope(entry.first);

        for (auto stream : entry.second) {
            CUDA_VERIFY(cudaStreamDestroy(stream));
        }
    }

    for (auto& entry : asyncCopyStreams_) {
        DeviceScope scope(entry.first);

        CUDA_VERIFY(cudaStreamDestroy(entry.second));
    }

    for (auto& entry : blasHandles_) {
        DeviceScope scope(entry.first);

        auto blasStatus = cublasDestroy(entry.second);
        FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
    }

    if (pinnedMemAlloc_) {
#if defined USE_NVIDIA_RAFT
        pmr_->deallocate(pinnedMemAlloc_, pinnedMemAllocSize_);
#else
        auto err = cudaFreeHost(pinnedMemAlloc_);
        FAISS_ASSERT_FMT(
                err == cudaSuccess,
                "Failed to cudaFreeHost pointer %p (error %d %s)",
                pinnedMemAlloc_,
                (int)err,
                cudaGetErrorString(err));
#endif
    }
}

size_t StandardGpuResourcesImpl::getDefaultTempMemForGPU(
        int device,
        size_t requested) {
    auto totalMem = device != -1 ? getDeviceProperties(device).totalGlobalMem
                                 : std::numeric_limits<size_t>::max();

    if (totalMem <= (size_t)4 * 1024 * 1024 * 1024) {
        // If the GPU has <= 4 GiB of memory, reserve 512 MiB

        if (requested > k4GiBTempMem) {
            return k4GiBTempMem;
        }
    } else if (totalMem <= (size_t)8 * 1024 * 1024 * 1024) {
        // If the GPU has <= 8 GiB of memory, reserve 1 GiB

        if (requested > k8GiBTempMem) {
            return k8GiBTempMem;
        }
    } else {
        // Never use more than 1.5 GiB
        if (requested > kMaxTempMem) {
            return kMaxTempMem;
        }
    }

    // use whatever lower limit the user requested
    return requested;
}

void StandardGpuResourcesImpl::noTempMemory() {
    setTempMemory(0);
}

void StandardGpuResourcesImpl::setTempMemory(size_t size) {
    if (tempMemSize_ != size) {
        // adjust based on general limits
        tempMemSize_ = getDefaultTempMemForGPU(-1, size);

        // We need to re-initialize memory resources for all current devices
        // that have been initialized. This should be safe to do, even if we are
        // currently running work, because the cudaFree call that this implies
        // will force-synchronize all GPUs with the CPU
        for (auto& p : tempMemory_) {
            int device = p.first;
            // Free the existing memory first
            p.second.reset();

            // Allocate new
            p.second = std::make_unique<StackDeviceMemory>(
                    this,
                    p.first,
                    // adjust for this specific device
                    getDefaultTempMemForGPU(device, tempMemSize_));
        }
    }
}

void StandardGpuResourcesImpl::setPinnedMemory(size_t size) {
    // Should not call this after devices have been initialized
    FAISS_ASSERT(defaultStreams_.size() == 0);
    FAISS_ASSERT(!pinnedMemAlloc_);

    pinnedMemSize_ = size;
}

void StandardGpuResourcesImpl::setDefaultStream(
        int device,
        cudaStream_t stream) {
    if (isInitialized(device)) {
        // A new series of calls may not be ordered with what was the previous
        // stream, so if the stream being specified is different, then we need
        // to ensure ordering between the two (new stream waits on old).
        auto it = userDefaultStreams_.find(device);
        cudaStream_t prevStream = nullptr;

        if (it != userDefaultStreams_.end()) {
            prevStream = it->second;
        } else {
            FAISS_ASSERT(defaultStreams_.count(device));
            prevStream = defaultStreams_[device];
        }

        if (prevStream != stream) {
            streamWait({stream}, {prevStream});
        }
#if defined USE_NVIDIA_RAFT
        // delete the raft handle for this device, which will be initialized
        // with the updated stream during any subsequent calls to getRaftHandle
        auto it2 = raftHandles_.find(device);
        if (it2 != raftHandles_.end()) {
            raftHandles_.erase(it2);
        }
#endif
    }

    userDefaultStreams_[device] = stream;
}

void StandardGpuResourcesImpl::revertDefaultStream(int device) {
    if (isInitialized(device)) {
        auto it = userDefaultStreams_.find(device);

        if (it != userDefaultStreams_.end()) {
            // There was a user stream set that we need to synchronize against
            cudaStream_t prevStream = userDefaultStreams_[device];

            FAISS_ASSERT(defaultStreams_.count(device));
            cudaStream_t newStream = defaultStreams_[device];

            streamWait({newStream}, {prevStream});
        }
#if defined USE_NVIDIA_RAFT
        // delete the raft handle for this device, which will be initialized
        // with the updated stream during any subsequent calls to getRaftHandle
        auto it2 = raftHandles_.find(device);
        if (it2 != raftHandles_.end()) {
            raftHandles_.erase(it2);
        }
#endif
    }

    userDefaultStreams_.erase(device);
}

void StandardGpuResourcesImpl::setDefaultNullStreamAllDevices() {
    for (int dev = 0; dev < getNumDevices(); ++dev) {
        setDefaultStream(dev, nullptr);
    }
}

void StandardGpuResourcesImpl::setLogMemoryAllocations(bool enable) {
    allocLogging_ = enable;
}

bool StandardGpuResourcesImpl::isInitialized(int device) const {
    // Use default streams as a marker for whether or not a certain
    // device has been initialized
    return defaultStreams_.count(device) != 0;
}

void StandardGpuResourcesImpl::initializeForDevice(int device) {
    if (isInitialized(device)) {
        return;
    }

    FAISS_ASSERT(device < getNumDevices());
    DeviceScope scope(device);

    // If this is the first device that we're initializing, create our
    // pinned memory allocation
    if (defaultStreams_.empty() && pinnedMemSize_ > 0) {
#if defined USE_NVIDIA_RAFT
        // If this is the first device that we're initializing, create our
        // pinned memory allocation
        if (defaultStreams_.empty() && pinnedMemSize_ > 0) {
            try {
                pinnedMemAlloc_ = pmr_->allocate(pinnedMemSize_);
            } catch (const std::bad_alloc& rmm_ex) {
                FAISS_THROW_MSG("CUDA memory allocation error");
            }

            pinnedMemAllocSize_ = pinnedMemSize_;
        }
#else
        auto err = cudaHostAlloc(
                &pinnedMemAlloc_, pinnedMemSize_, cudaHostAllocDefault);

        FAISS_THROW_IF_NOT_FMT(
                err == cudaSuccess,
                "failed to cudaHostAlloc %zu bytes for CPU <-> GPU "
                "async copy buffer (error %d %s)",
                pinnedMemSize_,
                (int)err,
                cudaGetErrorString(err));

        pinnedMemAllocSize_ = pinnedMemSize_;
#endif
    }

    // Make sure that device properties for all devices are cached
    auto& prop = getDeviceProperties(device);

    // Also check to make sure we meet our minimum compute capability (3.0)
    FAISS_ASSERT_FMT(
            prop.major >= 3,
            "Device id %d with CC %d.%d not supported, "
            "need 3.0+ compute capability",
            device,
            prop.major,
            prop.minor);

#if USE_AMD_ROCM
    // Our code is pre-built with and expects warpSize == 32 or 64, validate
    // that
    FAISS_ASSERT_FMT(
            prop.warpSize == 32 || prop.warpSize == 64,
            "Device id %d does not have expected warpSize of 32 or 64",
            device);
#else
    // Our code is pre-built with and expects warpSize == 32, validate that
    FAISS_ASSERT_FMT(
            prop.warpSize == 32,
            "Device id %d does not have expected warpSize of 32",
            device);
#endif

    // Create streams
    cudaStream_t defaultStream = nullptr;
    CUDA_VERIFY(
            cudaStreamCreateWithFlags(&defaultStream, cudaStreamNonBlocking));

    defaultStreams_[device] = defaultStream;

#if defined USE_NVIDIA_RAFT
    raftHandles_.emplace(std::make_pair(device, defaultStream));
#endif

    cudaStream_t asyncCopyStream = 0;
    CUDA_VERIFY(
            cudaStreamCreateWithFlags(&asyncCopyStream, cudaStreamNonBlocking));

    asyncCopyStreams_[device] = asyncCopyStream;

    std::vector<cudaStream_t> deviceStreams;
    for (int j = 0; j < kNumStreams; ++j) {
        cudaStream_t stream = nullptr;
        CUDA_VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        deviceStreams.push_back(stream);
    }

    alternateStreams_[device] = std::move(deviceStreams);

    // Create cuBLAS handle
    cublasHandle_t blasHandle = nullptr;
    auto blasStatus = cublasCreate(&blasHandle);
    FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
    blasHandles_[device] = blasHandle;

    // For CUDA 10 on V100, enabling tensor core usage would enable automatic
    // rounding down of inputs to f16 (though accumulate in f32) which results
    // in unacceptable loss of precision in general. For CUDA 11 / A100, only
    // enable tensor core support if it doesn't result in a loss of precision.
#if CUDA_VERSION >= 11000
    cublasSetMathMode(
            blasHandle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
#endif

    FAISS_ASSERT(allocs_.count(device) == 0);
    allocs_[device] = std::unordered_map<void*, AllocRequest>();

    FAISS_ASSERT(tempMemory_.count(device) == 0);
    auto mem = std::make_unique<StackDeviceMemory>(
            this,
            device,
            // adjust for this specific device
            getDefaultTempMemForGPU(device, tempMemSize_));

    tempMemory_.emplace(device, std::move(mem));
}

cublasHandle_t StandardGpuResourcesImpl::getBlasHandle(int device) {
    initializeForDevice(device);
    return blasHandles_[device];
}

cudaStream_t StandardGpuResourcesImpl::getDefaultStream(int device) {
    initializeForDevice(device);

    auto it = userDefaultStreams_.find(device);
    if (it != userDefaultStreams_.end()) {
        // There is a user override stream set
        return it->second;
    }

    // Otherwise, our base default stream
    return defaultStreams_[device];
}

#if defined USE_NVIDIA_RAFT
raft::device_resources& StandardGpuResourcesImpl::getRaftHandle(int device) {
    initializeForDevice(device);

    auto it = raftHandles_.find(device);
    if (it == raftHandles_.end()) {
        // Make sure we are using the stream the user may have already assigned
        // to the current GpuResources
        raftHandles_.emplace(device, getDefaultStream(device));

        // Initialize cublas handle
        raftHandles_[device].get_cublas_handle();
    }

    // Otherwise, our base default handle
    return raftHandles_[device];
}
#endif

std::vector<cudaStream_t> StandardGpuResourcesImpl::getAlternateStreams(
        int device) {
    initializeForDevice(device);
    return alternateStreams_[device];
}

std::pair<void*, size_t> StandardGpuResourcesImpl::getPinnedMemory() {
    return std::make_pair(pinnedMemAlloc_, pinnedMemAllocSize_);
}

cudaStream_t StandardGpuResourcesImpl::getAsyncCopyStream(int device) {
    initializeForDevice(device);
    return asyncCopyStreams_[device];
}

void* StandardGpuResourcesImpl::allocMemory(const AllocRequest& req) {
    initializeForDevice(req.device);

    // We don't allocate a placeholder for zero-sized allocations
    if (req.size == 0) {
        return nullptr;
    }

    // cudaMalloc guarantees allocation alignment to 256 bytes; do the same here
    // for alignment purposes (to reduce memory transaction overhead etc)
    auto adjReq = req;
    adjReq.size = utils::roundUp(adjReq.size, (size_t)256);

    void* p = nullptr;

    if (adjReq.space == MemorySpace::Temporary) {
        auto& tempMem = tempMemory_[adjReq.device];

        if (adjReq.size > tempMem->getSizeAvailable()) {
            // We need to allocate this ourselves
            AllocRequest newReq = adjReq;
            newReq.space = MemorySpace::Device;
            newReq.type = AllocType::TemporaryMemoryOverflow;

            if (allocLogging_) {
                std::cout
                        << "StandardGpuResources: alloc fail "
                        << adjReq.toString()
                        << " (no temp space); retrying as MemorySpace::Device\n";
            }

            return allocMemory(newReq);
        }

        // Otherwise, we can handle this locally
        p = tempMemory_[adjReq.device]->allocMemory(adjReq.stream, adjReq.size);
    } else if (adjReq.space == MemorySpace::Device) {
#if defined USE_NVIDIA_RAFT
        try {
            rmm::mr::device_memory_resource* current_mr =
                    rmm::mr::get_per_device_resource(
                            rmm::cuda_device_id{adjReq.device});
            p = current_mr->allocate_async(adjReq.size, adjReq.stream);
            adjReq.mr = current_mr;
        } catch (const std::bad_alloc& rmm_ex) {
            FAISS_THROW_MSG("CUDA memory allocation error");
        }
#else
        auto err = cudaMalloc(&p, adjReq.size);

        // Throw if we fail to allocate
        if (err != cudaSuccess) {
            // FIXME: as of CUDA 11, a memory allocation error appears to be
            // presented via cudaGetLastError as well, and needs to be
            // cleared. Just call the function to clear it
            cudaGetLastError();

            std::stringstream ss;
            ss << "StandardGpuResources: alloc fail " << adjReq.toString()
               << " (cudaMalloc error " << cudaGetErrorString(err) << " ["
               << (int)err << "])\n";
            auto str = ss.str();

            if (allocLogging_) {
                std::cout << str;
            }

            FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "%s", str.c_str());
        }
#endif
    } else if (adjReq.space == MemorySpace::Unified) {
#if defined USE_NVIDIA_RAFT
        try {
            // for now, use our own managed MR to do Unified Memory allocations.
            // TODO: change this to use the current device resource once RMM has
            // a way to retrieve a "guaranteed" managed memory resource for a
            // device.
            p = mmr_->allocate_async(adjReq.size, adjReq.stream);
            adjReq.mr = mmr_.get();
        } catch (const std::bad_alloc& rmm_ex) {
            FAISS_THROW_MSG("CUDA memory allocation error");
        }
#else
        auto err = cudaMallocManaged(&p, adjReq.size);

        if (err != cudaSuccess) {
            // FIXME: as of CUDA 11, a memory allocation error appears to be
            // presented via cudaGetLastError as well, and needs to be cleared.
            // Just call the function to clear it
            cudaGetLastError();

            std::stringstream ss;
            ss << "StandardGpuResources: alloc fail " << adjReq.toString()
               << " failed (cudaMallocManaged error " << cudaGetErrorString(err)
               << " [" << (int)err << "])\n";
            auto str = ss.str();

            if (allocLogging_) {
                std::cout << str;
            }

            FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "%s", str.c_str());
        }
#endif
    } else {
        FAISS_ASSERT_FMT(false, "unknown MemorySpace %d", (int)adjReq.space);
    }

    if (allocLogging_) {
        std::cout << "StandardGpuResources: alloc ok " << adjReq.toString()
                  << " ptr 0x" << p << "\n";
    }

    allocs_[adjReq.device][p] = adjReq;

    return p;
}

void StandardGpuResourcesImpl::deallocMemory(int device, void* p) {
    FAISS_ASSERT(isInitialized(device));

    if (!p) {
        return;
    }

    auto& a = allocs_[device];
    auto it = a.find(p);
    FAISS_ASSERT(it != a.end());

    auto& req = it->second;

    if (allocLogging_) {
        std::cout << "StandardGpuResources: dealloc " << req.toString() << "\n";
    }

    if (req.space == MemorySpace::Temporary) {
        tempMemory_[device]->deallocMemory(device, req.stream, req.size, p);
    } else if (
            req.space == MemorySpace::Device ||
            req.space == MemorySpace::Unified) {
#if defined USE_NVIDIA_RAFT
        req.mr->deallocate_async(p, req.size, req.stream);
#else
        auto err = cudaFree(p);
        FAISS_ASSERT_FMT(
                err == cudaSuccess,
                "Failed to cudaFree pointer %p (error %d %s)",
                p,
                (int)err,
                cudaGetErrorString(err));
#endif
    } else {
        FAISS_ASSERT_FMT(false, "unknown MemorySpace %d", (int)req.space);
    }

    a.erase(it);
}

size_t StandardGpuResourcesImpl::getTempMemoryAvailable(int device) const {
    FAISS_ASSERT(isInitialized(device));

    auto it = tempMemory_.find(device);
    FAISS_ASSERT(it != tempMemory_.end());

    return it->second->getSizeAvailable();
}

std::map<int, std::map<std::string, std::pair<int, size_t>>>
StandardGpuResourcesImpl::getMemoryInfo() const {
    using AT = std::map<std::string, std::pair<int, size_t>>;

    std::map<int, AT> out;

    for (auto& entry : allocs_) {
        AT outDevice;

        for (auto& a : entry.second) {
            auto& v = outDevice[allocTypeToString(a.second.type)];
            v.first++;
            v.second += a.second.size;
        }

        out[entry.first] = std::move(outDevice);
    }

    return out;
}

//
// StandardGpuResources
//

StandardGpuResources::StandardGpuResources()
        : res_(new StandardGpuResourcesImpl) {}

StandardGpuResources::~StandardGpuResources() = default;

std::shared_ptr<GpuResources> StandardGpuResources::getResources() {
    return res_;
}

void StandardGpuResources::noTempMemory() {
    res_->noTempMemory();
}

void StandardGpuResources::setTempMemory(size_t size) {
    res_->setTempMemory(size);
}

void StandardGpuResources::setPinnedMemory(size_t size) {
    res_->setPinnedMemory(size);
}

void StandardGpuResources::setDefaultStream(int device, cudaStream_t stream) {
    res_->setDefaultStream(device, stream);
}

void StandardGpuResources::revertDefaultStream(int device) {
    res_->revertDefaultStream(device);
}

void StandardGpuResources::setDefaultNullStreamAllDevices() {
    res_->setDefaultNullStreamAllDevices();
}

std::map<int, std::map<std::string, std::pair<int, size_t>>>
StandardGpuResources::getMemoryInfo() const {
    return res_->getMemoryInfo();
}

cudaStream_t StandardGpuResources::getDefaultStream(int device) {
    return res_->getDefaultStream(device);
}

#if defined USE_NVIDIA_RAFT
raft::device_resources& StandardGpuResources::getRaftHandle(int device) {
    return res_->getRaftHandle(device);
}
#endif

size_t StandardGpuResources::getTempMemoryAvailable(int device) const {
    return res_->getTempMemoryAvailable(device);
}

void StandardGpuResources::syncDefaultStreamCurrentDevice() {
    res_->syncDefaultStreamCurrentDevice();
}

void StandardGpuResources::setLogMemoryAllocations(bool enable) {
    res_->setLogMemoryAllocations(enable);
}

} // namespace gpu
} // namespace faiss
