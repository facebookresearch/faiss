/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
This code contains unnecessary code duplication. These could be deleted
once the relevant changes would be made on the FAISS side. Indeed most of
the logic in the below code is similar to FAISS's standard implementation
and should thus be inherited instead of duplicated. This FAISS's issue
once solved should allow the removal of the unnecessary duplicates
in this file : https://github.com/facebookresearch/faiss/issues/2097
*/

#pragma once

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StackDeviceMemory.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <raft/core/handle.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

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

std::string allocsToString(const std::unordered_map<void*, AllocRequest>& map)
{
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
        ss << "Alloc type " << allocTypeToString(entry.first) << ": " << entry.second.first
           << " allocations, " << entry.second.second << " bytes\n";
    }

    return ss.str();
}

}  // namespace

/// RMM implementation of the GpuResources object that provides for a
/// temporary memory manager
class RmmGpuResourcesImpl : public GpuResources {
   public:
    RmmGpuResourcesImpl()
            : pinnedMemAlloc_(nullptr),
              pinnedMemAllocSize_(0),
            // let the adjustment function determine the memory size for us by passing
            // in a huge value that will then be adjusted
              tempMemSize_(getDefaultTempMemForGPU(-1, std::numeric_limits<size_t>::max())),
              pinnedMemSize_(kDefaultPinnedMemoryAllocation),
              allocLogging_(false),
              cmr(new rmm::mr::cuda_memory_resource),
              mmr(new rmm::mr::managed_memory_resource),
              pmr(new rmm::mr::pinned_memory_resource){};

    ~RmmGpuResourcesImpl()
    {
        // The temporary memory allocator has allocated memory through us, so clean
        // that up before we finish fully de-initializing ourselves
        tempMemory_.clear();

        // Make sure all allocations have been freed
        bool allocError = false;

        for (auto& entry : allocs_) {
            auto& map = entry.second;

            if (!map.empty()) {
                std::cerr << "RmmGpuResources destroyed with allocations outstanding:\n"
                          << "Device " << entry.first << " outstanding allocations:\n";
                std::cerr << allocsToString(map);
                allocError = true;
            }
        }

        FAISS_ASSERT_MSG(!allocError, "GPU memory allocations not properly cleaned up");

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

        if (pinnedMemAlloc_) { pmr->deallocate(pinnedMemAlloc_, pinnedMemAllocSize_); }
    };

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call cudaMalloc / cudaFree at the point of use
    void noTempMemory() { setTempMemory(0); };

    /// Specify that we wish to use a certain fixed size of memory on
    /// all devices as temporary memory. This is the upper bound for the GPU
    /// memory that we will reserve. We will never go above 1.5 GiB on any GPU;
    /// smaller GPUs (with <= 4 GiB or <= 8 GiB) will use less memory than that.
    /// To avoid any temporary memory allocation, pass 0.
    void setTempMemory(size_t size)
    {
        if (tempMemSize_ != size) {
            // adjust based on general limits
            tempMemSize_ = getDefaultTempMemForGPU(-1, size);

            // We need to re-initialize memory resources for all current devices that
            // have been initialized.
            // This should be safe to do, even if we are currently running work, because
            // the cudaFree call that this implies will force-synchronize all GPUs with
            // the CPU
            for (auto& p : tempMemory_) {
                int device = p.first;
                // Free the existing memory first
                p.second.reset();

                // Allocate new
                p.second = std::unique_ptr<StackDeviceMemory>(
                        new StackDeviceMemory(this,
                                              p.first,
                                // adjust for this specific device
                                              getDefaultTempMemForGPU(device, tempMemSize_)));
            }
        }
    };

    /// Set amount of pinned memory to allocate, for async GPU <-> CPU
    /// transfers
    void setPinnedMemory(size_t size)
    {
        // Should not call this after devices have been initialized
        FAISS_ASSERT(defaultStreams_.size() == 0);
        FAISS_ASSERT(!pinnedMemAlloc_);

        pinnedMemSize_ = size;
    };

    /// Called to change the stream for work ordering. We do not own `stream`;
    /// i.e., it will not be destroyed when the GpuResources object gets cleaned
    /// up.
    /// We are guaranteed that all Faiss GPU work is ordered with respect to
    /// this stream upon exit from an index or other Faiss GPU call.
    void setDefaultStream(int device, cudaStream_t stream)
    {
        if (isInitialized(device)) {
            // A new series of calls may not be ordered with what was the previous
            // stream, so if the stream being specified is different, then we need to
            // ensure ordering between the two (new stream waits on old).
            auto it                 = userDefaultStreams_.find(device);
            cudaStream_t prevStream = nullptr;

            if (it != userDefaultStreams_.end()) {
                prevStream = it->second;
            } else {
                FAISS_ASSERT(defaultStreams_.count(device));
                prevStream = defaultStreams_[device];
            }

            if (prevStream != stream) { streamWait({stream}, {prevStream}); }
        }

        userDefaultStreams_[device] = stream;
    };

    /// Revert the default stream to the original stream managed by this resources
    /// object, in case someone called `setDefaultStream`.
    void revertDefaultStream(int device)
    {
        if (isInitialized(device)) {
            auto it = userDefaultStreams_.find(device);

            if (it != userDefaultStreams_.end()) {
                // There was a user stream set that we need to synchronize against
                cudaStream_t prevStream = userDefaultStreams_[device];

                FAISS_ASSERT(defaultStreams_.count(device));
                cudaStream_t newStream = defaultStreams_[device];

                streamWait({newStream}, {prevStream});
            }
        }

        userDefaultStreams_.erase(device);
    };

    /// Returns the stream for the given device on which all Faiss GPU work is
    /// ordered.
    /// We are guaranteed that all Faiss GPU work is ordered with respect to
    /// this stream upon exit from an index or other Faiss GPU call.
    cudaStream_t getDefaultStream(int device)
    {
        initializeForDevice(device);

        auto it = userDefaultStreams_.find(device);
        if (it != userDefaultStreams_.end()) {
            // There is a user override stream set
            return it->second;
        }

        // Otherwise, our base default stream
        return defaultStreams_[device];
    };

    /// Called to change the work ordering streams to the null stream
    /// for all devices
    void setDefaultNullStreamAllDevices()
    {
        for (int dev = 0; dev < getNumDevices(); ++dev) {
            setDefaultStream(dev, nullptr);
        }
    };

    /// If enabled, will print every GPU memory allocation and deallocation to
    /// standard output
    void setLogMemoryAllocations(bool enable) { allocLogging_ = enable; };

   public:
    /// Internal system calls

    /// Initialize resources for this device
    void initializeForDevice(int device)
    {
        if (isInitialized(device)) { return; }

        // If this is the first device that we're initializing, create our
        // pinned memory allocation
        if (defaultStreams_.empty() && pinnedMemSize_ > 0) {
            pinnedMemAlloc_     = pmr->allocate(pinnedMemSize_);
            pinnedMemAllocSize_ = pinnedMemSize_;
        }

        FAISS_ASSERT(device < getNumDevices());
        DeviceScope scope(device);

        // Make sure that device properties for all devices are cached
        auto& prop = getDeviceProperties(device);

        // Also check to make sure we meet our minimum compute capability (3.0)
        FAISS_ASSERT_FMT(prop.major >= 3,
                         "Device id %d with CC %d.%d not supported, "
                         "need 3.0+ compute capability",
                         device,
                         prop.major,
                         prop.minor);

        // Create streams
        cudaStream_t defaultStream = 0;
        CUDA_VERIFY(cudaStreamCreateWithFlags(&defaultStream, cudaStreamNonBlocking));

        defaultStreams_[device] = defaultStream;

        cudaStream_t asyncCopyStream = 0;
        CUDA_VERIFY(cudaStreamCreateWithFlags(&asyncCopyStream, cudaStreamNonBlocking));

        asyncCopyStreams_[device] = asyncCopyStream;

        std::vector<cudaStream_t> deviceStreams;
        for (int j = 0; j < kNumStreams; ++j) {
            cudaStream_t stream = 0;
            CUDA_VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

            deviceStreams.push_back(stream);
        }

        alternateStreams_[device] = std::move(deviceStreams);

        // Create cuBLAS handle

        // TODO: We need to be able to use this cublas handle within the raft handle
        cublasHandle_t blasHandle = 0;
        auto blasStatus           = cublasCreate(&blasHandle);
        FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
        blasHandles_[device] = blasHandle;

        // For CUDA 10 on V100, enabling tensor core usage would enable automatic
        // rounding down of inputs to f16 (though accumulate in f32) which results in
        // unacceptable loss of precision in general.
        // For CUDA 11 / A100, only enable tensor core support if it doesn't result in
        // a loss of precision.
#if CUDA_VERSION >= 11000
        cublasSetMathMode(blasHandle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
#endif

        FAISS_ASSERT(allocs_.count(device) == 0);
        allocs_[device] = std::unordered_map<void*, AllocRequest>();

        FAISS_ASSERT(tempMemory_.count(device) == 0);
        auto mem = std::unique_ptr<StackDeviceMemory>(
                new StackDeviceMemory(this,
                                      device,
                        // adjust for this specific device
                                      getDefaultTempMemForGPU(device, tempMemSize_)));

        tempMemory_.emplace(device, std::move(mem));
    };

    cublasHandle_t getBlasHandle(int device)
    {
        initializeForDevice(device);
        return blasHandles_[device];
    };

    std::vector<cudaStream_t> getAlternateStreams(int device)
    {
        initializeForDevice(device);
        return alternateStreams_[device];
    };

    /// Allocate non-temporary GPU memory
    void* allocMemory(const AllocRequest& req)
    {
        initializeForDevice(req.device);

        // We don't allocate a placeholder for zero-sized allocations
        if (req.size == 0) { return nullptr; }

        // Make sure that the allocation is a multiple of 16 bytes for alignment
        // purposes
        auto adjReq = req;
        adjReq.size = utils::roundUp(adjReq.size, (size_t)16);

        void* p = nullptr;

        if (allocLogging_) { std::cout << "RmmGpuResources: alloc " << adjReq.toString() << "\n"; }

        if (adjReq.space == MemorySpace::Temporary) {
            // If we don't have enough space in our temporary memory manager, we need
            // to allocate this request separately
            auto& tempMem = tempMemory_[adjReq.device];

            if (adjReq.size > tempMem->getSizeAvailable()) {
                // We need to allocate this ourselves
                AllocRequest newReq = adjReq;
                newReq.space        = MemorySpace::Device;
                newReq.type         = AllocType::TemporaryMemoryOverflow;

                return allocMemory(newReq);
            }

            // Otherwise, we can handle this locally
            p = tempMemory_[adjReq.device]->allocMemory(adjReq.stream, adjReq.size);

        } else if (adjReq.space == MemorySpace::Device) {
            p = cmr->allocate(adjReq.size, adjReq.stream);
        } else if (adjReq.space == MemorySpace::Unified) {
            p = mmr->allocate(adjReq.size, adjReq.stream);
        } else {
            FAISS_ASSERT_FMT(false, "unknown MemorySpace %d", (int)adjReq.space);
        }

        allocs_[adjReq.device][p] = adjReq;

        return p;
    };

    /// Returns a previous allocation
    void deallocMemory(int device, void* p)
    {
        FAISS_ASSERT(isInitialized(device));

        if (!p) { return; }

        auto& a = allocs_[device];
        auto it = a.find(p);
        FAISS_ASSERT(it != a.end());

        auto& req = it->second;

        if (allocLogging_) { std::cout << "RmmGpuResources: dealloc " << req.toString() << "\n"; }

        if (req.space == MemorySpace::Temporary) {
            tempMemory_[device]->deallocMemory(device, req.stream, req.size, p);
        } else if (req.space == MemorySpace::Device) {
            cmr->deallocate(p, req.size, req.stream);
        } else if (req.space == MemorySpace::Unified) {
            mmr->deallocate(p, req.size, req.stream);
        } else {
            FAISS_ASSERT_FMT(false, "unknown MemorySpace %d", (int)req.space);
        }

        a.erase(it);
    };

    size_t getTempMemoryAvailable(int device) const
    {
        FAISS_ASSERT(isInitialized(device));

        auto it = tempMemory_.find(device);
        FAISS_ASSERT(it != tempMemory_.end());

        return it->second->getSizeAvailable();
    };

    /// Export a description of memory used for Python
    std::map<int, std::map<std::string, std::pair<int, size_t>>> getMemoryInfo() const
    {
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
    };

    std::pair<void*, size_t> getPinnedMemory()
    {
        return std::make_pair(pinnedMemAlloc_, pinnedMemAllocSize_);
    };

    cudaStream_t getAsyncCopyStream(int device)
    {
        initializeForDevice(device);
        return asyncCopyStreams_[device];
    };

   private:
    /// Have GPU resources been initialized for this device yet?
    bool isInitialized(int device) const
    {
        // Use default streams as a marker for whether or not a certain
        // device has been initialized
        return defaultStreams_.count(device) != 0;
    };

    raft::handle_t &getRaftHandle(int device) {
        initializeForDevice(device);

        auto it = raftHandles_.find(device);
        if (it != raftHandles_.end()) {
            // There is a user override handle set
            return it->second;
        }

        // Otherwise, our base default handle
        return raftHandles_[device];
    }

    /// Adjust the default temporary memory allocation based on the total GPU
    /// memory size
    static size_t getDefaultTempMemForGPU(int device, size_t requested)
    {
        auto totalMem = device != -1 ? getDeviceProperties(device).totalGlobalMem
                                     : std::numeric_limits<size_t>::max();

        if (totalMem <= (size_t)4 * 1024 * 1024 * 1024) {
            // If the GPU has <= 4 GiB of memory, reserve 512 MiB

            if (requested > k4GiBTempMem) { return k4GiBTempMem; }
        } else if (totalMem <= (size_t)8 * 1024 * 1024 * 1024) {
            // If the GPU has <= 8 GiB of memory, reserve 1 GiB

            if (requested > k8GiBTempMem) { return k8GiBTempMem; }
        } else {
            // Never use more than 1.5 GiB
            if (requested > kMaxTempMem) { return kMaxTempMem; }
        }

        // use whatever lower limit the user requested
        return requested;
    };

   private:
    /// Set of currently outstanding memory allocations per device
    /// device -> (alloc request, allocated ptr)
    std::unordered_map<int, std::unordered_map<void*, AllocRequest>> allocs_;

    /// Temporary memory provider, per each device
    std::unordered_map<int, std::unique_ptr<StackDeviceMemory>> tempMemory_;

    /// Our default stream that work is ordered on, one per each device
    std::unordered_map<int, cudaStream_t> defaultStreams_;

    /// This contains particular streams as set by the user for
    /// ordering, if any
    std::unordered_map<int, cudaStream_t> userDefaultStreams_;

    /// Other streams we can use, per each device
    std::unordered_map<int, std::vector<cudaStream_t>> alternateStreams_;

    /// Async copy stream to use for GPU <-> CPU pinned memory copies
    std::unordered_map<int, cudaStream_t> asyncCopyStreams_;

    /// cuBLAS handle for each device
    std::unordered_map<int, cublasHandle_t> blasHandles_;

    /// Pinned memory allocation for use with this GPU
    void* pinnedMemAlloc_;
    size_t pinnedMemAllocSize_;

    /// Another option is to use a specified amount of memory on all
    /// devices
    size_t tempMemSize_;

    /// Amount of pinned memory we should allocate
    size_t pinnedMemSize_;

    /// Whether or not we log every GPU memory allocation and deallocation
    bool allocLogging_;

    // cuda_memory_resource
    std::unique_ptr<rmm::mr::device_memory_resource> cmr;

    // managed_memory_resource
    std::unique_ptr<rmm::mr::device_memory_resource> mmr;

    // pinned_memory_resource
    std::unique_ptr<rmm::mr::host_memory_resource> pmr;

    /// Our raft handle that maintains additional library resources, one per each device
    std::unordered_map<int, raft::handle_t> raftHandles_;

};

/// Default implementation of GpuResources that allocates a cuBLAS
/// stream and 2 streams for use, as well as temporary memory.
/// Internally, the Faiss GPU code uses the instance managed by getResources,
/// but this is the user-facing object that is internally reference counted.
class RmmGpuResources : public GpuResourcesProvider {
   public:
    RmmGpuResources() : res_(new RmmGpuResourcesImpl){};

    ~RmmGpuResources(){};

    std::shared_ptr<GpuResources> getResources() { return res_; };

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call cudaMalloc / cudaFree at the point of use
    void noTempMemory() { res_->noTempMemory(); };

    /// Specify that we wish to use a certain fixed size of memory on
    /// all devices as temporary memory. This is the upper bound for the GPU
    /// memory that we will reserve. We will never go above 1.5 GiB on any GPU;
    /// smaller GPUs (with <= 4 GiB or <= 8 GiB) will use less memory than that.
    /// To avoid any temporary memory allocation, pass 0.
    void setTempMemory(size_t size) { res_->setTempMemory(size); };

    /// Set amount of pinned memory to allocate, for async GPU <-> CPU
    /// transfers
    void setPinnedMemory(size_t size) { res_->setPinnedMemory(size); };

    /// Called to change the stream for work ordering. We do not own `stream`;
    /// i.e., it will not be destroyed when the GpuResources object gets cleaned
    /// up.
    /// We are guaranteed that all Faiss GPU work is ordered with respect to
    /// this stream upon exit from an index or other Faiss GPU call.
    void setDefaultStream(int device, cudaStream_t stream)
    {
        res_->setDefaultStream(device, stream);
    };

    /// Revert the default stream to the original stream managed by this resources
    /// object, in case someone called `setDefaultStream`.
    void revertDefaultStream(int device) { res_->revertDefaultStream(device); };

    /// Called to change the work ordering streams to the null stream
    /// for all devices
    void setDefaultNullStreamAllDevices() { res_->setDefaultNullStreamAllDevices(); };

    /// Export a description of memory used for Python
    std::map<int, std::map<std::string, std::pair<int, size_t>>> getMemoryInfo() const
    {
        return res_->getMemoryInfo();
    };

    /// Returns the current default stream
    cudaStream_t getDefaultStream(int device) { return res_->getDefaultStream(device); };

    /// Returns the current amount of temp memory available
    size_t getTempMemoryAvailable(int device) const { return res_->getTempMemoryAvailable(device); };

    /// Synchronize our default stream with the CPU
    void syncDefaultStreamCurrentDevice() { res_->syncDefaultStreamCurrentDevice(); };

    /// If enabled, will print every GPU memory allocation and deallocation to
    /// standard output
    void setLogMemoryAllocations(bool enable) { res_->setLogMemoryAllocations(enable); };

   private:
    std::shared_ptr<RmmGpuResourcesImpl> res_;
};

}  // namespace gpu
}  // namespace faiss