/**
* Copyright (c) Facebook, Inc. and its affiliates.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/RmmGpuResources.h>
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

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

namespace faiss::gpu {

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
}


/// RMM implementation of the GpuResources object that provides for a
/// temporary memory manager
RmmGpuResourcesImpl::RmmGpuResourcesImpl(): StandardGpuResourcesImpl(),
          cmr(new rmm::mr::cuda_memory_resource),
          mmr(new rmm::mr::managed_memory_resource),
          pmr(new rmm::mr::pinned_memory_resource){}

RmmGpuResourcesImpl::~RmmGpuResourcesImpl() {}


   /// Initialize resources for this device
   void RmmGpuResourcesImpl::initializeForDevice(int device)
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

   /// Allocate non-temporary GPU memory
   void* RmmGpuResourcesImpl::allocMemory(const AllocRequest& req)
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
   void RmmGpuResourcesImpl::deallocMemory(int device, void* p)
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


   /// Default implementation of GpuResources that allocates a cuBLAS
   /// stream and 2 streams for use, as well as temporary memory.
   /// Internally, the Faiss GPU code uses the instance managed by getResources,
   /// but this is the user-facing object that is internally reference counted.
   RmmGpuResources::RmmGpuResources(): res_(new RmmGpuResourcesImpl){};
   RmmGpuResources::~RmmGpuResources()  {};
   std::shared_ptr<GpuResources> RmmGpuResources::getResources() { return res_; };

};
