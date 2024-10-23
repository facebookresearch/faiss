/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <faiss/gpu/GpuResources.h>
#include <thrust/device_vector.h>
#include <unordered_set>

namespace faiss {
namespace gpu {

/// Allocator for Thrust that comes out of a specified memory space
class ThrustAllocator {
   public:
    typedef char value_type;

    inline ThrustAllocator(
            GpuResources* res,
            cudaStream_t stream,
            void* mem,
            size_t size)
            : res_(res),
              stream_(stream),
              start_((char*)mem),
              cur_((char*)mem),
              end_((char*)mem + size) {}

    inline ~ThrustAllocator() {
        // In the case of an exception being thrown, we may not have called
        // deallocate on all of our sub-allocations. Free them here
        for (auto p : mallocAllocs_) {
            res_->deallocMemory(getCurrentDevice(), p);
        }
    }

    inline char* allocate(std::ptrdiff_t size) {
        if (size <= (end_ - cur_)) {
            char* p = cur_;
            cur_ += size;
            FAISS_ASSERT(cur_ <= end_);

            return p;
        } else {
            // FIXME: we cannot use temporary memory for new requests because
            // the current temporary memory allocator cannot handle stream
            // synchronization at present, so just allocate through the general
            // device
            char* p = (char*)res_->allocMemory(AllocRequest(
                    makeDevAlloc(AllocType::Other, stream_), size));

            mallocAllocs_.insert(p);
            return p;
        }
    }

    inline void deallocate(char* p, size_t size) {
        // Allocations could be returned out-of-order; ignore those we
        // didn't cudaMalloc
        auto it = mallocAllocs_.find(p);
        if (it != mallocAllocs_.end()) {
            res_->deallocMemory(getCurrentDevice(), p);
            mallocAllocs_.erase(it);
        }
    }

   private:
    GpuResources* res_;
    cudaStream_t stream_;
    char* start_;
    char* cur_;
    char* end_;
    std::unordered_set<char*> mallocAllocs_;
};

} // namespace gpu
} // namespace faiss
