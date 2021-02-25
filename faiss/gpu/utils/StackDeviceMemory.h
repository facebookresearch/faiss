/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <faiss/gpu/GpuResources.h>
#include <list>
#include <memory>
#include <tuple>
#include <unordered_map>

namespace faiss {
namespace gpu {

/// Device memory manager that provides temporary memory allocations
/// out of a region of memory, for a single device
class StackDeviceMemory {
   public:
    /// Allocate a new region of memory that we manage
    StackDeviceMemory(GpuResources* res, int device, size_t allocPerDevice);

    /// Manage a region of memory for a particular device, with or
    /// without ownership
    StackDeviceMemory(int device, void* p, size_t size, bool isOwner);

    ~StackDeviceMemory();

    int getDevice() const;

    /// All allocations requested should be a multiple of 16 bytes
    void* allocMemory(cudaStream_t stream, size_t size);
    void deallocMemory(int device, cudaStream_t, size_t size, void* p);

    size_t getSizeAvailable() const;
    std::string toString() const;

   protected:
    /// Previous allocation ranges and the streams for which
    /// synchronization is required
    struct Range {
        inline Range(char* s, char* e, cudaStream_t str)
                : start_(s), end_(e), stream_(str) {}

        // References a memory range [start, end)
        char* start_;
        char* end_;
        cudaStream_t stream_;
    };

    struct Stack {
        /// Constructor that allocates memory via cudaMalloc
        Stack(GpuResources* res, int device, size_t size);

        ~Stack();

        /// Returns how much size is available for an allocation without
        /// calling cudaMalloc
        size_t getSizeAvailable() const;

        /// Obtains an allocation; all allocations are guaranteed to be 16
        /// byte aligned
        char* getAlloc(size_t size, cudaStream_t stream);

        /// Returns an allocation
        void returnAlloc(char* p, size_t size, cudaStream_t stream);

        /// Returns the stack state
        std::string toString() const;

        /// Our GpuResources object
        GpuResources* res_;

        /// Device this allocation is on
        int device_;

        /// Where our temporary memory buffer is allocated; we allocate starting
        /// 16 bytes into this
        char* alloc_;

        /// Total size of our allocation
        size_t allocSize_;

        /// Our temporary memory region; [start_, end_) is valid
        char* start_;
        char* end_;

        /// Stack head within [start, end)
        char* head_;

        /// List of previous last users of allocations on our stack, for
        /// possible synchronization purposes
        std::list<Range> lastUsers_;

        /// What's the high water mark in terms of memory used from the
        /// temporary buffer?
        size_t highWaterMemoryUsed_;
    };

    /// Our device
    int device_;

    /// Memory stack
    Stack stack_;
};

} // namespace gpu
} // namespace faiss
