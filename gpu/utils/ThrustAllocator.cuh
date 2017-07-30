/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <cuda.h>
#include <unordered_set>

namespace faiss { namespace gpu {

/// Allocator for Thrust that comes out of a specified memory space
class GpuResourcesThrustAllocator {
 public:
  typedef char value_type;

  GpuResourcesThrustAllocator(void* mem, size_t size)
      : start_((char*) mem),
        cur_((char*) mem),
        end_((char*) mem + size) {
  }

  ~GpuResourcesThrustAllocator() {
  }

  char* allocate(std::ptrdiff_t size) {
    if (size <= (end_ - cur_)) {
      char* p = cur_;
      cur_ += size;
      FAISS_ASSERT(cur_ <= end_);

      return p;
    } else {
      char* p = nullptr;
      CUDA_VERIFY(cudaMalloc(&p, size));
      mallocAllocs_.insert(p);
      return p;
    }
  }

  void deallocate(char* p, size_t size) {
    // Allocations could be returned out-of-order; ignore those we
    // didn't cudaMalloc
    auto it = mallocAllocs_.find(p);
    if (it != mallocAllocs_.end()) {
      CUDA_VERIFY(cudaFree(p));
      mallocAllocs_.erase(it);
    }
  }

 private:
  char* start_;
  char* cur_;
  char* end_;
  std::unordered_set<char*> mallocAllocs_;
};


} } // namespace
