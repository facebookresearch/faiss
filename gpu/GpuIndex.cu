/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "GpuIndex.h"
#include "../FaissAssert.h"
#include "GpuResources.h"
#include "utils/DeviceUtils.h"
#include <stdio.h>

namespace faiss { namespace gpu {

// Default size for which we page add or search
constexpr size_t kAddPageSize = (size_t) 256 * 1024 * 1024;
constexpr size_t kSearchPageSize = (size_t) 256 * 1024 * 1024;

// Or, maximum number of vectors to consider per page of add or search
constexpr size_t kAddVecSize = (size_t) 512 * 1024;

// Use a smaller search size, as precomputed code usage on IVFPQ
// requires substantial amounts of memory
// FIXME: parameterize based on algorithm need
constexpr size_t kSearchVecSize = (size_t) 32 * 1024;

GpuIndex::GpuIndex(GpuResources* resources,
                   int dims,
                   faiss::MetricType metric,
                   GpuIndexConfig config) :
    Index(dims, metric),
    resources_(resources),
    device_(config.device),
    memorySpace_(config.memorySpace) {
  FAISS_THROW_IF_NOT_FMT(device_ < getNumDevices(),
                     "Invalid GPU device %d", device_);

  FAISS_THROW_IF_NOT_MSG(dims > 0, "Invalid number of dimensions");

#ifdef FAISS_UNIFIED_MEM
  FAISS_THROW_IF_NOT_FMT(
    memorySpace_ == MemorySpace::Device ||
    (memorySpace_ == MemorySpace::Unified &&
     getFullUnifiedMemSupport(device_)),
    "Device %d does not support full CUDA 8 Unified Memory (CC 6.0+)",
    config.device);
#else
  FAISS_THROW_IF_NOT_MSG(memorySpace_ == MemorySpace::Device,
                     "Must compile with CUDA 8+ for Unified Memory support");
#endif

  FAISS_ASSERT(resources_);
  resources_->initializeForDevice(device_);
}

void
GpuIndex::add(Index::idx_t n, const float* x) {
  addInternal_(n, x, nullptr);
}

void
GpuIndex::add_with_ids(Index::idx_t n,
                       const float* x,
                       const Index::idx_t* ids) {
  addInternal_(n, x, ids);
}

void
GpuIndex::addInternal_(Index::idx_t n,
                       const float* x,
                       const Index::idx_t* ids) {
  DeviceScope scope(device_);

  FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

  if (n > 0) {
    size_t totalSize = n * (size_t) this->d * sizeof(float);

    if (totalSize > kAddPageSize || n > kAddVecSize) {
      // How many vectors fit into kAddPageSize?
      size_t maxNumVecsForPageSize =
        kAddPageSize / ((size_t) this->d * sizeof(float));

      // Always add at least 1 vector, if we have huge vectors
      maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t) 1);

      size_t tileSize = std::min((size_t) n, maxNumVecsForPageSize);
      tileSize = std::min(tileSize, kSearchVecSize);

      for (size_t i = 0; i < n; i += tileSize) {
        size_t curNum = std::min(tileSize, n - i);

        addImpl_(curNum,
                 x + i * (size_t) this->d,
                 ids ? ids + i : nullptr);
      }
    } else {
      addImpl_(n, x, ids);
    }
  }
}

void
GpuIndex::search(Index::idx_t n,
                 const float* x,
                 Index::idx_t k,
                 float* distances,
                 Index::idx_t* labels) const {
  DeviceScope scope(device_);

  FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

  if (n > 0) {
    size_t totalSize = n * (size_t) this->d * sizeof(float);

    if ((totalSize > kSearchPageSize) || (n > kSearchVecSize)) {
      // How many vectors fit into kSearchPageSize?
      // Just consider `x`, not the size of `distances` or `labels`
      // since they should be small, relatively speaking
      size_t maxNumVecsForPageSize =
        kSearchPageSize / ((size_t) this->d * sizeof(float));

      // Always search at least 1 vector, if we have huge vectors
      maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t) 1);

      size_t tileSize = std::min((size_t) n, maxNumVecsForPageSize);
      tileSize = std::min(tileSize, kSearchVecSize);

      for (size_t i = 0; i < n; i += tileSize) {
        size_t curNum = std::min(tileSize, n - i);

        searchImpl_(curNum,
                    x + i * (size_t) this->d,
                    k,
                    distances + i * k,
                    labels + i * k);
      }
    } else {
      searchImpl_(n, x, k, distances, labels);
    }
  }
}

} } // namespace
