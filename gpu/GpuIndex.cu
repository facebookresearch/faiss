
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "GpuIndex.h"
#include "../FaissAssert.h"
#include "GpuResources.h"
#include "utils/DeviceUtils.h"

namespace faiss { namespace gpu {

/// Default size for which we page add or search
constexpr size_t kAddPageSize = (size_t) 256 * 1024 * 1024;
constexpr size_t kSearchPageSize = (size_t) 256 * 1024 * 1024;

GpuIndex::GpuIndex(GpuResources* resources,
                   int device,
                   int dims,
                   faiss::MetricType metric) :
    Index(dims, metric),
    resources_(resources),
    device_(device) {
  FAISS_ASSERT(device_ < getNumDevices());

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
  FAISS_ASSERT(this->is_trained);

  if (n > 0) {
    size_t totalSize = n * (size_t) this->d * sizeof(float);

    if (totalSize > kAddPageSize) {
      // How many vectors fit into kAddPageSize?
      size_t numVecsPerPage =
        kAddPageSize / ((size_t) this->d * sizeof(float));

      // Always add at least 1 vector, if we have huge vectors
      numVecsPerPage = std::max(numVecsPerPage, (size_t) 1);

      for (size_t i = 0; i < n; i += numVecsPerPage) {
        size_t curNum = std::min(numVecsPerPage, n - i);

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
  FAISS_ASSERT(this->is_trained);

  if (n > 0) {
    size_t totalSize = n * (size_t) this->d * sizeof(float);

    if (totalSize > kSearchPageSize) {
      // How many vectors fit into kSearchPageSize?
      // Just consider `x`, not the size of `distances` or `labels`
      // since they should be small, relatively speaking
      size_t numVecsPerPage =
        kSearchPageSize / ((size_t) this->d * sizeof(float));

      // Always search at least 1 vector, if we have huge vectors
      numVecsPerPage = std::max(numVecsPerPage, (size_t) 1);

      for (size_t i = 0; i < n; i += numVecsPerPage) {
        size_t curNum = std::min(numVecsPerPage, n - i);

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
