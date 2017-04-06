
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "../Index.h"

namespace faiss { namespace gpu {

class GpuResources;

class GpuIndex : public faiss::Index {
 public:
  GpuIndex(GpuResources* resources,
           int device,
           int dims,
           faiss::MetricType metric);

  inline int getDevice() const {
    return device_;
  }

  GpuResources* getResources() {
    return resources_;
  }

  /// `x` can be resident on the CPU or any GPU; copies are performed
  /// as needed
  /// Handles paged adds if the add set is too large; calls addInternal_
  virtual void add(faiss::Index::idx_t, const float* x);

  /// `x` and `ids` can be resident on the CPU or any GPU; copies are
  /// performed as needed
  /// Handles paged adds if the add set is too large; calls addInternal_
  virtual void add_with_ids(Index::idx_t n,
                            const float* x,
                            const Index::idx_t* ids);

  /// `x`, `distances` and `labels` can be resident on the CPU or any
  /// GPU; copies are performed as needed
  virtual void search(faiss::Index::idx_t n,
                      const float* x,
                      faiss::Index::idx_t k,
                      float* distances,
                      faiss::Index::idx_t* labels) const;


 protected:
  /// Handles paged adds if the add set is too large, passes to
  /// addImpl_ to actually perform the add for the current page
  void addInternal_(Index::idx_t n,
                    const float* x,
                    const Index::idx_t* ids);

  /// Overridden to actually perform the add
  virtual void addImpl_(Index::idx_t n,
                        const float* x,
                        const Index::idx_t* ids) = 0;

  /// Overridden to actually perform the search
  virtual void searchImpl_(faiss::Index::idx_t n,
                           const float* x,
                           faiss::Index::idx_t k,
                           float* distances,
                           faiss::Index::idx_t* labels) const = 0;

 protected:
  /// Manages streans, cuBLAS handles and scratch memory for devices
  GpuResources* resources_;

  /// The GPU device we are resident on
  int device_;
};

} } // namespace
