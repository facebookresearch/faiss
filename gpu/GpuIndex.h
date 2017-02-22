
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

  // redeclare an abstract method to quiet SWIG warning
  virtual void add(faiss::Index::idx_t,float const *) = 0;

 protected:
  /// Manages streans, cuBLAS handles and scratch memory for devices
  GpuResources* resources_;

  /// The GPU device we are resident on
  int device_;
};

} } // namespace
