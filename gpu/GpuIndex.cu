
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

} } // namespace
