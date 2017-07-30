/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "../../FaissAssert.h"

namespace faiss { namespace gpu {

template <typename GpuIndex>
IndexWrapper<GpuIndex>::IndexWrapper(
  int numGpus,
  std::function<std::unique_ptr<GpuIndex>(GpuResources*, int)> init) {
  FAISS_ASSERT(numGpus <= faiss::gpu::getNumDevices());
  for (int i = 0; i < numGpus; ++i) {
    auto res = std::unique_ptr<faiss::gpu::StandardGpuResources>(
      new StandardGpuResources);

    subIndex.emplace_back(init(res.get(), i));
    resources.emplace_back(std::move(res));
  }

  if (numGpus > 1) {
    // create proxy
    proxyIndex =
      std::unique_ptr<faiss::gpu::IndexProxy>(new faiss::gpu::IndexProxy);

    for (auto& index : subIndex) {
      proxyIndex->addIndex(index.get());
    }
  }
}

template <typename GpuIndex>
faiss::Index*
IndexWrapper<GpuIndex>::getIndex() {
  if ((bool) proxyIndex) {
    return proxyIndex.get();
  } else {
    FAISS_ASSERT(!subIndex.empty());
    return subIndex.front().get();
  }
}

template <typename GpuIndex>
void
IndexWrapper<GpuIndex>::runOnIndices(std::function<void(GpuIndex*)> f) {

  if ((bool) proxyIndex) {
    proxyIndex->runOnIndex(
      [f](faiss::Index* index) {
        f(dynamic_cast<GpuIndex*>(index));
      });
  } else {
    FAISS_ASSERT(!subIndex.empty());
    f(subIndex.front().get());
  }
}

template <typename GpuIndex>
void
IndexWrapper<GpuIndex>::setNumProbes(int nprobe) {
  runOnIndices([nprobe](GpuIndex* index) {
      index->setNumProbes(nprobe);
    });
}

} }
