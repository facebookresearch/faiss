// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Objective-C++ header (uses MetalResources).
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu_metal/MetalResources.h>
#include <memory>

namespace faiss {
namespace gpu_metal {

/// Configuration for Metal index (mirrors GpuIndexConfig roles).
struct MetalIndexConfig {
    int device = 0;
};

/// Base class for Metal-backed indexes. Mirrors faiss::gpu::GpuIndex.
class MetalIndex : public faiss::Index {
   public:
    MetalIndex(
            std::shared_ptr<MetalResources> resources,
            int dims,
            faiss::MetricType metric,
            float metricArg,
            MetalIndexConfig config = MetalIndexConfig());

    int getDevice() const {
        return config_.device;
    }
    std::shared_ptr<MetalResources> getResources() {
        return resources_;
    }
    std::shared_ptr<const MetalResources> getResources() const {
        return resources_;
    }

   protected:
    std::shared_ptr<MetalResources> resources_;
    MetalIndexConfig config_;
};

} // namespace gpu_metal
} // namespace faiss
