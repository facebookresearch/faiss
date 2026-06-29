// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Mirrors the role of StandardGpuResources for the Metal backend.
 */

#pragma once

#include <faiss/gpu_metal/MetalResources.h>
#include <memory>

namespace faiss {
namespace gpu_metal {

/// Default Metal resources (single device). Use with index_cpu_to_metal_gpu.
class StandardMetalResources {
   public:
    StandardMetalResources();
    std::shared_ptr<MetalResources> getResources() const {
        return res_;
    }
    bool isAvailable() const {
        return res_ && res_->isAvailable();
    }

   private:
    std::shared_ptr<MetalResources> res_;
};

} // namespace gpu_metal
} // namespace faiss
