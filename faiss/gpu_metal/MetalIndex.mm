// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalIndex.h"
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu_metal {

MetalIndex::MetalIndex(
        std::shared_ptr<MetalResources> resources,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        MetalIndexConfig config)
        : Index(dims, metric),
          resources_(std::move(resources)),
          config_(config) {
    metric_arg = metricArg;
    FAISS_THROW_IF_NOT_MSG(
            config_.device >= 0 && config_.device < 1,
            "Metal backend supports only device 0 (single GPU).");
    FAISS_THROW_IF_NOT(resources_ != nullptr);
    FAISS_THROW_IF_NOT(resources_->isAvailable());
}

} // namespace gpu_metal
} // namespace faiss
