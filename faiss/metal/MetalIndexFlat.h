#pragma once

#include <faiss/IndexFlat.h>
#include "MetalResources.h"

namespace faiss {
namespace metal {

class MetalIndexFlat : public faiss::IndexFlat {
   public:
    MetalIndexFlat(
            std::shared_ptr<MetalResources> resources,
            int d,
            faiss::MetricType metric = faiss::METRIC_L2);

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

   private:
    std::shared_ptr<MetalResources> resources_;
    id<MTLBuffer> vectors_;
    idx_t ntotal_;
};

} // namespace metal
} // namespace faiss
