#pragma once

#include <faiss/IndexHNSW.h>
#include "MetalResources.h"

namespace faiss {
namespace metal {

class MetalIndexHNSW : public faiss::IndexHNSW {
   public:
    MetalIndexHNSW(
            std::shared_ptr<MetalResources> resources,
            int d,
            int M, faiss::MetricType metric = faiss::METRIC_L2);

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

   private:
    std::shared_ptr<MetalResources> resources_;
};

} // namespace metal
} // namespace faiss
