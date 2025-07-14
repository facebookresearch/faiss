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
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

   private:
    void sync_with_gpu();
    std::shared_ptr<MetalResources> resources_;
    id<MTLBuffer> vectors_;
    id<MTLBuffer> levels_;
    id<MTLBuffer> neighbors_;
    id<MTLBuffer> graph_offsets_;
    id<MTLBuffer> graph_neighbors_;
    bool is_gpu_sync_;
};

} // namespace metal
} // namespace faiss
