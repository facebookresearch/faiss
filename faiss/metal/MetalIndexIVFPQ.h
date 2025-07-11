#pragma once

#include <faiss/IndexIVFPQ.h>
#include "MetalResources.h"

namespace faiss {
namespace metal {

class MetalIndexIVFPQ : public faiss::IndexIVFPQ {
   public:
    MetalIndexIVFPQ(
            std::shared_ptr<MetalResources> resources,
            faiss::Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits_per_idx,
            faiss::MetricType metric = faiss::METRIC_L2);

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

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
