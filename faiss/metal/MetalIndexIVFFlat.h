#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/MetricType.h>
#include "MetalResources.h"

namespace faiss {
namespace metal {

class MetalIndexIVFFlat : public faiss::IndexIVFFlat {
   public:
    MetalIndexIVFFlat(
            std::shared_ptr<MetalResources> resources,
            faiss::Index* quantizer,
            size_t d,
            size_t nlist,
            faiss::MetricType metric = faiss::METRIC_L2);

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

   private:
    std::shared_ptr<MetalResources> resources_;
    id<MTLBuffer> vectors_;
    id<MTLBuffer> ids_;
    id<MTLBuffer> offsets_;
};

} // namespace metal
} // namespace faiss
