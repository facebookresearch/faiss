#pragma once

#include <faiss/VectorTransform.h>
#include "MetalResources.h"

namespace faiss {
namespace metal {

class MetalVectorTransform : public faiss::VectorTransform {
   public:
    MetalVectorTransform(std::shared_ptr<MetalResources> resources);

    void apply_noalloc(idx_t n, const float* x, float* xt) const override;

   private:
    std::shared_ptr<MetalResources> resources_;
};

} // namespace metal
} // namespace faiss
