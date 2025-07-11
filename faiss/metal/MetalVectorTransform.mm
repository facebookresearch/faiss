#include "MetalVectorTransform.h"

namespace faiss {
namespace metal {

MetalVectorTransform::MetalVectorTransform(std::shared_ptr<MetalResources> resources)
    : resources_(resources) {}

void MetalVectorTransform::apply_noalloc(idx_t n, const float* x, float* xt) const {
    FAISS_THROW_MSG("MetalVectorTransform::apply_noalloc not yet implemented");
}

} // namespace metal
} // namespace faiss
