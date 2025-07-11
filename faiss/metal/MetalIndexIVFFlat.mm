#include "MetalIndexIVFFlat.h"

namespace faiss {
namespace metal {

MetalIndexIVFFlat::MetalIndexIVFFlat(
        std::shared_ptr<MetalResources> resources,
        faiss::Index* quantizer,
        size_t d,
        size_t nlist,
        faiss::MetricType metric)
    : faiss::IndexIVFFlat(quantizer, d, nlist, metric),
      resources_(resources) {}

void MetalIndexIVFFlat::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_MSG("Metal IndexIVFFlat::add_with_ids not yet implemented");
}

void MetalIndexIVFFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_MSG("Metal IndexIVFFlat::search not yet implemented");
}

} // namespace metal
} // namespace faiss
