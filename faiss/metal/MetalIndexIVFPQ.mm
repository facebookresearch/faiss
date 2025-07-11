#include "MetalIndexIVFPQ.h"

namespace faiss {
namespace metal {

MetalIndexIVFPQ::MetalIndexIVFPQ(
        std::shared_ptr<MetalResources> resources,
        faiss::Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        faiss::MetricType metric)
    : faiss::IndexIVFPQ(quantizer, d, nlist, M, nbits_per_idx, metric),
      resources_(resources) {}

void MetalIndexIVFPQ::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    FAISS_THROW_MSG("Metal IndexIVFPQ::add_with_ids not yet implemented");
}

void MetalIndexIVFPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_MSG("Metal IndexIVFPQ::search not yet implemented");
}

} // namespace metal
} // namespace faiss
