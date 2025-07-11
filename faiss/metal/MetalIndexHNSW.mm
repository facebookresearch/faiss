#include "MetalIndexHNSW.h"

namespace faiss {
namespace metal {

MetalIndexHNSW::MetalIndexHNSW(
        std::shared_ptr<MetalResources> resources,
        int d,
        int M,
        faiss::MetricType metric)
    : faiss::IndexHNSW(d, M, metric),
      resources_(resources) {}

void MetalIndexHNSW::add(idx_t n, const float* x) {
    FAISS_THROW_MSG("Metal IndexHNSW::add not yet implemented");
}

void MetalIndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_MSG("Metal IndexHNSW::search not yet implemented");
}

} // namespace metal
} // namespace faiss
