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
    // For now, we use the CPU-based implementation for graph construction.
    faiss::IndexHNSW::add(n, x);
}

void MetalIndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    // For now, we use the CPU-based implementation for search.
    faiss::IndexHNSW::search(n, x, k, distances, labels);
}

} // namespace metal
} // namespace faiss
