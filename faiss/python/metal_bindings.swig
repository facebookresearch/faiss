/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Metal-specific Python bindings

#ifdef FAISS_ENABLE_METAL

%shared_ptr(faiss::metal::MetalResources);

%{
#include <faiss/metal/MetalResources.h>
%}

%extend faiss::metal::MetalResources {
    static std::shared_ptr<faiss::metal::MetalResources> get_default() {
        return faiss::metal::get_default_metal_resources();
    }
}

namespace faiss {
namespace metal {

// Python-friendly wrapper for getting default Metal resources
%pythoncode %{
def get_default_metal_resources():
    """Get the default Metal resources singleton."""
    return MetalResources.get_default()

def is_metal_available():
    """Check if Metal is available on this system."""
    try:
        res = get_default_metal_resources()
        return res is not None
    except:
        return False
%}

} // namespace metal
} // namespace faiss

// Add convenience constructors for Python
%extend faiss::metal::MetalIndexFlat {
    MetalIndexFlat(int d, faiss::MetricType metric = faiss::METRIC_L2) {
        auto resources = faiss::metal::get_default_metal_resources();
        return new faiss::metal::MetalIndexFlat(resources, d, metric);
    }
}

%extend faiss::metal::MetalIndexHNSW {
    MetalIndexHNSW(int d, int M = 32, faiss::MetricType metric = faiss::METRIC_L2) {
        auto resources = faiss::metal::get_default_metal_resources();
        return new faiss::metal::MetalIndexHNSW(resources, d, M, metric);
    }
}

%extend faiss::metal::MetalIndexIVFFlat {
    MetalIndexIVFFlat(faiss::Index* quantizer, int d, int nlist, 
                      faiss::MetricType metric = faiss::METRIC_L2) {
        auto resources = faiss::metal::get_default_metal_resources();
        return new faiss::metal::MetalIndexIVFFlat(resources, quantizer, d, nlist, metric);
    }
}

%extend faiss::metal::MetalIndexIVFPQ {
    MetalIndexIVFPQ(faiss::Index* quantizer, int d, int nlist, 
                    int M, int nbits, faiss::MetricType metric = faiss::METRIC_L2) {
        auto resources = faiss::metal::get_default_metal_resources();
        return new faiss::metal::MetalIndexIVFPQ(resources, quantizer, d, nlist, M, nbits, metric);
    }
}

#endif // FAISS_ENABLE_METAL