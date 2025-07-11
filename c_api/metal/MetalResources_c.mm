#include "faiss/c_api/metal/MetalResources_c.h"
#include "faiss/metal/MetalResources.h"
#include "faiss/c_api/macros_impl.h"

using faiss::metal::MetalResources;

DEFINE_DESTRUCTOR(MetalResources)

int faiss_MetalResources_getDevice(FaissMetalResources* res, int device, id<MTLDevice>* out) {
    try {
        auto o = reinterpret_cast<MetalResources*>(res)->getDevice(device);
        *out = o;
    }
    CATCH_AND_HANDLE
}

int faiss_MetalResources_getCommandQueue(FaissMetalResources* res, int device, id<MTLCommandQueue>* out) {
    try {
        auto o = reinterpret_cast<MetalResources*>(res)->getCommandQueue(device);
        *out = o;
    }
    CATCH_AND_HANDLE
}

int faiss_get_default_metal_resources(FaissMetalResources** p_res) {
    try {
        auto res = faiss::metal::get_default_metal_resources();
        *p_res = reinterpret_cast<FaissMetalResources*>(res.get());
    }
    CATCH_AND_HANDLE
}
