#pragma once

#include <stddef.h>
#include <faiss/c_api/faiss_c.h>

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_HANDLE(FaissMetalResources);

int faiss_MetalResources_getDevice(FaissMetalResources* res, int device, id<MTLDevice>* out);
int faiss_MetalResources_getCommandQueue(FaissMetalResources* res, int device, id<MTLCommandQueue>* out);

int faiss_get_default_metal_resources(FaissMetalResources** p_res);

#ifdef __cplusplus
}
#endif
