// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * C++-only API for Python/SWIG. No Objective-C types so SWIG can
 * parse it. Implemented in MetalPythonBridge.mm.
 */

#pragma once

#include <faiss/Index.h>

namespace faiss {
namespace gpu_metal {

/// Opaque holder for Metal resources.
struct StandardMetalResourcesHolder {
    void* impl = nullptr;
    StandardMetalResourcesHolder();
    ~StandardMetalResourcesHolder();
    StandardMetalResourcesHolder(const StandardMetalResourcesHolder&) = delete;
    StandardMetalResourcesHolder& operator=(
            const StandardMetalResourcesHolder&) = delete;
};

/// Same names as GPU API for unified Python binding.
int get_num_gpus();
void gpu_profiler_start();
void gpu_profiler_stop();
void gpu_sync_all_devices();

/// Clone CPU index to Metal GPU. Caller owns returned index.
faiss::Index* index_cpu_to_gpu(
        StandardMetalResourcesHolder* res,
        int device,
        const faiss::Index* index);

/// Copy Metal index back to CPU. Caller owns returned index.
faiss::Index* index_gpu_to_cpu(const faiss::Index* index);

} // namespace gpu_metal
} // namespace faiss
