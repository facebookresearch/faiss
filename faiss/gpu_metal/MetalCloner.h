// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Clone CPU <-> Metal GPU. Mirrors GpuCloner roles for Metal backend.
 */

#pragma once

#include <faiss/Index.h>

namespace faiss {
namespace gpu_metal {

class StandardMetalResources;

/// Returns the number of Metal "devices" (1 if Metal is available, else 0).
int get_num_gpus();

/// Clone a CPU index to Metal GPU. Supports IndexFlat, IndexFlatL2,
/// IndexFlatIP. device must be 0. Caller owns the returned index.
faiss::Index* index_cpu_to_metal_gpu(
        StandardMetalResources* res,
        int device,
        const faiss::Index* index);

/// Copy a Metal index back to CPU. Supports MetalIndexFlat -> IndexFlat.
/// Caller owns the returned index.
faiss::Index* index_metal_gpu_to_cpu(const faiss::Index* index);

} // namespace gpu_metal
} // namespace faiss
