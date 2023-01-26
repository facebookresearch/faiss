/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndicesOptions.h>

namespace faiss {
namespace gpu {

/// set some options on how to copy to GPU
struct GpuClonerOptions {
    GpuClonerOptions();

    /// how should indices be stored on index types that support indices
    /// (anything but GpuIndexFlat*)?
    IndicesOptions indicesOptions = INDICES_64_BIT;

    /// is the coarse quantizer in float16?
    bool useFloat16CoarseQuantizer = false;

    /// for GpuIndexIVFFlat, is storage in float16?
    /// for GpuIndexIVFPQ, are intermediate calculations in float16?
    bool useFloat16 = false;

    /// use precomputed tables?
    bool usePrecomputed = false;

    /// reserve vectors in the invfiles?
    long reserveVecs = 0;

    /// For GpuIndexFlat, store data in transposed layout?
    bool storeTransposed = false;

    /// Set verbose options on the index
    bool verbose = false;
};

struct GpuMultipleClonerOptions : public GpuClonerOptions {
    GpuMultipleClonerOptions();

    /// Whether to shard the index across GPUs, versus replication
    /// across GPUs
    bool shard = false;

    /// IndexIVF::copy_subset_to subset type
    int shard_type = 1;
};

} // namespace gpu
} // namespace faiss
