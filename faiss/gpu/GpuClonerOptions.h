/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

    /// use the cuVS implementation
#if defined USE_NVIDIA_CUVS
    bool use_cuvs = true;
#else
    bool use_cuvs = false;
#endif

    /// This flag controls the CPU fallback logic for coarse quantizer
    /// component of the index. When set to false (default), the cloner will
    /// throw an exception for indices not implemented on GPU. When set to
    /// true, it will fallback to a CPU implementation.
    bool allowCpuCoarseQuantizer = false;
};

struct GpuMultipleClonerOptions : public GpuClonerOptions {
    /// Whether to shard the index across GPUs, versus replication
    /// across GPUs
    bool shard = false;

    /// IndexIVF::copy_subset_to subset type
    int shard_type = 1;

    /// set to true if an IndexIVF is to be dispatched to multiple GPUs with a
    /// single common IVF quantizer, ie. only the inverted lists are sharded on
    /// the sub-indexes (uses an IndexShardsIVF)
    bool common_ivf_quantizer = false;
};

} // namespace gpu
} // namespace faiss
