/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <memory>

namespace faiss {
struct IndexIVFFlat;
}

namespace faiss {
namespace gpu {

class IVFFlat;
class GpuIndexFlat;

/// Wrapper around the GPU implementation that looks like
/// faiss::gpu::GpuIndexIVFFlat
class RaftIndexIVFFlat : public GpuIndexIVFFlat {
   public:
    /// Construct from a pre-existing faiss::IndexIVFFlat instance, copying
    /// data over to the given GPU, if the input index is trained.
    RaftIndexIVFFlat(
            GpuResourcesProvider* provider,
            const faiss::IndexIVFFlat* index,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    /// Constructs a new instance with an empty flat quantizer; the user
    /// provides the number of lists desired.
    RaftIndexIVFFlat(
            GpuResourcesProvider* provider,
            int dims,
            int nlist,
            faiss::MetricType metric,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    ~RaftIndexIVFFlat() override;
};

} // namespace gpu
} // namespace faiss
