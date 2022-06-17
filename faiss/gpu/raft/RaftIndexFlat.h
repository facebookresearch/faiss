/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndex.h>
#include <memory>

namespace faiss {

struct IndexFlat;
struct IndexFlatL2;
struct IndexFlatIP;

} // namespace faiss

namespace faiss {
namespace gpu {

class FlatIndex;

/// Wrapper around the GPU implementation that looks like
/// faiss::IndexFlatL2; copies over centroid data from a given
/// faiss::IndexFlat
class RaftIndexFlatL2 : public GpuIndexFlat {
   public:
    /// Construct from a pre-existing faiss::IndexFlatL2 instance, copying
    /// data over to the given GPU
    RaftIndexFlatL2(
            GpuResourcesProvider* provider,
            faiss::IndexFlatL2* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    RaftIndexFlatL2(
            std::shared_ptr<GpuResources> resources,
            faiss::IndexFlatL2* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Construct an empty instance that can be added to
    RaftIndexFlatL2(
            GpuResourcesProvider* provider,
            int dims,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    RaftIndexFlatL2(
            std::shared_ptr<GpuResources> resources,
            int dims,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(faiss::IndexFlat* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexFlat* index);
};

/// Wrapper around the GPU implementation that looks like
/// faiss::IndexFlatIP; copies over centroid data from a given
/// faiss::IndexFlat
class RaftIndexFlatIP : public GpuIndexFlat {
   public:
    /// Construct from a pre-existing faiss::IndexFlatIP instance, copying
    /// data over to the given GPU
    RaftIndexFlatIP(
            GpuResourcesProvider* provider,
            faiss::IndexFlatIP* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    RaftIndexFlatIP(
            std::shared_ptr<GpuResources> resources,
            faiss::IndexFlatIP* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Construct an empty instance that can be added to
    RaftIndexFlatIP(
            GpuResourcesProvider* provider,
            int dims,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    RaftIndexFlatIP(
            std::shared_ptr<GpuResources> resources,
            int dims,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(faiss::IndexFlat* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexFlat* index);
};

} // namespace gpu
} // namespace faiss
