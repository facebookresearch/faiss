/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/clone_index.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndicesOptions.h>

namespace faiss {
namespace gpu {

class GpuResourcesProvider;

/// Cloner specialized for GPU -> CPU
struct ToCPUCloner : faiss::Cloner {
    void merge_index(Index* dst, Index* src, bool successive_ids);
    Index* clone_Index(const Index* index) override;
};

/// Cloner specialized for CPU -> 1 GPU
struct ToGpuCloner : faiss::Cloner, GpuClonerOptions {
    GpuResourcesProvider* provider;
    int device;

    ToGpuCloner(
            GpuResourcesProvider* prov,
            int device,
            const GpuClonerOptions& options);

    Index* clone_Index(const Index* index) override;
};

/// Cloner specialized for CPU -> multiple GPUs
struct ToGpuClonerMultiple : faiss::Cloner, GpuMultipleClonerOptions {
    std::vector<ToGpuCloner> sub_cloners;

    ToGpuClonerMultiple(
            std::vector<GpuResourcesProvider*>& provider,
            std::vector<int>& devices,
            const GpuMultipleClonerOptions& options);

    ToGpuClonerMultiple(
            const std::vector<ToGpuCloner>& sub_cloners,
            const GpuMultipleClonerOptions& options);

    void copy_ivf_shard(
            const IndexIVF* index_ivf,
            IndexIVF* idx2,
            idx_t n,
            idx_t i);

    Index* clone_Index_to_shards(const Index* index);

    /// main function
    Index* clone_Index(const Index* index) override;
};

/// converts any GPU index inside gpu_index to a CPU index
faiss::Index* index_gpu_to_cpu(const faiss::Index* gpu_index);

/// converts any CPU index that can be converted to GPU
faiss::Index* index_cpu_to_gpu(
        GpuResourcesProvider* provider,
        int device,
        const faiss::Index* index,
        const GpuClonerOptions* options = nullptr);

faiss::Index* index_cpu_to_gpu_multiple(
        std::vector<GpuResourcesProvider*>& provider,
        std::vector<int>& devices,
        const faiss::Index* index,
        const GpuMultipleClonerOptions* options = nullptr);

/// index factory for the ProgressiveDimClustering object

struct GpuProgressiveDimIndexFactory : ProgressiveDimIndexFactory {
    GpuMultipleClonerOptions options;
    std::vector<GpuResourcesProvider*> vres;
    std::vector<int> devices;
    int ncall;

    explicit GpuProgressiveDimIndexFactory(int ngpu);

    Index* operator()(int dim) override;

    virtual ~GpuProgressiveDimIndexFactory() override;
};

/*********************************************
 * Cloning binary indexes
 *********************************************/

faiss::IndexBinary* index_binary_gpu_to_cpu(
        const faiss::IndexBinary* gpu_index);

/// converts any CPU index that can be converted to GPU
faiss::IndexBinary* index_binary_cpu_to_gpu(
        GpuResourcesProvider* provider,
        int device,
        const faiss::IndexBinary* index,
        const GpuClonerOptions* options = nullptr);

faiss::IndexBinary* index_binary_cpu_to_gpu_multiple(
        std::vector<GpuResourcesProvider*>& provider,
        std::vector<int>& devices,
        const faiss::IndexBinary* index,
        const GpuMultipleClonerOptions* options = nullptr);

} // namespace gpu
} // namespace faiss
