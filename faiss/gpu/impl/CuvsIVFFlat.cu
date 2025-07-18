// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsIVFFlat.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/utils/Transpose.cuh>

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>

#include <limits>
#include <memory>

namespace faiss {
namespace gpu {

CuvsIVFFlat::CuvsIVFFlat(
        GpuResources* res,
        int dim,
        int nlist,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        faiss::ScalarQuantizer* scalarQ,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IVFFlat(res,
                  dim,
                  nlist,
                  metric,
                  metricArg,
                  useResidual,
                  scalarQ,
                  interleavedLayout,
                  // skip ptr allocations in base class (handled by cuVS
                  // internally)
                  indicesOptions,
                  space) {
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS index");
}

CuvsIVFFlat::~CuvsIVFFlat() {}

void CuvsIVFFlat::reserveMemory(idx_t numVecs) {
    fprintf(stderr,
            "WARN: reserveMemory is NOP. Pre-allocation of IVF lists is not supported with cuVS enabled.\n");
}

void CuvsIVFFlat::reset() {
    cuvs_index.reset();
}

void CuvsIVFFlat::setCuvsIndex(
        cuvs::neighbors::ivf_flat::index<float, idx_t>&& idx) {
    cuvs_index =
            std::make_shared<cuvs::neighbors::ivf_flat::index<float, idx_t>>(
                    std::move(idx));
}

void CuvsIVFFlat::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    /// NB: The coarse quantizer is ignored here. The user is assumed to have
    /// called updateQuantizer() to modify the cuVS index if the quantizer was
    /// modified externally

    uint32_t numQueries = queries.getSize(0);
    uint32_t cols = queries.getSize(1);
    uint32_t k_ = k;

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(cuvs_index != nullptr);
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    cuvs::neighbors::ivf_flat::search_params pams;
    pams.n_probes = nprobe;

    auto queries_view = raft::make_device_matrix_view<const float, idx_t>(
            queries.data(), (idx_t)numQueries, (idx_t)cols);
    auto out_inds_view = raft::make_device_matrix_view<idx_t, idx_t>(
            outIndices.data(), (idx_t)numQueries, (idx_t)k_);
    auto out_dists_view = raft::make_device_matrix_view<float, idx_t>(
            outDistances.data(), (idx_t)numQueries, (idx_t)k_);

    cuvs::neighbors::ivf_flat::search(
            raft_handle,
            pams,
            *cuvs_index,
            queries_view,
            out_inds_view,
            out_dists_view);

    /// Identify NaN rows and mask their nearest neighbors
    auto nan_flag = raft::make_device_vector<bool>(raft_handle, numQueries);

    validRowIndices(resources_, queries, nan_flag.data_handle());

    raft::linalg::map_offset(
            raft_handle,
            raft::make_device_vector_view(outIndices.data(), numQueries * k_),
            [nan_flag = nan_flag.data_handle(),
             out_inds = outIndices.data(),
             k_] __device__(uint32_t i) {
                uint32_t row = i / k_;
                if (!nan_flag[row])
                    return idx_t(-1);
                return out_inds[i];
            });

    float max_val = std::numeric_limits<float>::max();
    raft::linalg::map_offset(
            raft_handle,
            raft::make_device_vector_view(outDistances.data(), numQueries * k_),
            [nan_flag = nan_flag.data_handle(),
             out_dists = outDistances.data(),
             max_val,
             k_] __device__(uint32_t i) {
                uint32_t row = i / k_;
                if (!nan_flag[row])
                    return max_val;
                return out_dists[i];
            });
}

idx_t CuvsIVFFlat::addVectors(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
    /// NB: The coarse quantizer is ignored here. The user is assumed to have
    /// called updateQuantizer() to update the cuVS index if the quantizer was
    /// modified externally

    FAISS_ASSERT(cuvs_index != nullptr);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    /// Remove rows containing NaNs
    idx_t n_rows_valid = inplaceGatherFilteredRows(resources_, vecs, indices);

    cuvs::neighbors::ivf_flat::extend(
            raft_handle,
            raft::make_device_matrix_view<const float, idx_t>(
                    vecs.data(), n_rows_valid, dim_),
            std::make_optional<raft::device_vector_view<const idx_t, idx_t>>(
                    raft::make_device_vector_view<const idx_t, idx_t>(
                            indices.data(), n_rows_valid)),
            cuvs_index.get());

    return n_rows_valid;
}

idx_t CuvsIVFFlat::getListLength(idx_t listId) const {
    FAISS_ASSERT(cuvs_index != nullptr);
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    uint32_t size;
    raft::update_host(
            &size,
            cuvs_index->list_sizes().data_handle() + listId,
            1,
            raft_handle.get_stream());
    raft_handle.sync_stream();

    return static_cast<int>(size);
}

/// Return the list indices of a particular list back to the CPU
std::vector<idx_t> CuvsIVFFlat::getListIndices(idx_t listId) const {
    FAISS_ASSERT(cuvs_index != nullptr);
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    idx_t listSize = getListLength(listId);

    std::vector<idx_t> vec(listSize);

    // fetch the list indices ptr on host
    idx_t* list_indices_ptr;

    raft::update_host(
            &list_indices_ptr,
            const_cast<idx_t**>(cuvs_index->inds_ptrs().data_handle()) + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_host(vec.data(), list_indices_ptr, listSize, stream);
    raft_handle.sync_stream();

    return vec;
}

/// Return the encoded vectors of a particular list back to the CPU
std::vector<uint8_t> CuvsIVFFlat::getListVectorData(
        idx_t listId,
        bool gpuFormat) const {
    if (gpuFormat) {
        FAISS_THROW_MSG("gpuFormat should be false for cuVS indices");
    }
    FAISS_ASSERT(cuvs_index != nullptr);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    idx_t listSize = getListLength(listId);

    // the interleaved block can be slightly larger than the list size (it's
    // rounded up)
    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(listSize);
    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(listSize);

    std::vector<uint8_t> interleaved_codes(gpuListSizeInBytes);
    std::vector<uint8_t> flat_codes(cpuListSizeInBytes);

    float* list_data_ptr;

    // fetch the list data ptr on host
    raft::update_host(
            &list_data_ptr,
            cuvs_index->data_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_host(
            interleaved_codes.data(),
            reinterpret_cast<uint8_t*>(list_data_ptr),
            gpuListSizeInBytes,
            stream);
    raft_handle.sync_stream();

    CuvsIVFFlatCodePackerInterleaved packer(
            (size_t)listSize, dim_, cuvs_index->veclen());
    packer.unpack_all(interleaved_codes.data(), flat_codes.data());
    return flat_codes;
}

/// Performs search when we are already given the IVF cells to look at
/// (GpuIndexIVF::search_preassigned implementation)
void CuvsIVFFlat::searchPreassigned(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& ivfDistances,
        Tensor<idx_t, 2, true>& ivfAssignments,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool storePairs) {
    // TODO: Fill this in!
    // Reference issue: https://github.com/facebookresearch/faiss/issues/3243
    FAISS_THROW_MSG("searchPreassigned is not implemented for cuVS index");
}

void CuvsIVFFlat::updateQuantizer(Index* quantizer) {
    FAISS_THROW_IF_NOT(quantizer->is_trained);

    // Must match our basic IVF parameters
    FAISS_THROW_IF_NOT(quantizer->d == getDim());
    FAISS_THROW_IF_NOT(quantizer->ntotal == getNumLists());

    size_t total_elems = quantizer->ntotal * quantizer->d;

    auto stream = resources_->getDefaultStreamCurrentDevice();
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    cuvs::neighbors::ivf_flat::index_params pams;
    pams.add_data_on_build = false;
    pams.metric = metricFaissToCuvs(metric_, false);
    pams.n_lists = numLists_;
    cuvs_index =
            std::make_shared<cuvs::neighbors::ivf_flat::index<float, idx_t>>(
                    raft_handle, pams, static_cast<uint32_t>(dim_));
    cuvs::neighbors::ivf_flat::helpers::reset_index(
            raft_handle, cuvs_index.get());

    // If the index instance is a GpuIndexFlat, then we can use direct access to
    // the centroids within.
    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
    if (gpuQ) {
        auto gpuData = gpuQ->getGpuData();

        if (gpuData->getUseFloat16()) {
            // The FlatIndex keeps its data in float16; we need to reconstruct
            // as float32 and store locally
            DeviceTensor<float, 2, true> centroids(
                    resources_,
                    makeSpaceAlloc(AllocType::FlatData, space_, stream),
                    {getNumLists(), getDim()});

            gpuData->reconstruct(0, gpuData->getSize(), centroids);

            raft::update_device(
                    cuvs_index->centers().data_handle(),
                    centroids.data(),
                    total_elems,
                    stream);
        } else {
            /// No reconstruct needed since the centers are already in float32
            auto centroids = gpuData->getVectorsFloat32Ref();

            raft::update_device(
                    cuvs_index->centers().data_handle(),
                    centroids.data(),
                    total_elems,
                    stream);
        }
    } else {
        // Otherwise, we need to reconstruct all vectors from the index and copy
        // them to the GPU, in order to have access as needed for residual
        // computation
        auto vecs = std::vector<float>(getNumLists() * getDim());
        quantizer->reconstruct_n(0, quantizer->ntotal, vecs.data());

        raft::update_device(
                cuvs_index->centers().data_handle(),
                vecs.data(),
                total_elems,
                stream);
    }
}

void CuvsIVFFlat::copyInvertedListsFrom(const InvertedLists* ivf) {
    size_t nlist = ivf ? ivf->nlist : 0;
    size_t ntotal = ivf ? ivf->compute_ntotal() : 0;

    raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    std::vector<uint32_t> list_sizes_(nlist);
    std::vector<idx_t> indices_(ntotal);

    // the index must already exist
    FAISS_ASSERT(cuvs_index != nullptr);

    auto& cuvs_index_lists = cuvs_index->lists();

    // conservative memory alloc for cloning cpu inverted lists
    cuvs::neighbors::ivf_flat::list_spec<uint32_t, float, idx_t> ivf_list_spec{
            static_cast<uint32_t>(dim_), true};

    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);

        // GPU index can only support max int entries per list
        FAISS_THROW_IF_NOT_FMT(
                listSize <= (size_t)std::numeric_limits<int>::max(),
                "GPU inverted list can only support "
                "%zu entries; %zu found",
                (size_t)std::numeric_limits<int>::max(),
                listSize);

        // store the list size
        list_sizes_[i] = static_cast<uint32_t>(listSize);

        // This cuVS list must currently be empty
        FAISS_ASSERT(getListLength(i) == 0);

        cuvs::neighbors::ivf::resize_list(
                raft_handle,
                cuvs_index_lists[i],
                ivf_list_spec,
                (uint32_t)listSize,
                (uint32_t)0);
    }

    // Update the pointers and the sizes
    cuvs::neighbors::ivf_flat::helpers::recompute_internal_state(
            raft_handle, cuvs_index.get());

    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        addEncodedVectorsToList_(
                i, ivf->get_codes(i), ivf->get_ids(i), listSize);
    }

    raft::update_device(
            cuvs_index->list_sizes().data_handle(),
            list_sizes_.data(),
            nlist,
            raft_handle.get_stream());

    // Precompute the centers vector norms for L2Expanded distance
    if (this->metric_ == faiss::METRIC_L2) {
        cuvs_index->allocate_center_norms(raft_handle);
        raft::linalg::rowNorm<raft::linalg::L2Norm, true, float, uint32_t>(
                cuvs_index->center_norms().value().data_handle(),
                cuvs_index->centers().data_handle(),
                cuvs_index->dim(),
                (uint32_t)nlist,
                raft_handle.get_stream());
    }
}

size_t CuvsIVFFlat::getGpuVectorsEncodingSize_(idx_t numVecs) const {
    idx_t bits = 32 /* float */;

    // bytes to encode a block of 32 vectors (single dimension)
    idx_t bytesPerDimBlock = bits * 32 / 8; // = 128

    // bytes to fully encode 32 vectors
    idx_t bytesPerBlock = bytesPerDimBlock * dim_;

    // number of blocks of 32 vectors we have
    idx_t numBlocks =
            utils::divUp(numVecs, cuvs::neighbors::ivf_flat::kIndexGroupSize);

    // total size to encode numVecs
    return bytesPerBlock * numBlocks;
}

void CuvsIVFFlat::addEncodedVectorsToList_(
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // If there's nothing to add, then there's nothing we have to do
    if (numVecs == 0) {
        return;
    }

    // The GPU might have a different layout of the memory
    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);

    // We only have int32 length representations on the GPU per each
    // list; the length is in sizeof(char)
    FAISS_ASSERT(gpuListSizeInBytes <= (size_t)std::numeric_limits<int>::max());

    std::vector<uint8_t> interleaved_codes(gpuListSizeInBytes);
    CuvsIVFFlatCodePackerInterleaved packer(
            (size_t)numVecs, (uint32_t)dim_, cuvs_index->veclen());

    packer.pack_all(
            reinterpret_cast<const uint8_t*>(codes), interleaved_codes.data());

    float* list_data_ptr;
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    /// fetch the list data ptr on host
    raft::update_host(
            &list_data_ptr,
            cuvs_index->data_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_device(
            reinterpret_cast<uint8_t*>(list_data_ptr),
            interleaved_codes.data(),
            gpuListSizeInBytes,
            stream);

    /// Handle the indices as well
    idx_t* list_indices_ptr;

    // fetch the list indices ptr on host
    raft::update_host(
            &list_indices_ptr,
            cuvs_index->inds_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_device(list_indices_ptr, indices, numVecs, stream);
}

CuvsIVFFlatCodePackerInterleaved::CuvsIVFFlatCodePackerInterleaved(
        size_t list_size,
        uint32_t dim,
        uint32_t chunk_size) {
    this->dim = dim;
    this->chunk_size = chunk_size;
    // NB: dim should be divisible by the number of 4 byte records in one chunk
    FAISS_ASSERT(dim % chunk_size == 0);
    nvec = list_size;
    code_size = dim * 4;
    block_size =
            utils::roundUp(nvec, cuvs::neighbors::ivf_flat::kIndexGroupSize);
}

void CuvsIVFFlatCodePackerInterleaved::pack_1(
        const uint8_t* flat_code,
        size_t offset,
        uint8_t* block) const {
    cuvs::neighbors::ivf_flat::helpers::codepacker::pack_1(
            reinterpret_cast<const float*>(flat_code),
            reinterpret_cast<float*>(block),
            dim,
            chunk_size,
            static_cast<uint32_t>(offset));
}

void CuvsIVFFlatCodePackerInterleaved::unpack_1(
        const uint8_t* block,
        size_t offset,
        uint8_t* flat_code) const {
    cuvs::neighbors::ivf_flat::helpers::codepacker::unpack_1(
            reinterpret_cast<const float*>(block),
            reinterpret_cast<float*>(flat_code),
            dim,
            chunk_size,
            static_cast<uint32_t>(offset));
}

} // namespace gpu
} // namespace faiss
