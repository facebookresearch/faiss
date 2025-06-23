// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsIVFPQ.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/utils/Transpose.cuh>

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/linalg/map.cuh>

#include <limits>
#include <memory>

namespace faiss {
namespace gpu {

CuvsIVFPQ::CuvsIVFPQ(
        GpuResources* resources,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool useFloat16LookupTables,
        bool useMMCodeDistance,
        bool interleavedLayout,
        float* pqCentroidData,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : IVFPQ(resources,
                dim,
                nlist,
                metric,
                metricArg,
                numSubQuantizers,
                bitsPerSubQuantizer,
                useFloat16LookupTables,
                useMMCodeDistance,
                interleavedLayout,
                // skip ptr allocations in base class (handled by cuVS
                // internally) false,
                pqCentroidData,
                indicesOptions,
                space) {
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS index");
}

CuvsIVFPQ::~CuvsIVFPQ() {}

void CuvsIVFPQ::reserveMemory(idx_t numVecs) {
    fprintf(stderr,
            "WARN: reserveMemory is NOP. Pre-allocation of IVF lists is not supported with cuVS enabled.\n");
}

void CuvsIVFPQ::reset() {
    cuvs_index.reset();
}

size_t CuvsIVFPQ::reclaimMemory() {
    fprintf(stderr,
            "WARN: reclaimMemory is NOP. reclaimMemory is not supported with cuVS enabled.\n");
    return 0;
}

void CuvsIVFPQ::setPrecomputedCodes(Index* quantizer, bool enable) {}

idx_t CuvsIVFPQ::getListLength(idx_t listId) const {
    FAISS_ASSERT(cuvs_index);
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

void CuvsIVFPQ::updateQuantizer(Index* quantizer) {
    FAISS_THROW_IF_NOT(quantizer->is_trained);

    // Must match our basic IVF parameters
    FAISS_THROW_IF_NOT(quantizer->d == getDim());
    FAISS_THROW_IF_NOT(quantizer->ntotal == getNumLists());

    auto stream = resources_->getDefaultStreamCurrentDevice();
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    cuvs::neighbors::ivf_pq::index_params pams;
    pams.metric = metricFaissToCuvs(metric_, false);
    pams.codebook_kind = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
    pams.n_lists = numLists_;
    pams.pq_bits = bitsPerSubQuantizer_;
    pams.pq_dim = numSubQuantizers_;
    cuvs_index = std::make_shared<cuvs::neighbors::ivf_pq::index<idx_t>>(
            raft_handle, pams, static_cast<uint32_t>(dim_));

    cuvs::neighbors::ivf_pq::helpers::reset_index(
            raft_handle, cuvs_index.get());
    cuvs::neighbors::ivf_pq::helpers::make_rotation_matrix(
            raft_handle, cuvs_index.get(), false);

    // If the index instance is a GpuIndexFlat, then we can use direct access to
    // the centroids within.
    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);

    if (gpuQ) {
        auto gpuData = gpuQ->getGpuData();

        if (gpuData->getUseFloat16()) {
            DeviceTensor<float, 2, true> centroids(
                    resources_,
                    makeSpaceAlloc(AllocType::FlatData, space_, stream),
                    {getNumLists(), getDim()});

            // The FlatIndex keeps its data in float16; we need to reconstruct
            // as float32 and store locally
            gpuData->reconstruct(0, gpuData->getSize(), centroids);

            cuvs::neighbors::ivf_pq::helpers::set_centers(
                    raft_handle,
                    cuvs_index.get(),
                    raft::make_device_matrix_view<float, uint32_t>(
                            centroids.data(), numLists_, dim_));
        } else {
            /// No reconstruct needed since the centers are already in float32
            // The FlatIndex keeps its data in float32, so we can merely
            // reference it
            auto centroids = gpuData->getVectorsFloat32Ref();

            cuvs::neighbors::ivf_pq::helpers::set_centers(
                    raft_handle,
                    cuvs_index.get(),
                    raft::make_device_matrix_view<float, uint32_t>(
                            centroids.data(), numLists_, dim_));
        }
    } else {
        DeviceTensor<float, 2, true> centroids(
                resources_,
                makeSpaceAlloc(AllocType::FlatData, space_, stream),
                {getNumLists(), getDim()});

        // Otherwise, we need to reconstruct all vectors from the index and copy
        // them to the GPU, in order to have access as needed for residual
        // computation
        auto vecs = std::vector<float>(getNumLists() * getDim());
        quantizer->reconstruct_n(0, quantizer->ntotal, vecs.data());

        centroids.copyFrom(vecs, stream);

        cuvs::neighbors::ivf_pq::helpers::set_centers(
                raft_handle,
                cuvs_index.get(),
                raft::make_device_matrix_view<float, uint32_t>(
                        centroids.data(), numLists_, dim_));
    }

    setPQCentroids_();
}

/// Return the list indices of a particular list back to the CPU
std::vector<idx_t> CuvsIVFPQ::getListIndices(idx_t listId) const {
    FAISS_ASSERT(cuvs_index);
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

/// Performs search when we are already given the IVF cells to look at
/// (GpuIndexIVF::search_preassigned implementation)
void CuvsIVFPQ::searchPreassigned(
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

size_t CuvsIVFPQ::getGpuListEncodingSize_(idx_t listId) {
    return static_cast<size_t>(cuvs_index->get_list_size_in_bytes(listId));
}

/// Return the encoded vectors of a particular list back to the CPU
std::vector<uint8_t> CuvsIVFPQ::getListVectorData(idx_t listId, bool gpuFormat)
        const {
    if (gpuFormat) {
        FAISS_THROW_MSG(
                "gpuFormat should be false for cuVS indices. Unpacked codes are flat.");
    }
    FAISS_ASSERT(cuvs_index);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    idx_t listSize = getListLength(listId);

    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(listSize);

    std::vector<uint8_t> flat_codes(
            cpuListSizeInBytes, static_cast<uint8_t>(0));

    idx_t maxBatchSize = 65536;
    for (idx_t offset_b = 0; offset_b < listSize; offset_b += maxBatchSize) {
        uint32_t batchSize = min(maxBatchSize, listSize - offset_b);
        uint32_t bufferSize = getCpuVectorsEncodingSize_(batchSize);
        uint32_t codesOffset = getCpuVectorsEncodingSize_(offset_b);

        // Fetch flat PQ codes for the current batch
        auto codes_d = raft::make_device_vector<uint8_t>(
                raft_handle, static_cast<uint32_t>(bufferSize));

        cuvs::neighbors::ivf_pq::helpers::codepacker::
                unpack_contiguous_list_data(
                        raft_handle,
                        *cuvs_index,
                        codes_d.data_handle(),
                        batchSize,
                        listId,
                        offset_b);

        // Copy the flat PQ codes to host
        raft::update_host(
                flat_codes.data() + codesOffset,
                codes_d.data_handle(),
                bufferSize,
                stream);
        raft_handle.sync_stream();
    }

    return flat_codes;
}

/// Find the approximate k nearest neighbors for `queries` against
/// our database
void CuvsIVFPQ::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    uint32_t numQueries = queries.getSize(0);
    uint32_t cols = queries.getSize(1);
    idx_t k_ = std::min(static_cast<idx_t>(k), cuvs_index->size());

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(cuvs_index);
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    cuvs::neighbors::ivf_pq::search_params pams;
    pams.n_probes = nprobe;
    pams.lut_dtype = useFloat16LookupTables_ ? CUDA_R_16F : CUDA_R_32F;

    auto queries_view = raft::make_device_matrix_view<const float, idx_t>(
            queries.data(), (idx_t)numQueries, (idx_t)cols);
    auto out_inds_view = raft::make_device_matrix_view<idx_t, idx_t>(
            outIndices.data(), (idx_t)numQueries, (idx_t)k_);
    auto out_dists_view = raft::make_device_matrix_view<float, idx_t>(
            outDistances.data(), (idx_t)numQueries, (idx_t)k_);

    cuvs::neighbors::ivf_pq::search(
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
    raft_handle.sync_stream();
}

idx_t CuvsIVFPQ::addVectors(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
    /// NB: The coarse quantizer is ignored here. The user is assumed to have
    /// called updateQuantizer() to update the cuVS index if the quantizer was
    /// modified externally

    FAISS_ASSERT(cuvs_index);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    /// Remove rows containing NaNs
    idx_t n_rows_valid = inplaceGatherFilteredRows(resources_, vecs, indices);

    cuvs::neighbors::ivf_pq::extend(
            raft_handle,
            raft::make_device_matrix_view<const float, idx_t>(
                    vecs.data(), n_rows_valid, dim_),
            raft::make_device_vector_view<const idx_t, idx_t>(
                    indices.data(), n_rows_valid),
            cuvs_index.get());

    return n_rows_valid;
}

void CuvsIVFPQ::copyInvertedListsFrom(const InvertedLists* ivf) {
    size_t nlist = ivf ? ivf->nlist : 0;
    size_t ntotal = ivf ? ivf->compute_ntotal() : 0;

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    std::vector<uint32_t> list_sizes_(nlist);
    std::vector<idx_t> indices_(ntotal);

    // the index must already exist
    FAISS_ASSERT(cuvs_index);

    auto& cuvs_index_lists = cuvs_index->lists();

    // conservative memory alloc for cloning cpu inverted lists
    cuvs::neighbors::ivf_pq::list_spec<uint32_t, idx_t> ivf_list_spec{
            static_cast<uint32_t>(bitsPerSubQuantizer_),
            static_cast<uint32_t>(numSubQuantizers_),
            true};

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
                static_cast<uint32_t>(listSize),
                static_cast<uint32_t>(0));
    }

    raft::update_device(
            cuvs_index->list_sizes().data_handle(),
            list_sizes_.data(),
            nlist,
            raft_handle.get_stream());

    //     Update the pointers and the sizes
    cuvs::neighbors::ivf_pq::helpers::recompute_internal_state(
            raft_handle, cuvs_index.get());

    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        addEncodedVectorsToList_(
                i, ivf->get_codes(i), ivf->get_ids(i), listSize);
    }
}

void CuvsIVFPQ::setCuvsIndex(cuvs::neighbors::ivf_pq::index<idx_t>&& idx) {
    cuvs_index = std::make_shared<cuvs::neighbors::ivf_pq::index<idx_t>>(
            std::move(idx));
    setBasePQCentroids_();
}

void CuvsIVFPQ::addEncodedVectorsToList_(
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    // If there's nothing to add, then there's nothing we have to do
    if (numVecs == 0) {
        return;
    }

    // The GPU might have a different layout of the memory
    auto gpuListSizeInBytes = getGpuListEncodingSize_(listId);

    // We only have int32 length representations on the GPU per each
    // list; the length is in sizeof(char)
    FAISS_ASSERT(gpuListSizeInBytes <= (size_t)std::numeric_limits<int>::max());

    idx_t maxBatchSize = 4096;
    for (idx_t offset_b = 0; offset_b < numVecs; offset_b += maxBatchSize) {
        uint32_t batchSize = min(maxBatchSize, numVecs - offset_b);
        uint32_t bufferSize = getCpuVectorsEncodingSize_(batchSize);
        uint32_t codesOffset = getCpuVectorsEncodingSize_(offset_b);

        // Translate the codes as needed to our preferred form
        auto codes_d = raft::make_device_vector<uint8_t>(
                raft_handle, static_cast<uint32_t>(bufferSize));
        raft::update_device(
                codes_d.data_handle(),
                static_cast<const uint8_t*>(codes) + codesOffset,
                bufferSize,
                stream);

        cuvs::neighbors::ivf_pq::helpers::codepacker::pack_contiguous_list_data(
                raft_handle,
                cuvs_index.get(),
                codes_d.data_handle(),
                batchSize,
                listId,
                offset_b);
    }

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

void CuvsIVFPQ::setPQCentroids_() {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    raft::copy(
            cuvs_index->pq_centers().data_handle(),
            pqCentroidsInnermostCode_.data(),
            pqCentroidsInnermostCode_.numElements(),
            stream);
}

void CuvsIVFPQ::setBasePQCentroids_() {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    raft::copy(
            pqCentroidsInnermostCode_.data(),
            cuvs_index->pq_centers().data_handle(),
            cuvs_index->pq_centers().size(),
            stream);

    DeviceTensor<float, 3, true> pqCentroidsMiddleCode(
            resources_,
            makeDevAlloc(AllocType::Quantizer, stream),
            {numSubQuantizers_, numSubQuantizerCodes_, dimPerSubQuantizer_});

    runTransposeAny(
            pqCentroidsInnermostCode_, 1, 2, pqCentroidsMiddleCode, stream);

    pqCentroidsMiddleCode_ = std::move(pqCentroidsMiddleCode);
}

} // namespace gpu
} // namespace faiss
