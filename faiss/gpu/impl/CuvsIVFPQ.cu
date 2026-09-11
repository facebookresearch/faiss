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
#include <faiss/gpu/utils/CuvsFilterConvert.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsIVFPQ.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/utils/Transpose.cuh>

#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>

#include <limits>
#include <memory>

namespace faiss {
namespace gpu {

// Compatibility note: release/26.10 has no C API for resetting,
// importing/exporting, resizing, recomputing, or packing raw IVF-PQ list
// storage. Public build, search, extend, and getter operations use C except for
// the selector-only search fallback documented below.
using CuvsIVFPQCppIndex = cuvs::neighbors::ivf_pq::index<idx_t>;

static CuvsIVFPQCppIndex* getCppIndex(cuvsIvfPqIndex_t index) {
    return reinterpret_cast<CuvsIVFPQCppIndex*>(index->addr);
}

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

CuvsIVFPQ::~CuvsIVFPQ() {
    if (cuvs_index) {
        cuvsIvfPqIndexDestroy(cuvs_index);
    }
}

void CuvsIVFPQ::reserveMemory(idx_t numVecs) {
    fprintf(stderr,
            "WARN: reserveMemory is NOP. Pre-allocation of IVF lists is not supported with cuVS enabled.\n");
}

void CuvsIVFPQ::reset() {
    if (cuvs_index) {
        const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
        cuvs::neighbors::ivf_pq::helpers::reset_index(
                raftHandle, getCppIndex(cuvs_index));
    }
}

size_t CuvsIVFPQ::reclaimMemory() {
    fprintf(stderr,
            "WARN: reclaimMemory is NOP. reclaimMemory is not supported with cuVS enabled.\n");
    return 0;
}

void CuvsIVFPQ::setPrecomputedCodes(Index* quantizer, bool enable) {}

idx_t CuvsIVFPQ::getListLength(idx_t listId) const {
    FAISS_ASSERT(cuvs_index);
    const raft::device_resources& raftHandle =
            resources_->getRaftHandleCurrentDevice();
    CuvsOutputTensor listSizes;
    cuvsCheck(
            cuvsIvfPqIndexGetListSizes(cuvs_index, listSizes.get()),
            "cuvsIvfPqIndexGetListSizes");

    uint32_t size;
    raft::update_host(
            &size,
            listSizes.data<uint32_t>() + listId,
            1,
            raftHandle.get_stream());
    raftHandle.sync_stream();

    return static_cast<int>(size);
}

void CuvsIVFPQ::updateQuantizer(Index* quantizer) {
    FAISS_THROW_IF_NOT(quantizer->is_trained);

    // Must match our basic IVF parameters
    FAISS_THROW_IF_NOT(quantizer->d == getDim());
    FAISS_THROW_IF_NOT(quantizer->ntotal == getNumLists());
    auto stream = resources_->getDefaultStreamCurrentDevice();
    const raft::device_resources& raftHandle =
            resources_->getRaftHandleCurrentDevice();
    const uint32_t dimExt = utils::roundUp((uint32_t)dim_ + 1, 8u);

    cuvsPaddedCenters_ = DeviceTensor<float, 2, true>(
            resources_,
            makeSpaceAlloc(AllocType::Quantizer, space_, stream),
            {numLists_, dimExt});
    cuvsRotatedCenters_ = DeviceTensor<float, 2, true>(
            resources_,
            makeSpaceAlloc(AllocType::Quantizer, space_, stream),
            {numLists_, dim_});
    cuvsRotationMatrix_ = DeviceTensor<float, 2, true>(
            resources_,
            makeSpaceAlloc(AllocType::Quantizer, space_, stream),
            {dim_, dim_});

    auto prepareCenters = [&](const float* centers) {
        raft::update_device(
                cuvsRotatedCenters_.data(),
                centers,
                (size_t)numLists_ * dim_,
                stream);
        cuvsPaddedCenters_.zero(stream);
        raft::copy_matrix(
                cuvsPaddedCenters_.data(),
                dimExt,
                cuvsRotatedCenters_.data(),
                dim_,
                dim_,
                numLists_,
                stream);

        auto centerNorms = raft::make_device_vector<float, uint32_t>(
                raftHandle, numLists_);
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
                raftHandle,
                raft::make_device_matrix_view<const float, uint32_t>(
                        cuvsRotatedCenters_.data(), numLists_, dim_),
                centerNorms.view());
        raft::copy_matrix(
                cuvsPaddedCenters_.data() + dim_,
                dimExt,
                centerNorms.data_handle(),
                1,
                1,
                numLists_,
                stream);

        raft::linalg::map_offset(
                raftHandle,
                raft::make_device_vector_view(
                        cuvsRotationMatrix_.data(), (uint32_t)(dim_ * dim_)),
                [stride = (uint32_t)dim_ + 1] __device__(uint32_t i) {
                    return static_cast<float>(i % stride == 0);
                });
    };

    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
    if (gpuQ) {
        auto gpuData = gpuQ->getGpuData();
        if (gpuData->getUseFloat16()) {
            DeviceTensor<float, 2, true> centroids(
                    resources_,
                    makeSpaceAlloc(AllocType::FlatData, space_, stream),
                    {getNumLists(), getDim()});
            gpuData->reconstruct(0, gpuData->getSize(), centroids);
            prepareCenters(centroids.data());
        } else {
            auto centroids = gpuData->getVectorsFloat32Ref();
            prepareCenters(centroids.data());
        }
    } else {
        auto centroids = std::vector<float>(getNumLists() * getDim());
        quantizer->reconstruct_n(0, quantizer->ntotal, centroids.data());
        prepareCenters(centroids.data());
    }

    cuvsIvfPqIndexParams_t params = nullptr;
    cuvsCheck(
            cuvsIvfPqIndexParamsCreate(&params), "cuvsIvfPqIndexParamsCreate");
    CuvsUniquePtr<cuvsIvfPqIndexParams, cuvsIvfPqIndexParamsDestroy>
            paramsHolder(params);
    params->metric = metricFaissToCuvs(metric_, false);
    params->metric_arg = metricArg_;
    params->add_data_on_build = false;
    params->n_lists = numLists_;
    params->pq_bits = bitsPerSubQuantizer_;
    params->pq_dim = numSubQuantizers_;
    params->codebook_kind = CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE;
    params->force_random_rotation = false;
    params->codes_layout = CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED;

    auto pqCentersTensor = makeCuvsTensor(
            pqCentroidsInnermostCode_.data(),
            (int64_t)numSubQuantizers_,
            (int64_t)dimPerSubQuantizer_,
            (int64_t)numSubQuantizerCodes_);
    auto paddedCentersTensor = makeCuvsTensor(
            cuvsPaddedCenters_.data(), (int64_t)numLists_, (int64_t)dimExt);
    auto rotatedCentersTensor = makeCuvsTensor(
            cuvsRotatedCenters_.data(), (int64_t)numLists_, (int64_t)dim_);
    auto rotationTensor = makeCuvsTensor(
            cuvsRotationMatrix_.data(), (int64_t)dim_, (int64_t)dim_);

    cuvsIvfPqIndex_t index = nullptr;
    cuvsCheck(cuvsIvfPqIndexCreate(&index), "cuvsIvfPqIndexCreate");
    CuvsUniquePtr<cuvsIvfPqIndex, cuvsIvfPqIndexDestroy> indexHolder(index);
    cuvsCheck(
            cuvsIvfPqBuildPrecomputed(
                    cuvsResourcesFromGpuResources(resources_),
                    params,
                    dim_,
                    pqCentersTensor.get(),
                    paddedCentersTensor.get(),
                    rotatedCentersTensor.get(),
                    rotationTensor.get(),
                    index),
            "cuvsIvfPqBuildPrecomputed");

    if (cuvs_index) {
        cuvsCheck(cuvsIvfPqIndexDestroy(cuvs_index), "cuvsIvfPqIndexDestroy");
    }
    cuvs_index = indexHolder.release();
    setPQCentroids_();
}

/// Return the list indices of a particular list back to the CPU
std::vector<idx_t> CuvsIVFPQ::getListIndices(idx_t listId) const {
    FAISS_ASSERT(cuvs_index);
    const raft::device_resources& raftHandle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raftHandle.get_stream();
    CuvsOutputTensor listIndices;
    cuvsCheck(
            cuvsIvfPqIndexGetListIndices(
                    cuvs_index, (uint32_t)listId, listIndices.get()),
            "cuvsIvfPqIndexGetListIndices");

    idx_t listSize = getListLength(listId);
    std::vector<idx_t> vec(listSize);
    raft::update_host(vec.data(), listIndices.data<idx_t>(), listSize, stream);
    raftHandle.sync_stream();

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
    return static_cast<size_t>(
            getCppIndex(cuvs_index)->get_list_size_in_bytes(listId));
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

        auto codesTensor = makeCuvsTensor(
                codes_d.data_handle(),
                (int64_t)batchSize,
                (int64_t)(bufferSize / batchSize));
        cuvsCheck(
                cuvsIvfPqIndexUnpackContiguousListData(
                        cuvsResourcesFromGpuResources(resources_),
                        cuvs_index,
                        codesTensor.get(),
                        (uint32_t)listId,
                        (uint32_t)offset_b),
                "cuvsIvfPqIndexUnpackContiguousListData");

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
        Tensor<idx_t, 2, true>& outIndices,
        const IDSelector* sel) {
    FAISS_ASSERT(cuvs_index);
    uint32_t numQueries = queries.getSize(0);
    uint32_t cols = queries.getSize(1);
    int64_t indexSize = 0;
    cuvsCheck(
            cuvsIvfPqIndexGetSize(cuvs_index, &indexSize),
            "cuvsIvfPqIndexGetSize");
    idx_t k_ = std::min(static_cast<idx_t>(k), (idx_t)indexSize);

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    // Compatibility note: release/26.10's IVF-PQ C search has no filter
    // argument. Retain C++ search only when a Faiss IDSelector is supplied.
    if (sel) {
        cuvs::neighbors::ivf_pq::search_params searchParams;
        searchParams.n_probes = nprobe;
        searchParams.lut_dtype =
                useFloat16LookupTables_ ? CUDA_R_16F : CUDA_R_32F;

        auto queryView = raft::make_device_matrix_view<const float, idx_t>(
                queries.data(), (idx_t)numQueries, (idx_t)cols);
        auto indexView = raft::make_device_matrix_view<idx_t, idx_t>(
                outIndices.data(), (idx_t)numQueries, k_);
        auto distanceView = raft::make_device_matrix_view<float, idx_t>(
                outDistances.data(), (idx_t)numQueries, k_);
        raft::core::bitset<uint32_t, int64_t> bitset(
                raft_handle, indexSize, false);
        convert_to_bitset(resources_, *sel, bitset.view());
        cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(
                bitset.view());
        cuvs::neighbors::ivf_pq::search(
                raft_handle,
                searchParams,
                *getCppIndex(cuvs_index),
                queryView,
                indexView,
                distanceView,
                filter);
    } else {
        cuvsIvfPqSearchParams_t searchParams = nullptr;
        cuvsCheck(
                cuvsIvfPqSearchParamsCreate(&searchParams),
                "cuvsIvfPqSearchParamsCreate");
        CuvsUniquePtr<cuvsIvfPqSearchParams, cuvsIvfPqSearchParamsDestroy>
                searchParamsHolder(searchParams);
        searchParams->n_probes = nprobe;
        searchParams->lut_dtype =
                useFloat16LookupTables_ ? CUDA_R_16F : CUDA_R_32F;

        auto queriesTensor = makeCuvsTensor(
                queries.data(), (int64_t)numQueries, (int64_t)cols);
        auto indicesTensor = makeCuvsTensor(
                outIndices.data(), (int64_t)numQueries, (int64_t)k_);
        auto distancesTensor = makeCuvsTensor(
                outDistances.data(), (int64_t)numQueries, (int64_t)k_);
        cuvsCheck(
                cuvsIvfPqSearch(
                        cuvsResourcesFromGpuResources(resources_),
                        searchParams,
                        cuvs_index,
                        queriesTensor.get(),
                        indicesTensor.get(),
                        distancesTensor.get()),
                "cuvsIvfPqSearch");
    }

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

    /// Remove rows containing NaNs
    idx_t n_rows_valid = inplaceGatherFilteredRows(resources_, vecs, indices);

    auto vectorsTensor = makeCuvsTensor(vecs.data(), n_rows_valid, (idx_t)dim_);
    auto indicesTensor = makeCuvsTensor(indices.data(), n_rows_valid);
    cuvsCheck(
            cuvsIvfPqExtend(
                    cuvsResourcesFromGpuResources(resources_),
                    vectorsTensor.get(),
                    indicesTensor.get(),
                    cuvs_index),
            "cuvsIvfPqExtend");

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

    auto& cuvs_index_lists = getCppIndex(cuvs_index)->lists();

    // conservative memory alloc for cloning cpu inverted lists
    cuvs::neighbors::ivf_pq::list_spec_interleaved<uint32_t, idx_t>
            ivf_list_spec{
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

        cuvs::neighbors::ivf_pq::helpers::resize_list(
                raft_handle,
                cuvs_index_lists[i],
                ivf_list_spec,
                static_cast<uint32_t>(listSize),
                static_cast<uint32_t>(0));
    }

    raft::update_device(
            getCppIndex(cuvs_index)->list_sizes().data_handle(),
            list_sizes_.data(),
            nlist,
            raft_handle.get_stream());

    //     Update the pointers and the sizes
    cuvs::neighbors::ivf_pq::helpers::recompute_internal_state(
            raft_handle, getCppIndex(cuvs_index));

    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        addEncodedVectorsToList_(
                i, ivf->get_codes(i), ivf->get_ids(i), listSize);
    }
}

void CuvsIVFPQ::setCuvsIndex(cuvsIvfPqIndex_t index) {
    if (cuvs_index) {
        cuvsCheck(cuvsIvfPqIndexDestroy(cuvs_index), "cuvsIvfPqIndexDestroy");
    }
    cuvs_index = index;
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
                getCppIndex(cuvs_index),
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
            getCppIndex(cuvs_index)->inds_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_device(list_indices_ptr, indices, numVecs, stream);
}

void CuvsIVFPQ::setPQCentroids_() {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    CuvsOutputTensor pqCenters;
    cuvsCheck(
            cuvsIvfPqIndexGetPqCenters(cuvs_index, pqCenters.get()),
            "cuvsIvfPqIndexGetPqCenters");
    raft::copy(
            pqCenters.data<float>(),
            pqCentroidsInnermostCode_.data(),
            pqCentroidsInnermostCode_.numElements(),
            stream);
}

void CuvsIVFPQ::setBasePQCentroids_() {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    CuvsOutputTensor pqCenters;
    cuvsCheck(
            cuvsIvfPqIndexGetPqCenters(cuvs_index, pqCenters.get()),
            "cuvsIvfPqIndexGetPqCenters");
    raft::copy(
            pqCentroidsInnermostCode_.data(),
            pqCenters.data<float>(),
            pqCentroidsInnermostCode_.numElements(),
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
