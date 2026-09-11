// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <faiss/gpu/utils/CuvsFilterConvert.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsIVFSQ.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <cuvs/neighbors/ivf_sq.h>
#include <cuvs/neighbors/ivf_sq.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <thrust/extrema.h>
#include <raft/core/copy.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace faiss {
namespace gpu {

namespace {

// Compatibility note: release/26.10 has no C API for resetting, constructing
// from caller-provided coarse centers and scalar-quantizer state, or importing
// and exporting, resizing, inspecting, and packing raw IVF-SQ list storage.
// build, search, extend, total-size, and centers operations use C; these C++
// types/helpers are retained only for Faiss inverted-list interoperability.
constexpr float kFaissSQ8Levels = 255.0f;
using CuvsSQListSpec =
        cuvs::neighbors::ivf_sq::list_spec<uint32_t, uint8_t, int64_t>;
constexpr uint32_t kCuvsSQVecLen = CuvsSQListSpec::kVecLen;
using CuvsIVFSQCppIndex = cuvs::neighbors::ivf_sq::index<uint8_t>;

CuvsIVFSQCppIndex* getCppIndex(cuvsIvfSqIndex_t index) {
    return reinterpret_cast<CuvsIVFSQCppIndex*>(index->addr);
}

uint32_t roundUpTo(uint32_t value, uint32_t align) {
    return ((value + align - 1) / align) * align;
}

} // namespace

CuvsIVFSQ::CuvsIVFSQ(
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
                  indicesOptions,
                  space),
          faissSQ_(scalarQ) {
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == INDICES_64_BIT,
            "only INDICES_64_BIT is supported for cuVS IVF-SQ");
    FAISS_THROW_IF_NOT_MSG(
            useResidual, "cuVS IVF-SQ requires residual encoding");
    FAISS_THROW_IF_NOT_MSG(
            scalarQ &&
                    scalarQ->qtype ==
                            faiss::ScalarQuantizer::QuantizerType::QT_8bit,
            "cuVS IVF-SQ supports only QT_8bit scalar quantization");
}

CuvsIVFSQ::~CuvsIVFSQ() {
    if (cuvs_index) {
        cuvsIvfSqIndexDestroy(cuvs_index);
    }
}

void CuvsIVFSQ::reserveMemory(idx_t /* numVecs */) {
    fprintf(stderr,
            "WARN: reserveMemory is NOP. Pre-allocation of IVF lists is not supported with cuVS enabled.\n");
}

size_t CuvsIVFSQ::reclaimMemory() {
    fprintf(stderr,
            "WARN: reclaimMemory is NOP. Memory reclamation is not supported with cuVS enabled.\n");
    return 0;
}

void CuvsIVFSQ::reset() {
    maxVectorId_ = -1;
    hasNegativeVectorId_ = false;
    if (cuvs_index == nullptr) {
        return;
    }

    // Compatibility note: release/26.10 has no IVF-SQ C reset operation.
    // Reconstruct an empty C++ index with the existing training state, then
    // place it back behind the public C handle used by the rest of this class.
    const raft::device_resources& raftHandle =
            resources_->getRaftHandleCurrentDevice();
    auto* current = getCppIndex(cuvs_index);
    auto replacement = std::make_unique<CuvsIVFSQCppIndex>(
            raftHandle,
            current->metric(),
            current->n_lists(),
            current->dim(),
            current->conservative_memory_allocation());

    CuvsOutputTensor centers;
    cuvsCheck(
            cuvsIvfSqIndexGetCenters(cuvs_index, centers.get()),
            "cuvsIvfSqIndexGetCenters");
    auto stream = raftHandle.get_stream();
    raft::copy(
            replacement->centers().data_handle(),
            centers.data<float>(),
            static_cast<size_t>(numLists_) * dim_,
            stream);
    raft::copy(
            replacement->sq_vmin().data_handle(),
            current->sq_vmin().data_handle(),
            dim_,
            stream);
    raft::copy(
            replacement->sq_delta().data_handle(),
            current->sq_delta().data_handle(),
            dim_,
            stream);

    cuvsIvfSqIndex_t newIndex = nullptr;
    cuvsCheck(cuvsIvfSqIndexCreate(&newIndex), "cuvsIvfSqIndexCreate");
    CuvsUniquePtr<cuvsIvfSqIndex, cuvsIvfSqIndexDestroy> newIndexHolder(
            newIndex);
    newIndex->addr = reinterpret_cast<uintptr_t>(replacement.release());
    newIndex->dtype = cuvs_index->dtype;
    raftHandle.sync_stream();

    cuvsCheck(cuvsIvfSqIndexDestroy(cuvs_index), "cuvsIvfSqIndexDestroy");
    cuvs_index = newIndexHolder.release();
    computeCenterNorms_();
}

void CuvsIVFSQ::setCuvsIndex(cuvsIvfSqIndex_t index) {
    if (cuvs_index) {
        cuvsCheck(cuvsIvfSqIndexDestroy(cuvs_index), "cuvsIvfSqIndexDestroy");
    }
    cuvs_index = index;
    // Callers install a freshly-built index that has no vectors yet.
    maxVectorId_ = -1;
    hasNegativeVectorId_ = false;
    copyCuvsSQToFaiss_(faissSQ_);
    computeCenterNorms_();
}

void CuvsIVFSQ::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        const IDSelector* sel) {
    /// NB: The coarse quantizer is ignored here. The user is assumed to have
    /// called updateQuantizer() to modify the cuVS index if the quantizer was
    /// modified externally.

    uint32_t numQueries = queries.getSize(0);
    uint32_t cols = queries.getSize(1);
    uint32_t k_ = k;

    FAISS_ASSERT(cuvs_index != nullptr);
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    cuvsIvfSqSearchParams_t searchParams = nullptr;
    cuvsCheck(
            cuvsIvfSqSearchParamsCreate(&searchParams),
            "cuvsIvfSqSearchParamsCreate");
    CuvsUniquePtr<cuvsIvfSqSearchParams, cuvsIvfSqSearchParamsDestroy>
            searchParamsHolder(searchParams);
    searchParams->n_probes = nprobe;

    auto queriesTensor =
            makeCuvsTensor(queries.data(), (int64_t)numQueries, (int64_t)cols);
    auto indicesTensor =
            makeCuvsTensor(outIndices.data(), (int64_t)numQueries, (int64_t)k_);
    auto distancesTensor = makeCuvsTensor(
            outDistances.data(), (int64_t)numQueries, (int64_t)k_);
    CuvsFilter filter(resources_, sel, getBitsetSizeForFiltering_());
    cuvsCheck(
            cuvsIvfSqSearch(
                    cuvsResourcesFromGpuResources(resources_),
                    searchParams,
                    cuvs_index,
                    queriesTensor.get(),
                    indicesTensor.get(),
                    distancesTensor.get(),
                    filter.get()),
            "cuvsIvfSqSearch");

    /// Identify NaN rows and mask their nearest neighbors
    auto nan_flag = raft::make_device_vector<bool>(raft_handle, numQueries);

    validRowIndices(resources_, queries, nan_flag.data_handle());

    idx_t numResults = idx_t(numQueries) * k_;
    raft::linalg::map_offset(
            raft_handle,
            raft::make_device_vector_view<idx_t, idx_t>(
                    outIndices.data(), numResults),
            [nan_flag = nan_flag.data_handle(),
             out_inds = outIndices.data(),
             k_] __device__(idx_t i) {
                idx_t row = i / k_;
                if (!nan_flag[row]) {
                    return idx_t(-1);
                }
                return out_inds[i];
            });

    float max_val = std::numeric_limits<float>::max();
    raft::linalg::map_offset(
            raft_handle,
            raft::make_device_vector_view<float, idx_t>(
                    outDistances.data(), numResults),
            [nan_flag = nan_flag.data_handle(),
             out_dists = outDistances.data(),
             max_val,
             k_] __device__(idx_t i) {
                idx_t row = i / k_;
                if (!nan_flag[row]) {
                    return max_val;
                }
                return out_dists[i];
            });
    raft_handle.sync_stream();
}

void CuvsIVFSQ::searchPreassigned(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& ivfDistances,
        Tensor<idx_t, 2, true>& ivfAssignments,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool storePairs) {
    FAISS_THROW_MSG("searchPreassigned is not implemented for cuVS index");
}

idx_t CuvsIVFSQ::addVectors(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
    /// NB: The coarse quantizer is ignored here. The user is assumed to have
    /// called updateQuantizer() to update the cuVS index if the quantizer was
    /// modified externally.

    FAISS_ASSERT(cuvs_index != nullptr);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    /// Remove rows containing NaNs
    idx_t n_rows_valid = inplaceGatherFilteredRows(resources_, vecs, indices);

    auto vectorsTensor = makeCuvsTensor(vecs.data(), n_rows_valid, (idx_t)dim_);
    auto indicesTensor = makeCuvsTensor(indices.data(), n_rows_valid);
    cuvsCheck(
            cuvsIvfSqExtend(
                    cuvsResourcesFromGpuResources(resources_),
                    vectorsTensor.get(),
                    indicesTensor.get(),
                    cuvs_index),
            "cuvsIvfSqExtend");

    /// Track the id range once at add time so getBitsetSizeForFiltering_() does
    /// not have to read back the whole index on every filtered search. A single
    /// minmax pass keeps this to one device->host sync per add batch.
    if (n_rows_valid > 0) {
        auto thrust_policy = raft::resource::get_thrust_policy(raft_handle);
        auto minmax = thrust::minmax_element(
                thrust_policy, indices.data(), indices.data() + n_rows_valid);
        idx_t batchMinId = 0;
        idx_t batchMaxId = -1;
        raft::update_host(
                &batchMinId, minmax.first, 1, raft_handle.get_stream());
        raft::update_host(
                &batchMaxId, minmax.second, 1, raft_handle.get_stream());
        raft_handle.sync_stream();
        if (batchMinId < 0) {
            hasNegativeVectorId_ = true;
        }
        maxVectorId_ = std::max(maxVectorId_, batchMaxId);
    }

    return n_rows_valid;
}

idx_t CuvsIVFSQ::getListLength(idx_t listId) const {
    FAISS_ASSERT(cuvs_index != nullptr);
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    uint32_t size;
    raft::update_host(
            &size,
            getCppIndex(cuvs_index)->list_sizes().data_handle() + listId,
            1,
            raft_handle.get_stream());
    raft_handle.sync_stream();

    return static_cast<idx_t>(size);
}

std::vector<idx_t> CuvsIVFSQ::getListIndices(idx_t listId) const {
    FAISS_ASSERT(cuvs_index != nullptr);
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    idx_t listSize = getListLength(listId);
    std::vector<idx_t> vec(listSize);
    if (listSize == 0) {
        return vec;
    }

    idx_t* list_indices_ptr;
    raft::update_host(
            &list_indices_ptr,
            const_cast<idx_t**>(
                    getCppIndex(cuvs_index)->inds_ptrs().data_handle()) +
                    listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_host(vec.data(), list_indices_ptr, listSize, stream);
    raft_handle.sync_stream();

    return vec;
}

std::vector<uint8_t> CuvsIVFSQ::getListVectorData(idx_t listId, bool gpuFormat)
        const {
    if (gpuFormat) {
        FAISS_THROW_MSG("gpuFormat should be false for cuVS indices");
    }
    FAISS_ASSERT(cuvs_index != nullptr);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    idx_t listSize = getListLength(listId);
    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(listSize);
    std::vector<uint8_t> flat_codes(cpuListSizeInBytes);
    if (listSize == 0) {
        return flat_codes;
    }

    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(listSize);
    std::vector<uint8_t> interleaved_codes(gpuListSizeInBytes);

    uint8_t* list_data_ptr;
    raft::update_host(
            &list_data_ptr,
            const_cast<uint8_t**>(
                    getCppIndex(cuvs_index)->data_ptrs().data_handle()) +
                    listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_host(
            interleaved_codes.data(),
            list_data_ptr,
            gpuListSizeInBytes,
            stream);
    raft_handle.sync_stream();

    CuvsIVFSQCodePackerInterleaved packer((size_t)listSize, dim_);
    packer.unpack_all(interleaved_codes.data(), flat_codes.data());
    return flat_codes;
}

void CuvsIVFSQ::updateQuantizer(Index* quantizer) {
    FAISS_THROW_IF_NOT(quantizer->is_trained);
    FAISS_THROW_IF_NOT(quantizer->d == getDim());
    FAISS_THROW_IF_NOT(quantizer->ntotal == getNumLists());

    int64_t indexSize = 0;
    if (cuvs_index) {
        cuvsCheck(
                cuvsIvfSqIndexGetSize(cuvs_index, &indexSize),
                "cuvsIvfSqIndexGetSize");
    }
    FAISS_THROW_IF_NOT_MSG(
            indexSize == 0,
            "updateQuantizer() cannot be called after vectors have been added: "
            "it would discard the stored inverted lists");

    // Compatibility note: release/26.10 has no C API that constructs an
    // empty IVF-SQ index from caller-provided coarse centers and scalar
    // quantizer state. Preserve that state with the C++ constructor, then put
    // the replacement behind the C handle used by supported operations.
    const raft::device_resources& raftHandle =
            resources_->getRaftHandleCurrentDevice();
    cuvs::neighbors::ivf_sq::index_params params;
    params.add_data_on_build = false;
    params.metric = static_cast<cuvs::distance::DistanceType>(
            metricFaissToCuvs(metric_, false));
    params.metric_arg = metricArg_;
    params.n_lists = numLists_;
    auto replacement = std::make_unique<CuvsIVFSQCppIndex>(
            raftHandle, params, static_cast<uint32_t>(dim_));

    const size_t totalElements =
            static_cast<size_t>(quantizer->ntotal) * quantizer->d;
    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
    if (gpuQ) {
        auto gpuData = gpuQ->getGpuData();
        if (gpuData->getUseFloat16()) {
            DeviceTensor<float, 2, true> centroids(
                    resources_,
                    makeSpaceAlloc(AllocType::FlatData, space_, stream),
                    {getNumLists(), getDim()});
            gpuData->reconstruct(0, gpuData->getSize(), centroids);
            raft::update_device(
                    replacement->centers().data_handle(),
                    centroids.data(),
                    totalElements,
                    stream);
        } else {
            auto centroids = gpuData->getVectorsFloat32Ref();
            raft::update_device(
                    replacement->centers().data_handle(),
                    centroids.data(),
                    totalElements,
                    stream);
        }
    } else {
        std::vector<float> centroids(totalElements);
        quantizer->reconstruct_n(0, quantizer->ntotal, centroids.data());
        raft::update_device(
                replacement->centers().data_handle(),
                centroids.data(),
                totalElements,
                stream);
    }
    raftHandle.sync_stream();

    cuvsIvfSqIndex_t index = nullptr;
    cuvsCheck(cuvsIvfSqIndexCreate(&index), "cuvsIvfSqIndexCreate");
    CuvsUniquePtr<cuvsIvfSqIndex, cuvsIvfSqIndexDestroy> indexHolder(index);
    index->addr = reinterpret_cast<uintptr_t>(replacement.release());
    index->dtype = cuvsDtype<float>();
    if (cuvs_index) {
        cuvsCheck(cuvsIvfSqIndexDestroy(cuvs_index), "cuvsIvfSqIndexDestroy");
    }
    cuvs_index = indexHolder.release();
    maxVectorId_ = -1;
    hasNegativeVectorId_ = false;
    copyFaissSQToCuvs_();
    computeCenterNorms_();
}

void CuvsIVFSQ::copyInvertedListsFrom(const InvertedLists* ivf) {
    size_t nlist = ivf ? ivf->nlist : 0;

    FAISS_ASSERT(cuvs_index != nullptr);
    FAISS_ASSERT(faissSQ_ != nullptr);
    FAISS_THROW_IF_NOT(nlist == static_cast<size_t>(numLists_));
    FAISS_THROW_IF_NOT_MSG(
            faissSQ_->qtype == faiss::ScalarQuantizer::QT_8bit,
            "cuVS IVF-SQ supports only QT_8bit scalar quantization");
    FAISS_THROW_IF_NOT_MSG(
            faissSQ_->trained.size() == 2 * static_cast<size_t>(dim_),
            "cuVS IVF-SQ requires trained QT_8bit range data");

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    std::vector<uint32_t> listSizes(nlist);
    auto& cuvsIndexLists = getCppIndex(cuvs_index)->lists();
    CuvsSQListSpec listSpec{
            static_cast<uint32_t>(dim_),
            true /* conservative_memory_allocation */};

    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        FAISS_THROW_IF_NOT_FMT(
                listSize <= (size_t)std::numeric_limits<int>::max(),
                "GPU inverted list can only support "
                "%zu entries; %zu found",
                (size_t)std::numeric_limits<int>::max(),
                listSize);

        listSizes[i] = static_cast<uint32_t>(listSize);
        FAISS_ASSERT(getListLength(i) == 0);
        cuvs::neighbors::ivf::resize_list(
                raft_handle,
                cuvsIndexLists[i],
                listSpec,
                static_cast<uint32_t>(listSize),
                static_cast<uint32_t>(0));
    }

    recomputeListState_(listSizes);

    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        const idx_t* ids = ivf->get_ids(i);
        for (size_t j = 0; j < listSize; ++j) {
            if (ids[j] < 0) {
                hasNegativeVectorId_ = true;
            } else {
                maxVectorId_ = std::max(maxVectorId_, ids[j]);
            }
        }
        addEncodedVectorsToList_(i, ivf->get_codes(i), ids, listSize);
    }

    computeCenterNorms_();
}

void CuvsIVFSQ::setFaissSQFromCuvs(faiss::ScalarQuantizer* sq) const {
    copyCuvsSQToFaiss_(sq);
}

void CuvsIVFSQ::reconstruct_n(idx_t i0, idx_t ni, float* out) {
    if (ni == 0) {
        return;
    }

    FAISS_ASSERT(cuvs_index != nullptr);
    FAISS_ASSERT(faissSQ_ != nullptr);

    std::fill(out, out + ni * dim_, 0.0f);
    auto centers = getCentersHost_();

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    // Read all per-list metadata in three batched copies rather than syncing
    // once per list as getListLength()/getListIndices()/getListVectorData()
    // would; those helpers each issue their own sync_stream().
    std::vector<uint32_t> listSizes(numLists_);
    std::vector<idx_t*> indsPtrs(numLists_);
    std::vector<uint8_t*> dataPtrs(numLists_);
    raft::update_host(
            listSizes.data(),
            getCppIndex(cuvs_index)->list_sizes().data_handle(),
            numLists_,
            stream);
    raft::update_host(
            indsPtrs.data(),
            getCppIndex(cuvs_index)->inds_ptrs().data_handle(),
            numLists_,
            stream);
    raft::update_host(
            dataPtrs.data(),
            getCppIndex(cuvs_index)->data_ptrs().data_handle(),
            numLists_,
            stream);
    raft_handle.sync_stream();

    for (idx_t listId = 0; listId < numLists_; ++listId) {
        idx_t listSize = listSizes[listId];
        if (listSize == 0) {
            continue;
        }

        std::vector<idx_t> ids(listSize);
        raft::update_host(ids.data(), indsPtrs[listId], listSize, stream);
        raft_handle.sync_stream();

        // Skip the (expensive) code readback + decode unless this list holds at
        // least one id in the requested [i0, i0 + ni) range.
        bool hasIdInRange = false;
        for (idx_t id : ids) {
            if (id >= i0 && id < i0 + ni) {
                hasIdInRange = true;
                break;
            }
        }
        if (!hasIdInRange) {
            continue;
        }

        auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(listSize);
        std::vector<uint8_t> interleavedCodes(gpuListSizeInBytes);
        raft::update_host(
                interleavedCodes.data(),
                dataPtrs[listId],
                gpuListSizeInBytes,
                stream);
        raft_handle.sync_stream();

        std::vector<uint8_t> flatCodes(getCpuVectorsEncodingSize_(listSize));
        CuvsIVFSQCodePackerInterleaved packer((size_t)listSize, dim_);
        packer.unpack_all(interleavedCodes.data(), flatCodes.data());

        std::vector<float> decoded(listSize * dim_);
        faissSQ_->decode(flatCodes.data(), decoded.data(), listSize);

        const float* center = centers.data() + listId * dim_;
        for (idx_t offset = 0; offset < listSize; ++offset) {
            idx_t id = ids[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }

            float* outRow = out + (id - i0) * dim_;
            const float* decodedRow = decoded.data() + offset * dim_;
            for (int d = 0; d < dim_; ++d) {
                outRow[d] = decodedRow[d] + center[d];
            }
        }
    }
}

size_t CuvsIVFSQ::getGpuVectorsEncodingSize_(idx_t numVecs) const {
    idx_t paddedDim = roundUpTo(dim_, kCuvsSQVecLen);
    idx_t numBlocks =
            utils::divUp(numVecs, cuvs::neighbors::ivf_sq::kIndexGroupSize);
    return numBlocks * cuvs::neighbors::ivf_sq::kIndexGroupSize * paddedDim;
}

void CuvsIVFSQ::addEncodedVectorsToList_(
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        idx_t numVecs) {
    if (numVecs == 0) {
        return;
    }

    FAISS_ASSERT(cuvs_index != nullptr);

    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
    FAISS_ASSERT(gpuListSizeInBytes <= (size_t)std::numeric_limits<int>::max());

    std::vector<uint8_t> interleavedCodes(gpuListSizeInBytes);
    CuvsIVFSQCodePackerInterleaved packer((size_t)numVecs, dim_);
    packer.pack_all(
            reinterpret_cast<const uint8_t*>(codes), interleavedCodes.data());

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    uint8_t* listDataPtr;
    raft::update_host(
            &listDataPtr,
            getCppIndex(cuvs_index)->data_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_device(
            listDataPtr, interleavedCodes.data(), gpuListSizeInBytes, stream);

    idx_t* listIndicesPtr;
    raft::update_host(
            &listIndicesPtr,
            getCppIndex(cuvs_index)->inds_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_device(listIndicesPtr, indices, numVecs, stream);
}

void CuvsIVFSQ::copyFaissSQToCuvs_() {
    FAISS_ASSERT(cuvs_index != nullptr);
    FAISS_ASSERT(faissSQ_ != nullptr);
    FAISS_THROW_IF_NOT_MSG(
            faissSQ_->qtype == faiss::ScalarQuantizer::QT_8bit,
            "cuVS IVF-SQ supports only QT_8bit scalar quantization");
    FAISS_THROW_IF_NOT_MSG(
            faissSQ_->trained.size() == 2 * static_cast<size_t>(dim_),
            "cuVS IVF-SQ requires trained QT_8bit range data");

    std::vector<float> vmin(dim_);
    std::vector<float> delta(dim_);
    for (int d = 0; d < dim_; ++d) {
        delta[d] = faissSQ_->trained[dim_ + d] / kFaissSQ8Levels;
        vmin[d] = faissSQ_->trained[d] + 0.5f * delta[d];
    }

    auto stream = resources_->getDefaultStreamCurrentDevice();
    raft::update_device(
            getCppIndex(cuvs_index)->sq_vmin().data_handle(),
            vmin.data(),
            dim_,
            stream);
    raft::update_device(
            getCppIndex(cuvs_index)->sq_delta().data_handle(),
            delta.data(),
            dim_,
            stream);
}

void CuvsIVFSQ::copyCuvsSQToFaiss_(faiss::ScalarQuantizer* sq) const {
    FAISS_ASSERT(cuvs_index != nullptr);
    FAISS_THROW_IF_NOT_MSG(sq, "ScalarQuantizer pointer cannot be null");

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    std::vector<float> vmin(dim_);
    std::vector<float> delta(dim_);
    raft::update_host(
            vmin.data(),
            getCppIndex(cuvs_index)->sq_vmin().data_handle(),
            dim_,
            stream);
    raft::update_host(
            delta.data(),
            getCppIndex(cuvs_index)->sq_delta().data_handle(),
            dim_,
            stream);
    raft_handle.sync_stream();

    sq->d = dim_;
    sq->qtype = faiss::ScalarQuantizer::QT_8bit;
    sq->set_derived_sizes();
    sq->trained.resize(2 * static_cast<size_t>(dim_));
    for (int d = 0; d < dim_; ++d) {
        sq->trained[d] = vmin[d] - 0.5f * delta[d];
        sq->trained[dim_ + d] = kFaissSQ8Levels * delta[d];
    }
}

void CuvsIVFSQ::computeCenterNorms_() {
    FAISS_ASSERT(cuvs_index != nullptr);

    if (metric_ != faiss::METRIC_L2) {
        return;
    }

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    getCppIndex(cuvs_index)->allocate_center_norms(raft_handle);
    CuvsOutputTensor centers;
    cuvsCheck(
            cuvsIvfSqIndexGetCenters(cuvs_index, centers.get()),
            "cuvsIvfSqIndexGetCenters");
    raft::linalg::rowNorm<raft::linalg::L2Norm, true, float, uint32_t>(
            getCppIndex(cuvs_index)->center_norms().value().data_handle(),
            centers.data<float>(),
            dim_,
            static_cast<uint32_t>(numLists_),
            raft_handle.get_stream());
}

void CuvsIVFSQ::recomputeListState_(const std::vector<uint32_t>& listSizes) {
    FAISS_ASSERT(cuvs_index != nullptr);
    FAISS_THROW_IF_NOT(listSizes.size() == static_cast<size_t>(numLists_));

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    std::vector<uint8_t*> dataPtrs(numLists_);
    std::vector<idx_t*> indsPtrs(numLists_);
    auto& lists = getCppIndex(cuvs_index)->lists();
    for (idx_t i = 0; i < numLists_; ++i) {
        dataPtrs[i] = lists[i] ? lists[i]->data_ptr() : nullptr;
        indsPtrs[i] = lists[i] ? lists[i]->indices_ptr() : nullptr;
    }

    raft::update_device(
            getCppIndex(cuvs_index)->data_ptrs().data_handle(),
            dataPtrs.data(),
            dataPtrs.size(),
            stream);
    raft::update_device(
            getCppIndex(cuvs_index)->inds_ptrs().data_handle(),
            indsPtrs.data(),
            indsPtrs.size(),
            stream);
    raft::update_device(
            getCppIndex(cuvs_index)->list_sizes().data_handle(),
            listSizes.data(),
            listSizes.size(),
            stream);

    auto sortedListSizes = listSizes;
    std::sort(
            sortedListSizes.begin(),
            sortedListSizes.end(),
            [](uint32_t a, uint32_t b) { return a > b; });

    auto accumSortedSizes = getCppIndex(cuvs_index)->accum_sorted_sizes();
    accumSortedSizes(0) = 0;
    for (size_t i = 0; i < sortedListSizes.size(); ++i) {
        accumSortedSizes(i + 1) = accumSortedSizes(i) + sortedListSizes[i];
    }
    raft_handle.sync_stream();
}

idx_t CuvsIVFSQ::getBitsetSizeForFiltering_() const {
    FAISS_ASSERT(cuvs_index != nullptr);

    // maxVectorId_ / hasNegativeVectorId_ are maintained incrementally as
    // vectors are added, so no per-search readback of the index is needed.
    FAISS_THROW_IF_MSG(
            hasNegativeVectorId_,
            "cuVS IVF-SQ IDSelector filtering does not support "
            "negative vector ids");

    int64_t indexSize = 0;
    cuvsCheck(
            cuvsIvfSqIndexGetSize(cuvs_index, &indexSize),
            "cuvsIvfSqIndexGetSize");
    if (maxVectorId_ < 0) {
        return static_cast<idx_t>(indexSize);
    }
    FAISS_THROW_IF_NOT_MSG(
            maxVectorId_ < std::numeric_limits<idx_t>::max(),
            "cuVS IVF-SQ IDSelector filtering cannot represent the largest "
            "idx_t value");
    return std::max(static_cast<idx_t>(indexSize), maxVectorId_ + 1);
}

std::vector<float> CuvsIVFSQ::getCentersHost_() const {
    FAISS_ASSERT(cuvs_index != nullptr);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    CuvsOutputTensor centersTensor;
    cuvsCheck(
            cuvsIvfSqIndexGetCenters(cuvs_index, centersTensor.get()),
            "cuvsIvfSqIndexGetCenters");
    std::vector<float> centers(static_cast<size_t>(numLists_) * dim_);
    raft::update_host(
            centers.data(),
            centersTensor.data<float>(),
            centers.size(),
            stream);
    raft_handle.sync_stream();
    return centers;
}

CuvsIVFSQCodePackerInterleaved::CuvsIVFSQCodePackerInterleaved(
        size_t list_size,
        uint32_t dim) {
    this->dim = dim;
    this->padded_dim = roundUpTo(dim, kCuvsSQVecLen);
    nvec = list_size;
    code_size = dim;
    block_size =
            utils::roundUp(nvec, cuvs::neighbors::ivf_sq::kIndexGroupSize) *
            padded_dim;
}

void CuvsIVFSQCodePackerInterleaved::pack_1(
        const uint8_t* flat_code,
        size_t offset,
        uint8_t* block) const {
    const uint32_t groupOffset =
            (offset / cuvs::neighbors::ivf_sq::kIndexGroupSize) *
            cuvs::neighbors::ivf_sq::kIndexGroupSize;
    const uint32_t ingroupId =
            offset % cuvs::neighbors::ivf_sq::kIndexGroupSize;
    uint8_t* group = block + static_cast<size_t>(groupOffset) * padded_dim;

    for (uint32_t d = 0; d < dim; ++d) {
        uint32_t l = (d / kCuvsSQVecLen) * kCuvsSQVecLen;
        uint32_t j = d % kCuvsSQVecLen;
        group[l * cuvs::neighbors::ivf_sq::kIndexGroupSize +
              ingroupId * kCuvsSQVecLen + j] = flat_code[d];
    }
}

void CuvsIVFSQCodePackerInterleaved::unpack_1(
        const uint8_t* block,
        size_t offset,
        uint8_t* flat_code) const {
    const uint32_t groupOffset =
            (offset / cuvs::neighbors::ivf_sq::kIndexGroupSize) *
            cuvs::neighbors::ivf_sq::kIndexGroupSize;
    const uint32_t ingroupId =
            offset % cuvs::neighbors::ivf_sq::kIndexGroupSize;
    const uint8_t* group =
            block + static_cast<size_t>(groupOffset) * padded_dim;

    for (uint32_t d = 0; d < dim; ++d) {
        uint32_t l = (d / kCuvsSQVecLen) * kCuvsSQVecLen;
        uint32_t j = d % kCuvsSQVecLen;
        flat_code[d] =
                group[l * cuvs::neighbors::ivf_sq::kIndexGroupSize +
                      ingroupId * kCuvsSQVecLen + j];
    }
}

CodePacker* CuvsIVFSQCodePackerInterleaved::clone() const {
    return new CuvsIVFSQCodePackerInterleaved(*this);
}

} // namespace gpu
} // namespace faiss
