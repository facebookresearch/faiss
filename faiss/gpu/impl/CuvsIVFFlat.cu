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
#include <cstddef>
#include <cstdint>

#include <cuvs/neighbors/ivf_flat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/CuvsFilterConvert.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <faiss/gpu/impl/CuvsIVFFlat.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>
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

CuvsIVFFlat::~CuvsIVFFlat() {
    if (cuvs_index) {
        cuvsIvfFlatIndexDestroy(cuvs_index);
    }
}

void CuvsIVFFlat::reserveMemory(idx_t numVecs) {
    fprintf(stderr,
            "WARN: reserveMemory is NOP. Pre-allocation of IVF lists is not supported with cuVS enabled.\n");
}

void CuvsIVFFlat::reset() {
    if (cuvs_index) {
        cuvsCheck(
                cuvsIvfFlatIndexReset(
                        cuvsResourcesFromGpuResources(resources_), cuvs_index),
                "cuvsIvfFlatIndexReset");
    }
}

void CuvsIVFFlat::setCuvsIndex(cuvsIvfFlatIndex_t index) {
    if (cuvs_index) {
        cuvsCheck(
                cuvsIvfFlatIndexDestroy(cuvs_index), "cuvsIvfFlatIndexDestroy");
    }
    cuvs_index = index;
}

void CuvsIVFFlat::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        const IDSelector* sel) {
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
    cuvsIvfFlatSearchParams_t searchParams = nullptr;
    cuvsCheck(
            cuvsIvfFlatSearchParamsCreate(&searchParams),
            "cuvsIvfFlatSearchParamsCreate");
    CuvsUniquePtr<cuvsIvfFlatSearchParams, cuvsIvfFlatSearchParamsDestroy>
            searchParamsHolder(searchParams);
    searchParams->n_probes = nprobe;

    auto queriesTensor =
            makeCuvsTensor(queries.data(), (int64_t)numQueries, (int64_t)cols);
    auto indicesTensor =
            makeCuvsTensor(outIndices.data(), (int64_t)numQueries, (int64_t)k_);
    auto distancesTensor = makeCuvsTensor(
            outDistances.data(), (int64_t)numQueries, (int64_t)k_);
    int64_t indexSize = 0;
    cuvsCheck(
            cuvsIvfFlatIndexGetSize(cuvs_index, &indexSize),
            "cuvsIvfFlatIndexGetSize");
    CuvsFilter filter(resources_, sel, indexSize);

    cuvsCheck(
            cuvsIvfFlatSearch(
                    cuvsResourcesFromGpuResources(resources_),
                    searchParams,
                    cuvs_index,
                    queriesTensor.get(),
                    indicesTensor.get(),
                    distancesTensor.get(),
                    filter.get()),
            "cuvsIvfFlatSearch");

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

    /// Remove rows containing NaNs
    idx_t n_rows_valid = inplaceGatherFilteredRows(resources_, vecs, indices);

    auto vectorsTensor = makeCuvsTensor(vecs.data(), n_rows_valid, (idx_t)dim_);
    auto indicesTensor = makeCuvsTensor(indices.data(), n_rows_valid);
    cuvsCheck(
            cuvsIvfFlatExtend(
                    cuvsResourcesFromGpuResources(resources_),
                    vectorsTensor.get(),
                    indicesTensor.get(),
                    cuvs_index),
            "cuvsIvfFlatExtend");

    return n_rows_valid;
}

idx_t CuvsIVFFlat::getListLength(idx_t listId) const {
    FAISS_ASSERT(cuvs_index != nullptr);
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    CuvsOutputTensor listSizes;
    cuvsCheck(
            cuvsIvfFlatIndexGetListSizes(cuvs_index, listSizes.get()),
            "cuvsIvfFlatIndexGetListSizes");

    uint32_t size;
    raft::update_host(
            &size,
            listSizes.data<uint32_t>() + listId,
            1,
            raftHandle.get_stream());
    raftHandle.sync_stream();

    return static_cast<idx_t>(size);
}

/// Return the list indices of a particular list back to the CPU
std::vector<idx_t> CuvsIVFFlat::getListIndices(idx_t listId) const {
    FAISS_ASSERT(cuvs_index != nullptr);
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    const idx_t listSize = getListLength(listId);

    std::vector<idx_t> indices(listSize);
    if (listSize == 0) {
        return indices;
    }

    CuvsOutputTensor listIndices;
    cuvsCheck(
            cuvsIvfFlatIndexGetListIndices(
                    cuvs_index,
                    static_cast<uint32_t>(listId),
                    listIndices.get()),
            "cuvsIvfFlatIndexGetListIndices");

    raft::update_host(
            indices.data(),
            listIndices.data<idx_t>(),
            listSize,
            raftHandle.get_stream());
    raftHandle.sync_stream();

    return indices;
}

/// Return the encoded vectors of a particular list back to the CPU
std::vector<uint8_t> CuvsIVFFlat::getListVectorData(
        idx_t listId,
        bool gpuFormat) const {
    if (gpuFormat) {
        FAISS_THROW_MSG("gpuFormat should be false for cuVS indices");
    }
    FAISS_ASSERT(cuvs_index != nullptr);

    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    const idx_t listSize = getListLength(listId);

    std::vector<uint8_t> flatCodes(getCpuVectorsEncodingSize_(listSize));
    if (listSize == 0) {
        return flatCodes;
    }

    auto vectorsDevice = raft::make_device_matrix<float, uint32_t>(
            raftHandle,
            static_cast<uint32_t>(listSize),
            static_cast<uint32_t>(dim_));
    auto vectorsTensor = makeCuvsTensor(
            vectorsDevice.data_handle(), (int64_t)listSize, (int64_t)dim_);
    cuvsCheck(
            cuvsIvfFlatIndexUnpackListData(
                    cuvsResourcesFromGpuResources(resources_),
                    cuvs_index,
                    vectorsTensor.get(),
                    static_cast<uint32_t>(listId),
                    0),
            "cuvsIvfFlatIndexUnpackListData");

    raft::update_host(
            flatCodes.data(),
            reinterpret_cast<const uint8_t*>(vectorsDevice.data_handle()),
            flatCodes.size(),
            raftHandle.get_stream());
    raftHandle.sync_stream();
    return flatCodes;
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
    FAISS_THROW_IF_NOT(quantizer->d == getDim());
    FAISS_THROW_IF_NOT(quantizer->ntotal == getNumLists());

    const size_t totalElements =
            static_cast<size_t>(quantizer->ntotal) * quantizer->d;
    auto stream = resources_->getDefaultStreamCurrentDevice();
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    DeviceTensor<float, 2, true> centers(
            resources_,
            makeSpaceAlloc(AllocType::Quantizer, space_, stream),
            {getNumLists(), getDim()});
    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
    if (gpuQ) {
        auto gpuData = gpuQ->getGpuData();
        if (gpuData->getUseFloat16()) {
            gpuData->reconstruct(0, gpuData->getSize(), centers);
        } else {
            auto source = gpuData->getVectorsFloat32Ref();
            raft::copy(centers.data(), source.data(), totalElements, stream);
        }
    } else {
        std::vector<float> hostCenters(totalElements);
        quantizer->reconstruct_n(0, quantizer->ntotal, hostCenters.data());
        raft::update_device(
                centers.data(), hostCenters.data(), totalElements, stream);
    }

    DeviceTensor<float, 1, true> centerNorms;
    std::unique_ptr<CuvsTensor<float, 1>> centerNormsTensor;
    if (metric_ == faiss::METRIC_L2) {
        centerNorms = DeviceTensor<float, 1, true>(
                resources_,
                makeSpaceAlloc(AllocType::Quantizer, space_, stream),
                {getNumLists()});
        raft::linalg::rowNorm<raft::linalg::L2Norm, true, float, uint32_t>(
                centerNorms.data(),
                centers.data(),
                dim_,
                static_cast<uint32_t>(numLists_),
                stream);
        centerNormsTensor = std::make_unique<CuvsTensor<float, 1>>(
                centerNorms.data(), (int64_t)numLists_);
    }

    cuvsIvfFlatIndexParams_t params = nullptr;
    cuvsCheck(
            cuvsIvfFlatIndexParamsCreate(&params),
            "cuvsIvfFlatIndexParamsCreate");
    CuvsUniquePtr<cuvsIvfFlatIndexParams, cuvsIvfFlatIndexParamsDestroy>
            paramsHolder(params);
    params->add_data_on_build = false;
    params->metric = metricFaissToCuvs(metric_, false);
    params->metric_arg = metricArg_;
    params->n_lists = numLists_;

    auto centersTensor =
            makeCuvsTensor(centers.data(), (int64_t)numLists_, (int64_t)dim_);

    cuvsIvfFlatIndex_t index = nullptr;
    cuvsCheck(cuvsIvfFlatIndexCreate(&index), "cuvsIvfFlatIndexCreate");
    CuvsUniquePtr<cuvsIvfFlatIndex, cuvsIvfFlatIndexDestroy> indexHolder(index);
    cuvsCheck(
            cuvsIvfFlatBuildFromCenters(
                    cuvsResourcesFromGpuResources(resources_),
                    params,
                    cuvsDtype<float>(),
                    centersTensor.get(),
                    centerNormsTensor ? centerNormsTensor->get() : nullptr,
                    index),
            "cuvsIvfFlatBuildFromCenters");
    raftHandle.sync_stream();
    setCuvsIndex(indexHolder.release());
}

void CuvsIVFFlat::copyInvertedListsFrom(const InvertedLists* ivf) {
    const size_t nlist = ivf ? ivf->nlist : 0;
    FAISS_THROW_IF_NOT(nlist <= static_cast<size_t>(numLists_));
    FAISS_ASSERT(cuvs_index != nullptr);

    for (size_t i = 0; i < nlist; ++i) {
        const size_t listSize = ivf->list_size(i);
        FAISS_THROW_IF_NOT_FMT(
                listSize <= (size_t)std::numeric_limits<int>::max(),
                "GPU inverted list can only support %zu entries; %zu found",
                (size_t)std::numeric_limits<int>::max(),
                listSize);
        FAISS_ASSERT(getListLength(i) == 0);
        addEncodedVectorsToList_(
                i, ivf->get_codes(i), ivf->get_ids(i), listSize);
    }
}

size_t CuvsIVFFlat::getGpuVectorsEncodingSize_(idx_t numVecs) const {
    idx_t bits = 32 /* float */;

    // bytes to encode a block of 32 vectors (single dimension)
    idx_t bytesPerDimBlock = bits * 32 / 8; // = 128

    // bytes to fully encode 32 vectors
    idx_t bytesPerBlock = bytesPerDimBlock * dim_;

    // number of blocks of 32 vectors we have
    idx_t numBlocks = utils::divUp(numVecs, 32);

    // total size to encode numVecs
    return bytesPerBlock * numBlocks;
}

void CuvsIVFFlat::addEncodedVectorsToList_(
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        idx_t numVecs) {
    if (numVecs == 0) {
        return;
    }

    auto stream = resources_->getDefaultStreamCurrentDevice();
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    constexpr idx_t maxBatchSize = 4096;
    for (idx_t offset = 0; offset < numVecs; offset += maxBatchSize) {
        uint32_t batchSize = min(maxBatchSize, numVecs - offset);
        auto vectorsDevice = raft::make_device_matrix<float, uint32_t>(
                raftHandle, batchSize, static_cast<uint32_t>(dim_));
        auto indicesDevice = raft::make_device_vector<idx_t, uint32_t>(
                raftHandle, batchSize);

        raft::update_device(
                vectorsDevice.data_handle(),
                reinterpret_cast<const float*>(codes) + offset * dim_,
                static_cast<size_t>(batchSize) * dim_,
                stream);

        raft::update_device(
                indicesDevice.data_handle(),
                indices + offset,
                batchSize,
                stream);
        auto vectorsTensor = makeCuvsTensor(
                vectorsDevice.data_handle(), (int64_t)batchSize, (int64_t)dim_);
        auto indicesTensor =
                makeCuvsTensor(indicesDevice.data_handle(), (int64_t)batchSize);
        cuvsCheck(
                cuvsIvfFlatIndexExtendList(
                        cuvsResourcesFromGpuResources(resources_),
                        cuvs_index,
                        vectorsTensor.get(),
                        indicesTensor.get(),
                        static_cast<uint32_t>(listId)),
                "cuvsIvfFlatIndexExtendList");
    }
}

} // namespace gpu
} // namespace faiss
