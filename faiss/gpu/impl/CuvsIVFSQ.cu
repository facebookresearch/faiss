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

#include <cstddef>
#include <cstdint>

#include <cuvs/neighbors/ivf_sq.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/CuvsFilterConvert.h>
#include <faiss/gpu/utils/CuvsUtils.h>
#include <raft/core/resource/thrust_policy.hpp>
#include <thrust/extrema.h>
#include <faiss/gpu/impl/CuvsIVFSQ.cuh>
#include <faiss/gpu/impl/FlatIndex.cuh>
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

constexpr float kFaissSQ8Levels = 255.0f;

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
    if (cuvs_index) {
        cuvsCheck(
                cuvsIvfSqIndexReset(
                        cuvsResourcesFromGpuResources(resources_), cuvs_index),
                "cuvsIvfSqIndexReset");
    }
}

void CuvsIVFSQ::setCuvsIndex(cuvsIvfSqIndex_t index) {
    if (cuvs_index) {
        cuvsCheck(cuvsIvfSqIndexDestroy(cuvs_index), "cuvsIvfSqIndexDestroy");
    }
    cuvs_index = index;
    maxVectorId_ = -1;
    hasNegativeVectorId_ = false;
    copyCuvsSQToFaiss_(faissSQ_);
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
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    CuvsOutputTensor listSizes;
    cuvsCheck(
            cuvsIvfSqIndexGetListSizes(cuvs_index, listSizes.get()),
            "cuvsIvfSqIndexGetListSizes");

    uint32_t size;
    raft::update_host(
            &size,
            listSizes.data<uint32_t>() + listId,
            1,
            raftHandle.get_stream());
    raftHandle.sync_stream();

    return static_cast<idx_t>(size);
}

std::vector<idx_t> CuvsIVFSQ::getListIndices(idx_t listId) const {
    FAISS_ASSERT(cuvs_index != nullptr);
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    const idx_t listSize = getListLength(listId);
    std::vector<idx_t> indices(listSize);
    if (listSize == 0) {
        return indices;
    }
    CuvsOutputTensor listIndices;
    cuvsCheck(
            cuvsIvfSqIndexGetListIndices(
                    cuvs_index,
                    static_cast<uint32_t>(listId),
                    listIndices.get()),
            "cuvsIvfSqIndexGetListIndices");

    raft::update_host(
            indices.data(),
            listIndices.data<idx_t>(),
            listSize,
            raftHandle.get_stream());
    raftHandle.sync_stream();

    return indices;
}

std::vector<uint8_t> CuvsIVFSQ::getListVectorData(idx_t listId, bool gpuFormat)
        const {
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

    auto codesDevice = raft::make_device_matrix<uint8_t, uint32_t>(
            raftHandle,
            static_cast<uint32_t>(listSize),
            static_cast<uint32_t>(dim_));
    auto codesTensor = makeCuvsTensor(
            codesDevice.data_handle(), (int64_t)listSize, (int64_t)dim_);
    cuvsCheck(
            cuvsIvfSqIndexUnpackContiguousListData(
                    cuvsResourcesFromGpuResources(resources_),
                    cuvs_index,
                    codesTensor.get(),
                    static_cast<uint32_t>(listId),
                    0),
            "cuvsIvfSqIndexUnpackContiguousListData");

    raft::update_host(
            flatCodes.data(),
            codesDevice.data_handle(),
            flatCodes.size(),
            raftHandle.get_stream());
    raftHandle.sync_stream();
    return flatCodes;
}

void CuvsIVFSQ::updateQuantizer(Index* quantizer) {
    FAISS_THROW_IF_NOT(quantizer->is_trained);
    FAISS_THROW_IF_NOT(quantizer->d == getDim());
    FAISS_THROW_IF_NOT(quantizer->ntotal == getNumLists());
    FAISS_ASSERT(faissSQ_ != nullptr);
    FAISS_THROW_IF_NOT_MSG(
            faissSQ_->trained.size() == 2 * static_cast<size_t>(dim_),
            "cuVS IVF-SQ requires trained QT_8bit range data");

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

    std::vector<float> hostVmin(dim_);
    std::vector<float> hostDelta(dim_);
    for (int d = 0; d < dim_; ++d) {
        hostDelta[d] = faissSQ_->trained[dim_ + d] / kFaissSQ8Levels;
        hostVmin[d] = faissSQ_->trained[d] + 0.5f * hostDelta[d];
    }
    DeviceTensor<float, 1, true> vmin(
            resources_,
            makeSpaceAlloc(AllocType::Quantizer, space_, stream),
            {getDim()});
    DeviceTensor<float, 1, true> delta(
            resources_,
            makeSpaceAlloc(AllocType::Quantizer, space_, stream),
            {getDim()});
    raft::update_device(vmin.data(), hostVmin.data(), dim_, stream);
    raft::update_device(delta.data(), hostDelta.data(), dim_, stream);

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

    cuvsIvfSqIndexParams_t params = nullptr;
    cuvsCheck(
            cuvsIvfSqIndexParamsCreate(&params), "cuvsIvfSqIndexParamsCreate");
    CuvsUniquePtr<cuvsIvfSqIndexParams, cuvsIvfSqIndexParamsDestroy>
            paramsHolder(params);
    params->add_data_on_build = false;
    params->metric = metricFaissToCuvs(metric_, false);
    params->metric_arg = metricArg_;
    params->n_lists = numLists_;

    auto centersTensor =
            makeCuvsTensor(centers.data(), (int64_t)numLists_, (int64_t)dim_);
    auto vminTensor = makeCuvsTensor(vmin.data(), (int64_t)dim_);
    auto deltaTensor = makeCuvsTensor(delta.data(), (int64_t)dim_);

    cuvsIvfSqIndex_t index = nullptr;
    cuvsCheck(cuvsIvfSqIndexCreate(&index), "cuvsIvfSqIndexCreate");
    CuvsUniquePtr<cuvsIvfSqIndex, cuvsIvfSqIndexDestroy> indexHolder(index);
    cuvsCheck(
            cuvsIvfSqBuildFromCenters(
                    cuvsResourcesFromGpuResources(resources_),
                    params,
                    cuvsDtype<float>(),
                    centersTensor.get(),
                    centerNormsTensor ? centerNormsTensor->get() : nullptr,
                    vminTensor.get(),
                    deltaTensor.get(),
                    index),
            "cuvsIvfSqBuildFromCenters");
    raftHandle.sync_stream();
    setCuvsIndex(indexHolder.release());
}

void CuvsIVFSQ::copyInvertedListsFrom(const InvertedLists* ivf) {
    const size_t nlist = ivf ? ivf->nlist : 0;

    FAISS_ASSERT(cuvs_index != nullptr);
    FAISS_THROW_IF_NOT(nlist == static_cast<size_t>(numLists_));

    for (size_t i = 0; i < nlist; ++i) {
        const size_t listSize = ivf->list_size(i);
        FAISS_THROW_IF_NOT_FMT(
                listSize <= (size_t)std::numeric_limits<int>::max(),
                "GPU inverted list can only support %zu entries; %zu found",
                (size_t)std::numeric_limits<int>::max(),
                listSize);
        FAISS_ASSERT(getListLength(i) == 0);
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
    for (idx_t listId = 0; listId < numLists_; ++listId) {
        auto ids = getListIndices(listId);
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

        auto flatCodes = getListVectorData(listId, false);

        std::vector<float> decoded(ids.size() * dim_);
        faissSQ_->decode(flatCodes.data(), decoded.data(), ids.size());

        const float* center = centers.data() + listId * dim_;
        for (size_t offset = 0; offset < ids.size(); ++offset) {
            const idx_t id = ids[offset];
            if (id < i0 || id >= i0 + ni) {
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
    constexpr idx_t indexGroupSize = 32;
    constexpr idx_t vectorLength = 16;
    idx_t paddedDim = roundUpTo(dim_, vectorLength);
    idx_t numBlocks = utils::divUp(numVecs, indexGroupSize);
    return numBlocks * indexGroupSize * paddedDim;
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
    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    constexpr idx_t maxBatchSize = 4096;
    for (idx_t offset = 0; offset < numVecs; offset += maxBatchSize) {
        uint32_t batchSize = min(maxBatchSize, numVecs - offset);
        auto codesDevice = raft::make_device_matrix<uint8_t, uint32_t>(
                raftHandle, batchSize, static_cast<uint32_t>(dim_));
        auto indicesDevice = raft::make_device_vector<idx_t, uint32_t>(
                raftHandle, batchSize);
        raft::update_device(
                codesDevice.data_handle(),
                static_cast<const uint8_t*>(codes) + offset * dim_,
                static_cast<size_t>(batchSize) * dim_,
                stream);
        raft::update_device(
                indicesDevice.data_handle(),
                indices + offset,
                batchSize,
                stream);
        auto codesTensor = makeCuvsTensor(
                codesDevice.data_handle(), (int64_t)batchSize, (int64_t)dim_);
        auto indicesTensor =
                makeCuvsTensor(indicesDevice.data_handle(), (int64_t)batchSize);
        cuvsCheck(
                cuvsIvfSqIndexExtendList(
                        cuvsResourcesFromGpuResources(resources_),
                        cuvs_index,
                        codesTensor.get(),
                        indicesTensor.get(),
                        static_cast<uint32_t>(listId)),
                "cuvsIvfSqIndexExtendList");
    }
}

void CuvsIVFSQ::copyCuvsSQToFaiss_(faiss::ScalarQuantizer* sq) const {
    FAISS_ASSERT(cuvs_index != nullptr);
    FAISS_THROW_IF_NOT_MSG(sq, "ScalarQuantizer pointer cannot be null");

    const auto& raftHandle = resources_->getRaftHandleCurrentDevice();
    CuvsOutputTensor vminTensor;
    CuvsOutputTensor deltaTensor;
    cuvsCheck(
            cuvsIvfSqIndexGetVMin(cuvs_index, vminTensor.get()),
            "cuvsIvfSqIndexGetVMin");
    cuvsCheck(
            cuvsIvfSqIndexGetDelta(cuvs_index, deltaTensor.get()),
            "cuvsIvfSqIndexGetDelta");

    std::vector<float> vmin(dim_);
    std::vector<float> delta(dim_);
    raft::update_host(
            vmin.data(),
            vminTensor.data<float>(),
            dim_,
            raftHandle.get_stream());
    raft::update_host(
            delta.data(),
            deltaTensor.data<float>(),
            dim_,
            raftHandle.get_stream());
    raftHandle.sync_stream();

    sq->d = dim_;
    sq->qtype = faiss::ScalarQuantizer::QT_8bit;
    sq->set_derived_sizes();
    sq->trained.resize(2 * static_cast<size_t>(dim_));
    for (int d = 0; d < dim_; ++d) {
        sq->trained[d] = vmin[d] - 0.5f * delta[d];
        sq->trained[dim_ + d] = kFaissSQ8Levels * delta[d];
    }
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

} // namespace gpu
} // namespace faiss
