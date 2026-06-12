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

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>
#include <raft/core/copy.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace faiss {
namespace gpu {

namespace {

constexpr float kFaissSQ8Levels = 255.0f;
using CuvsSQListSpec =
        cuvs::neighbors::ivf_sq::list_spec<uint32_t, uint8_t, int64_t>;
constexpr uint32_t kCuvsSQVecLen = CuvsSQListSpec::kVecLen;

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

CuvsIVFSQ::~CuvsIVFSQ() {}

void CuvsIVFSQ::reserveMemory(idx_t numVecs) {
    fprintf(stderr,
            "WARN: reserveMemory is NOP. Pre-allocation of IVF lists is not supported with cuVS enabled.\n");
}

size_t CuvsIVFSQ::reclaimMemory() {
    fprintf(stderr,
            "WARN: reclaimMemory is NOP. Memory reclamation is not supported with cuVS enabled.\n");
    return 0;
}

void CuvsIVFSQ::reset() {
    if (cuvs_index == nullptr) {
        return;
    }

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    auto new_index = std::make_shared<cuvs::neighbors::ivf_sq::index<uint8_t>>(
            raft_handle,
            cuvs_index->metric(),
            cuvs_index->n_lists(),
            cuvs_index->dim(),
            cuvs_index->conservative_memory_allocation());

    raft::copy(
            new_index->centers().data_handle(),
            cuvs_index->centers().data_handle(),
            static_cast<size_t>(numLists_) * dim_,
            stream);
    raft::copy(
            new_index->sq_vmin().data_handle(),
            cuvs_index->sq_vmin().data_handle(),
            dim_,
            stream);
    raft::copy(
            new_index->sq_delta().data_handle(),
            cuvs_index->sq_delta().data_handle(),
            dim_,
            stream);

    cuvs_index = std::move(new_index);
    computeCenterNorms_();
}

void CuvsIVFSQ::setCuvsIndex(cuvs::neighbors::ivf_sq::index<uint8_t>&& idx) {
    cuvs_index = std::make_shared<cuvs::neighbors::ivf_sq::index<uint8_t>>(
            std::move(idx));
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
    cuvs::neighbors::ivf_sq::search_params pams;
    pams.n_probes = nprobe;

    auto queries_view = raft::make_device_matrix_view<const float, idx_t>(
            queries.data(), (idx_t)numQueries, (idx_t)cols);
    auto out_inds_view = raft::make_device_matrix_view<idx_t, idx_t>(
            outIndices.data(), (idx_t)numQueries, (idx_t)k_);
    auto out_dists_view = raft::make_device_matrix_view<float, idx_t>(
            outDistances.data(), (idx_t)numQueries, (idx_t)k_);

    std::optional<cuvs::core::bitset<uint32_t, int64_t>> bitset_cuvs;
    std::optional<cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>>
            bitset_filter_cuvs;
    cuvs::neighbors::filtering::none_sample_filter none_filter;

    if (sel) {
        bitset_cuvs = cuvs::core::bitset<uint32_t, int64_t>(
                raft_handle, getBitsetSizeForFiltering_(), false);
        faiss::gpu::convert_to_bitset(resources_, *sel, bitset_cuvs->view());
        bitset_filter_cuvs.emplace(bitset_cuvs->view());
    }

    const cuvs::neighbors::filtering::base_filter& filter_ref = sel
            ? static_cast<const cuvs::neighbors::filtering::base_filter&>(
                      bitset_filter_cuvs.value())
            : static_cast<const cuvs::neighbors::filtering::base_filter&>(
                      none_filter);

    cuvs::neighbors::ivf_sq::search(
            raft_handle,
            pams,
            *cuvs_index,
            queries_view,
            out_inds_view,
            out_dists_view,
            filter_ref);

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
                if (!nan_flag[row]) {
                    return idx_t(-1);
                }
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

    cuvs::neighbors::ivf_sq::extend(
            raft_handle,
            raft::make_device_matrix_view<const float, idx_t>(
                    vecs.data(), n_rows_valid, dim_),
            std::make_optional<raft::device_vector_view<const idx_t, idx_t>>(
                    raft::make_device_vector_view<const idx_t, idx_t>(
                            indices.data(), n_rows_valid)),
            cuvs_index.get());

    return n_rows_valid;
}

idx_t CuvsIVFSQ::getListLength(idx_t listId) const {
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
            const_cast<idx_t**>(cuvs_index->inds_ptrs().data_handle()) + listId,
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
            const_cast<uint8_t**>(cuvs_index->data_ptrs().data_handle()) +
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

    // Must match our basic IVF parameters
    FAISS_THROW_IF_NOT(quantizer->d == getDim());
    FAISS_THROW_IF_NOT(quantizer->ntotal == getNumLists());

    size_t total_elems = quantizer->ntotal * quantizer->d;

    auto stream = resources_->getDefaultStreamCurrentDevice();
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    cuvs::neighbors::ivf_sq::index_params pams;
    pams.add_data_on_build = false;
    pams.metric = metricFaissToCuvs(metric_, false);
    pams.n_lists = numLists_;
    cuvs_index = std::make_shared<cuvs::neighbors::ivf_sq::index<uint8_t>>(
            raft_handle, pams, static_cast<uint32_t>(dim_));

    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
    if (gpuQ) {
        auto gpuData = gpuQ->getGpuData();

        if (gpuData->getUseFloat16()) {
            DeviceTensor<float, 2, true> centroids(
                    resources_,
                    makeSpaceAlloc(AllocType::FlatData, space_, stream),
                    {getNumLists(), getDim()});

            gpuData->reconstruct(0, gpuData->getSize(), centroids);

            raft::copy(
                    cuvs_index->centers().data_handle(),
                    centroids.data(),
                    total_elems,
                    stream);
        } else {
            auto centroids = gpuData->getVectorsFloat32Ref();

            raft::copy(
                    cuvs_index->centers().data_handle(),
                    centroids.data(),
                    total_elems,
                    stream);
        }
    } else {
        auto vecs = std::vector<float>(getNumLists() * getDim());
        quantizer->reconstruct_n(0, quantizer->ntotal, vecs.data());

        raft::update_device(
                cuvs_index->centers().data_handle(),
                vecs.data(),
                total_elems,
                stream);
    }

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
    auto& cuvsIndexLists = cuvs_index->lists();
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
        addEncodedVectorsToList_(
                i, ivf->get_codes(i), ivf->get_ids(i), listSize);
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

    for (idx_t listId = 0; listId < numLists_; ++listId) {
        idx_t listSize = getListLength(listId);
        if (listSize == 0) {
            continue;
        }

        auto ids = getListIndices(listId);
        auto codes = getListVectorData(listId, false);
        std::vector<float> decoded(listSize * dim_);
        faissSQ_->decode(codes.data(), decoded.data(), listSize);

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
            cuvs_index->data_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_device(
            listDataPtr, interleavedCodes.data(), gpuListSizeInBytes, stream);

    idx_t* listIndicesPtr;
    raft::update_host(
            &listIndicesPtr,
            cuvs_index->inds_ptrs().data_handle() + listId,
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
            cuvs_index->sq_vmin().data_handle(), vmin.data(), dim_, stream);
    raft::update_device(
            cuvs_index->sq_delta().data_handle(), delta.data(), dim_, stream);
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
            vmin.data(), cuvs_index->sq_vmin().data_handle(), dim_, stream);
    raft::update_host(
            delta.data(), cuvs_index->sq_delta().data_handle(), dim_, stream);
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
    cuvs_index->allocate_center_norms(raft_handle);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true, float, uint32_t>(
            cuvs_index->center_norms().value().data_handle(),
            cuvs_index->centers().data_handle(),
            cuvs_index->dim(),
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
    auto& lists = cuvs_index->lists();
    for (idx_t i = 0; i < numLists_; ++i) {
        dataPtrs[i] = lists[i] ? lists[i]->data_ptr() : nullptr;
        indsPtrs[i] = lists[i] ? lists[i]->indices_ptr() : nullptr;
    }

    raft::update_device(
            cuvs_index->data_ptrs().data_handle(),
            dataPtrs.data(),
            dataPtrs.size(),
            stream);
    raft::update_device(
            cuvs_index->inds_ptrs().data_handle(),
            indsPtrs.data(),
            indsPtrs.size(),
            stream);
    raft::update_device(
            cuvs_index->list_sizes().data_handle(),
            listSizes.data(),
            listSizes.size(),
            stream);

    auto sortedListSizes = listSizes;
    std::sort(
            sortedListSizes.begin(),
            sortedListSizes.end(),
            [](uint32_t a, uint32_t b) { return a > b; });

    auto accumSortedSizes = cuvs_index->accum_sorted_sizes();
    accumSortedSizes(0) = 0;
    for (size_t i = 0; i < sortedListSizes.size(); ++i) {
        accumSortedSizes(i + 1) = accumSortedSizes(i) + sortedListSizes[i];
    }
    raft_handle.sync_stream();
}

idx_t CuvsIVFSQ::getBitsetSizeForFiltering_() const {
    FAISS_ASSERT(cuvs_index != nullptr);

    idx_t maxId = -1;
    for (idx_t listId = 0; listId < numLists_; ++listId) {
        auto ids = getListIndices(listId);
        for (idx_t id : ids) {
            FAISS_THROW_IF_NOT_MSG(
                    id >= 0,
                    "cuVS IVF-SQ IDSelector filtering does not support "
                    "negative vector ids");
            maxId = std::max(maxId, id);
        }
    }

    if (maxId < 0) {
        return cuvs_index->size();
    }
    FAISS_THROW_IF_NOT_MSG(
            maxId < std::numeric_limits<idx_t>::max(),
            "cuVS IVF-SQ IDSelector filtering cannot represent the largest "
            "idx_t value");
    return std::max(cuvs_index->size(), maxId + 1);
}

std::vector<float> CuvsIVFSQ::getCentersHost_() const {
    FAISS_ASSERT(cuvs_index != nullptr);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    std::vector<float> centers(static_cast<size_t>(numLists_) * dim_);
    raft::update_host(
            centers.data(),
            cuvs_index->centers().data_handle(),
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
