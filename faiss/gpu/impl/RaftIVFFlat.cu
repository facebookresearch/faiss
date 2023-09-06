/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/impl/RaftIVFFlat.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <unordered_map>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/neighbors/ivf_flat_codepacker.hpp>
#include <raft/neighbors/ivf_flat.cuh>

namespace faiss {
namespace gpu {

RaftIVFFlat::RaftIVFFlat(
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
                  space) {
    FAISS_THROW_IF_NOT_MSG(
            indicesOptions == INDICES_64_BIT,
            "only INDICES_64_BIT is supported for RAFT index");
    reset();
}

RaftIVFFlat::~RaftIVFFlat() {}

/// Find the approximate k nearest neighbors for `queries` against
/// our database
void RaftIVFFlat::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    // TODO: We probably don't want to ignore the coarse quantizer here...

    uint32_t numQueries = queries.getSize(0);
    uint32_t cols = queries.getSize(1);
    uint32_t k_ = k;

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    raft::neighbors::ivf_flat::search_params pams;
    pams.n_probes = nprobe;

    auto queries_view = raft::make_device_matrix_view<const float, idx_t>(
            queries.data(), (idx_t)numQueries, (idx_t)cols);
    auto out_inds_view = raft::make_device_matrix_view<idx_t, idx_t>(
            outIndices.data(), (idx_t)numQueries, (idx_t)k_);
    auto out_dists_view = raft::make_device_matrix_view<float, idx_t>(
            outDistances.data(), (idx_t)numQueries, (idx_t)k_);

    raft::neighbors::ivf_flat::search<float, idx_t>(
            raft_handle,
            pams,
            raft_knn_index.value(),
            queries_view,
            out_inds_view,
            out_dists_view);

    /// Identify NaN rows and mask their nearest neighbors
    auto nan_flag = raft::make_device_vector<bool>(raft_handle, numQueries);

    validRowIndices_(queries, nan_flag.data_handle());

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

/// Classify and encode/add vectors to our IVF lists.
/// The input data must be on our current device.
/// Returns the number of vectors successfully added. Vectors may
/// not be able to be added because they contain NaNs.
idx_t RaftIVFFlat::addVectors(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
    /// TODO: We probably don't want to ignore the coarse quantizer here

    idx_t n_rows = vecs.getSize(0);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    /// Remove NaN values
    auto nan_flag = raft::make_device_vector<bool, idx_t>(raft_handle, n_rows);

    validRowIndices_(vecs, nan_flag.data_handle());

    idx_t n_rows_valid = thrust::reduce(
            raft_handle.get_thrust_policy(),
            nan_flag.data_handle(),
            nan_flag.data_handle() + n_rows,
            0);

    if (n_rows_valid < n_rows) {
        auto gather_indices = raft::make_device_vector<idx_t, idx_t>(
                raft_handle, n_rows_valid);

        auto count = thrust::make_counting_iterator(0);

        thrust::copy_if(
                raft_handle.get_thrust_policy(),
                count,
                count + n_rows,
                gather_indices.data_handle(),
                [nan_flag = nan_flag.data_handle()] __device__(auto i) {
                    return nan_flag[i];
                });

        raft::matrix::gather(
                raft_handle,
                raft::make_device_matrix_view<float, idx_t>(
                        vecs.data(), n_rows, dim_),
                raft::make_const_mdspan(gather_indices.view()),
                (idx_t)16);

        auto valid_indices = raft::make_device_vector<idx_t, idx_t>(
                raft_handle, n_rows_valid);

        raft::matrix::gather(
                raft_handle,
                raft::make_device_matrix_view<idx_t>(
                        indices.data(), n_rows, (idx_t)1),
                raft::make_const_mdspan(gather_indices.view()));
    }

    FAISS_ASSERT(raft_knn_index.has_value());
    raft_knn_index.emplace(raft::neighbors::ivf_flat::extend(
            raft_handle,
            raft::make_device_matrix_view<const float, idx_t>(
                    vecs.data(), n_rows_valid, dim_),
            std::make_optional<raft::device_vector_view<const idx_t, idx_t>>(
                    raft::make_device_vector_view<const idx_t, idx_t>(
                            indices.data(), n_rows_valid)),
            raft_knn_index.value()));

    return n_rows_valid;
}

void RaftIVFFlat::reset() {
    raft_knn_index.reset();
}

idx_t RaftIVFFlat::getListLength(idx_t listId) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    uint32_t size;
    raft::update_host(
            &size,
            raft_knn_index.value().list_sizes().data_handle() + listId,
            1,
            raft_handle.get_stream());
    raft_handle.sync_stream();

    return static_cast<int>(size);
}

/// Return the list indices of a particular list back to the CPU
std::vector<idx_t> RaftIVFFlat::getListIndices(idx_t listId) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    idx_t listSize = getListLength(listId);

    std::vector<idx_t> vec(listSize);

    // fetch the list indices ptr on host
    idx_t* list_indices_ptr;

    // fetch the list indices ptr on host
    raft::update_host(
            &list_indices_ptr,
            raft_knn_index.value().inds_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_host(vec.data(), list_indices_ptr, listSize, stream);
    raft_handle.sync_stream();

    return vec;
}

/// Return the encoded vectors of a particular list back to the CPU
std::vector<uint8_t> RaftIVFFlat::getListVectorData(
        idx_t listId,
        bool gpuFormat) const {
    if (gpuFormat) {
        FAISS_THROW_MSG("gpuFormat is not suppported for raft indices");
    }
    FAISS_ASSERT(raft_knn_index.has_value());

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
            raft_knn_index.value().data_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_host(
            interleaved_codes.data(),
            reinterpret_cast<uint8_t*>(list_data_ptr),
            gpuListSizeInBytes,
            stream);
    raft_handle.sync_stream();

    RaftIVFFlatCodePackerInterleaved packer(
            (size_t)listSize, dim_, raft_knn_index.value().veclen());
    packer.unpack_all(interleaved_codes.data(), flat_codes.data());
    return flat_codes;
}

/// Performs search when we are already given the IVF cells to look at
/// (GpuIndexIVF::search_preassigned implementation)
void RaftIVFFlat::searchPreassigned(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& ivfDistances,
        Tensor<idx_t, 2, true>& ivfAssignments,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool storePairs) {
    // TODO: Fill this in!
}

void RaftIVFFlat::updateQuantizer(Index* quantizer) {
    idx_t quantizer_ntotal = quantizer->ntotal;

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    auto total_elems = size_t(quantizer_ntotal) * size_t(quantizer->d);

    raft::logger::get().set_level(RAFT_LEVEL_TRACE);

    raft::neighbors::ivf_flat::index_params pams;
    pams.add_data_on_build = false;

    pams.n_lists = this->numLists_;

    switch (this->metric_) {
        case faiss::METRIC_L2:
            pams.metric = raft::distance::DistanceType::L2Expanded;
            break;
        case faiss::METRIC_INNER_PRODUCT:
            pams.metric = raft::distance::DistanceType::InnerProduct;
            break;
        default:
            FAISS_THROW_MSG("Metric is not supported.");
    }

    raft_knn_index.emplace(raft_handle, pams, (uint32_t)this->dim_);

    cudaMemsetAsync(
            raft_knn_index.value().list_sizes().data_handle(),
            0,
            raft_knn_index.value().list_sizes().size() * sizeof(uint32_t),
            stream);
    cudaMemsetAsync(
            raft_knn_index.value().data_ptrs().data_handle(),
            0,
            raft_knn_index.value().data_ptrs().size() * sizeof(float*),
            stream);
    cudaMemsetAsync(
            raft_knn_index.value().inds_ptrs().data_handle(),
            0,
            raft_knn_index.value().inds_ptrs().size() * sizeof(idx_t*),
            stream);

    /// Copy (reconstructed) centroids over, rather than re-training
    std::vector<float> buf_host(total_elems);
    quantizer->reconstruct_n(0, quantizer_ntotal, buf_host.data());

    raft::update_device(
            raft_knn_index.value().centers().data_handle(),
            buf_host.data(),
            total_elems,
            stream);
}

void RaftIVFFlat::copyInvertedListsFrom(const InvertedLists* ivf) {
    size_t nlist = ivf ? ivf->nlist : 0;
    size_t ntotal = ivf ? ivf->compute_ntotal() : 0;

    raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    std::vector<uint32_t> list_sizes_(nlist);
    std::vector<idx_t> indices_(ntotal);

    // the index must already exist
    FAISS_ASSERT(raft_knn_index.has_value());

    auto& raft_lists = raft_knn_index.value().lists();

    // conservative memory alloc for cloning cpu inverted lists
    raft::neighbors::ivf_flat::list_spec<uint32_t, float, idx_t> raft_list_spec{
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

        raft::neighbors::ivf::resize_list(
                raft_handle,
                raft_lists[i],
                raft_list_spec,
                (uint32_t)listSize,
                (uint32_t)0);
    }

    // Update the pointers and the sizes
    raft_knn_index.value().recompute_internal_state(raft_handle);

    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        addEncodedVectorsToList_(
                i, ivf->get_codes(i), ivf->get_ids(i), listSize);
    }

    raft::update_device(
            raft_knn_index.value().list_sizes().data_handle(),
            list_sizes_.data(),
            nlist,
            raft_handle.get_stream());

    // Precompute the centers vector norms for L2Expanded distance
    if (this->metric_ == faiss::METRIC_L2) {
        raft_knn_index.value().allocate_center_norms(raft_handle);
        raft::linalg::rowNorm(
                raft_knn_index.value().center_norms().value().data_handle(),
                raft_knn_index.value().centers().data_handle(),
                raft_knn_index.value().dim(),
                (uint32_t)nlist,
                raft::linalg::L2Norm,
                true,
                raft_handle.get_stream());
    }
}

size_t RaftIVFFlat::getGpuVectorsEncodingSize_(idx_t numVecs) const {
    idx_t bits = 32 /* float */;

    // bytes to encode a block of 32 vectors (single dimension)
    idx_t bytesPerDimBlock = bits * 32 / 8; // = 128

    // bytes to fully encode 32 vectors
    idx_t bytesPerBlock = bytesPerDimBlock * dim_;

    // number of blocks of 32 vectors we have
    idx_t numBlocks =
            utils::divUp(numVecs, raft::neighbors::ivf_flat::kIndexGroupSize);

    // total size to encode numVecs
    return bytesPerBlock * numBlocks;
}

void RaftIVFFlat::addEncodedVectorsToList_(
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // This list must already exist
    FAISS_ASSERT(raft_knn_index.has_value());

    // This list must currently be empty
    FAISS_ASSERT(getListLength(listId) == 0);

    // If there's nothing to add, then there's nothing we have to do
    if (numVecs == 0) {
        return;
    }

    // The GPU might have a different layout of the memory
    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

    // We only have int32 length representations on the GPU per each
    // list; the length is in sizeof(char)
    FAISS_ASSERT(gpuListSizeInBytes <= (size_t)std::numeric_limits<int>::max());

    std::vector<uint8_t> interleaved_codes(gpuListSizeInBytes);
    RaftIVFFlatCodePackerInterleaved packer(
            (size_t)numVecs, (uint32_t)dim_, raft_knn_index.value().veclen());

    packer.pack_all(
            reinterpret_cast<const uint8_t*>(codes), interleaved_codes.data());

    float* list_data_ptr;
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    /// fetch the list data ptr on host
    raft::update_host(
            &list_data_ptr,
            raft_knn_index.value().data_ptrs().data_handle() + listId,
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
            raft_knn_index.value().inds_ptrs().data_handle() + listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_device(list_indices_ptr, indices, numVecs, stream);
}

void RaftIVFFlat::validRowIndices_(
        Tensor<float, 2, true>& vecs,
        bool* nan_flag) {
    raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    idx_t n_rows = vecs.getSize(0);

    thrust::fill_n(raft_handle.get_thrust_policy(), nan_flag, n_rows, true);
    raft::linalg::map_offset(
            raft_handle,
            raft::make_device_vector_view<bool, idx_t>(nan_flag, n_rows),
            [vecs = vecs.data(), dim_ = this->dim_] __device__(idx_t i) {
                for (idx_t col = 0; col < dim_; col++) {
                    if (!isfinite(vecs[i * dim_ + col])) {
                        return false;
                    }
                }
                return true;
            });
}

RaftIVFFlatCodePackerInterleaved::RaftIVFFlatCodePackerInterleaved(
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
            utils::roundUp(nvec, raft::neighbors::ivf_flat::kIndexGroupSize);
}

void RaftIVFFlatCodePackerInterleaved::pack_1(
        const uint8_t* flat_code,
        size_t offset,
        uint8_t* block) const {
    raft::neighbors::ivf_flat::codepacker::pack_1(
            reinterpret_cast<const uint32_t*>(flat_code),
            reinterpret_cast<uint32_t*>(block),
            dim,
            chunk_size,
            static_cast<uint32_t>(offset));
}

void RaftIVFFlatCodePackerInterleaved::unpack_1(
        const uint8_t* block,
        size_t offset,
        uint8_t* flat_code) const {
    raft::neighbors::ivf_flat::codepacker::unpack_1(
            reinterpret_cast<const uint32_t*>(block),
            reinterpret_cast<uint32_t*>(flat_code),
            dim,
            chunk_size,
            static_cast<uint32_t>(offset));
}

} // namespace gpu
} // namespace faiss
