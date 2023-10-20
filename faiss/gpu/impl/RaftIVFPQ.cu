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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/temporary_device_buffer.hpp>
#include <raft/neighbors/ivf_pq_codepacker.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_helpers.cuh>

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/RaftUtils.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/impl/IVFPQ.cuh>
#include <faiss/gpu/impl/RaftIVFPQ.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <limits>
#include <unordered_map>
#include "InterleavedCodes.h"

// #include <faiss/gpu/impl/RaftIVFPQ.cuh>

namespace faiss {
namespace gpu {

RaftIVFPQ::RaftIVFPQ(
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
                // skip ptr allocations in base class (handled by RAFT
                // internally) false,
                pqCentroidData,
                indicesOptions,
                space) {
    const raft::device_resources& raft_handle =
            resources->getRaftHandleCurrentDevice();

    raft::neighbors::ivf_pq::index_params pams;
    switch (metric) {
        case faiss::METRIC_L2:
            pams.metric = raft::distance::DistanceType::L2Expanded;
            break;
        case faiss::METRIC_INNER_PRODUCT:
            pams.metric = raft::distance::DistanceType::InnerProduct;
            break;
        default:
            FAISS_THROW_MSG("Metric is not supported.");
    }

    pams.codebook_kind = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
    pams.n_lists = nlist;
    pams.pq_bits = bitsPerSubQuantizer;
    pams.pq_dim = numSubQuantizers_;
    raft_knn_index.emplace(raft_handle, pams, static_cast<uint32_t>(dim));

    setPQCentroids_(pqCentroidData);
}

void RaftIVFPQ::reset() {
    raft_knn_index.reset();
}

RaftIVFPQ::~RaftIVFPQ() {}

void RaftIVFPQ::reserveMemory(idx_t numVecs) {
    fprintf(stderr,
            "WARN: NOP. Pre-allocation of IVF lists is not supported with RAFT enabled.");
}

void RaftIVFPQ::setPrecomputedCodes(Index* quantizer, bool enable) {
    fprintf(stderr,
            "WARN: NOP. Precompute codes is not supported with RAFT enabled.");
}

size_t RaftIVFPQ::reclaimMemory() {
    fprintf(stderr,
            "WARN: NOP. reclaimMemory is not supported with RAFT enabled.");
    return 0;
}

idx_t RaftIVFPQ::getListLength(idx_t listId) const {
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

void RaftIVFPQ::updateQuantizer(Index* quantizer) {
    printf("updateQuantizer from RaftIVFPQ was called\n");
    FAISS_THROW_IF_NOT(quantizer->is_trained);

    // Must match our basic IVF parameters
    FAISS_THROW_IF_NOT(quantizer->d == getDim());
    FAISS_THROW_IF_NOT(quantizer->ntotal == getNumLists());

    auto stream = resources_->getDefaultStreamCurrentDevice();
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    raft::neighbors::ivf_pq::helpers::make_rotation_matrix(
            raft_handle, &(raft_knn_index.value()), false);

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

            raft::neighbors::ivf_pq::helpers::set_centers(
                    raft_handle, &(raft_knn_index.value()), centroids.data());
        } else {
            /// No reconstruct needed since the centers are already in float32
            // The FlatIndex keeps its data in float32, so we can merely
            // reference it
            auto centroids = gpuData->getVectorsFloat32Ref();

            raft::neighbors::ivf_pq::helpers::set_centers(
                    raft_handle, &(raft_knn_index.value()), centroids.data());
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

        raft::neighbors::ivf_pq::helpers::set_centers(
                raft_handle, &(raft_knn_index.value()), centroids.data());
    }
}

/// Return the list indices of a particular list back to the CPU
std::vector<idx_t> RaftIVFPQ::getListIndices(idx_t listId) const {
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
            const_cast<idx_t**>(
                    raft_knn_index.value().inds_ptrs().data_handle()) +
                    listId,
            1,
            stream);
    raft_handle.sync_stream();

    raft::update_host(vec.data(), list_indices_ptr, listSize, stream);
    raft_handle.sync_stream();

    return vec;
}

/// Performs search when we are already given the IVF cells to look at
/// (GpuIndexIVF::search_preassigned implementation)
void RaftIVFPQ::searchPreassigned(
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

size_t RaftIVFPQ::getGpuListEncodingSize_(idx_t listId) const {
    return raft::neighbors::ivf_pq::helpers::get_list_size_in_bytes(
            &(raft_knn_index.value()), static_cast<uint32_t>(listId));
}

/// Return the encoded vectors of a particular list back to the CPU
std::vector<uint8_t> RaftIVFPQ::getListVectorData(idx_t listId, bool gpuFormat)
        const {
    if (gpuFormat) {
        FAISS_THROW_MSG(
                "gpuFormat should be false for RAFT indices. Unpacked codes are flat.");
    }
    FAISS_ASSERT(raft_knn_index.has_value());

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    auto stream = raft_handle.get_stream();

    idx_t listSize = getListLength(listId);

    // the interleaved block can be slightly larger than the list size (it's
    // rounded up)
    auto gpuListSizeInBytes = getGpuListEncodingSize_(listId);
    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(listSize);

//     std::vector<uint8_t> interleaved_codes(gpuListSizeInBytes);
    std::vector<uint8_t> flat_codes(cpuListSizeInBytes);

    uint8_t* list_data_ptr;

//     // fetch the list data ptr on host
//     raft::update_host(
//             &list_data_ptr,
//             const_cast<uint8_t**>(
//                     raft_knn_index.value().data_ptrs().data_handle()) +
//                     listId,
//             1,
//             stream);
//     raft_handle.sync_stream();

//     raft::update_host(
//             interleaved_codes.data(),
//             list_data_ptr,
//             gpuListSizeInBytes,
//             stream);
//     raft_handle.sync_stream();

    //     RaftIVFPQCodePackerInterleaved packer(
    //             (size_t)listSize, numSubQuantizers_, bitsPerSubQuantizer_);
    //     packer.unpack_all(interleaved_codes.data(), flat_codes.data());

    auto codes = raft::make_device_matrix<uint8_t, uint32_t>(
            raft_handle, listSize, numSubQuantizers_);

    raft::neighbors::ivf_pq::helpers::unpack_list_data(
            raft_handle, raft_knn_index.value(), codes.view(), listId, 0);

    raft::update_host(flat_codes.data(), codes.data_handle(), cpuListSizeInBytes, stream);
    raft_handle.sync_stream();

    return packNonInterleaved(
            std::move(flat_codes),
            listSize,
            numSubQuantizers_,
            bitsPerSubQuantizer_);
}

/// Find the approximate k nearest neighbors for `queries` against
/// our database
void RaftIVFPQ::search(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& queries,
        int nprobe,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    uint32_t numQueries = queries.getSize(0);
    uint32_t cols = queries.getSize(1);
    uint32_t k_ = k;

    printf("inside RaftIVFPQ's search\n");

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(numQueries > 0);
    FAISS_ASSERT(cols == dim_);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    raft::neighbors::ivf_pq::search_params pams;
    pams.n_probes = nprobe;

    auto queries_view = raft::make_device_matrix_view<const float, idx_t>(
            queries.data(), (idx_t)numQueries, (idx_t)cols);
    auto out_inds_view = raft::make_device_matrix_view<idx_t, idx_t>(
            outIndices.data(), (idx_t)numQueries, (idx_t)k_);
    auto out_dists_view = raft::make_device_matrix_view<float, idx_t>(
            outDistances.data(), (idx_t)numQueries, (idx_t)k_);

    raft::print_host_vector(
            "accum_sorted_sizes",
            raft_knn_index.value().accum_sorted_sizes().data_handle(),
            raft_knn_index.value().n_lists(),
            std::cout);
    raft::print_device_vector(
            "raft_list_sizes",
            raft_knn_index.value().list_sizes().data_handle(),
            raft_knn_index.value().n_lists(),
            std::cout);

    raft::neighbors::ivf_pq::search<float, idx_t>(
            raft_handle,
            pams,
            raft_knn_index.value(),
            queries_view,
            out_inds_view,
            out_dists_view);
    raft_handle.sync_stream();

    /// Identify NaN rows and mask their nearest neighbors
    auto nan_flag = raft::make_device_vector<bool>(raft_handle, numQueries);

    thrust::fill_n(
            raft_handle.get_thrust_policy(),
            nan_flag.data_handle(),
            numQueries,
            true);

    //     validRowIndices(resources_, queries, nan_flag.data_handle());

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

idx_t RaftIVFPQ::addVectors(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
    idx_t n_rows = vecs.getSize(0);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    /// Remove NaN values
    auto nan_flag = raft::make_device_vector<bool, idx_t>(raft_handle, n_rows);

    //     validRowIndices(resources_, vecs, nan_flag.data_handle());
    thrust::fill_n(
            raft_handle.get_thrust_policy(),
            nan_flag.data_handle(),
            n_rows,
            true);

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
    raft_knn_index.emplace(raft::neighbors::ivf_pq::extend(
            raft_handle,
            raft::make_device_matrix_view<const float, idx_t>(
                    vecs.data(), n_rows_valid, dim_),
            std::make_optional<raft::device_vector_view<const idx_t, idx_t>>(
                    raft::make_device_vector_view<const idx_t, idx_t>(
                            indices.data(), n_rows_valid)),
            raft_knn_index.value()));

    return n_rows_valid;
}

void RaftIVFPQ::copyInvertedListsFrom(const InvertedLists* ivf) {
    printf("inside RaftIVFPQ copyInvertedListsFrom\n");
    size_t nlist = ivf ? ivf->nlist : 0;
    size_t ntotal = ivf ? ivf->compute_ntotal() : 0;

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    std::vector<uint32_t> list_sizes_(nlist);
    std::vector<idx_t> indices_(ntotal);

    // the index must already exist
    FAISS_ASSERT(raft_knn_index.has_value());

    auto& raft_lists = raft_knn_index.value().lists();

    // conservative memory alloc for cloning cpu inverted lists
    raft::neighbors::ivf_pq::list_spec<uint32_t, idx_t> raft_list_spec{
            static_cast<uint32_t>(numSubQuantizers_),
            static_cast<uint32_t>(bitsPerSubQuantizer_),
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

        // This RAFT list must currently be empty
        FAISS_ASSERT(getListLength(i) == 0);

        raft::neighbors::ivf::resize_list(
                raft_handle,
                raft_lists[i],
                raft_list_spec,
                static_cast<uint32_t>(listSize),
                static_cast<uint32_t>(0));
    }

    raft::update_device(
            raft_knn_index.value().list_sizes().data_handle(),
            list_sizes_.data(),
            nlist,
            raft_handle.get_stream());

    //     Update the pointers and the sizes
    raft::neighbors::ivf_pq::helpers::recompute_internal_state(
            raft_handle, &(raft_knn_index.value()));

    for (size_t i = 0; i < nlist; ++i) {
        size_t listSize = ivf->list_size(i);
        addEncodedVectorsToList_(
                i, ivf->get_codes(i), ivf->get_ids(i), listSize);
    }
}

void RaftIVFPQ::setRaftIndex(
        std::optional<raft::neighbors::ivf_pq::index<idx_t>>& idx) {
    raft_knn_index = std::move(idx);
}

void RaftIVFPQ::addEncodedVectorsToList_(
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
    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

    // We only have int32 length representations on the GPU per each
    // list; the length is in sizeof(char)
    FAISS_ASSERT(gpuListSizeInBytes <= (size_t)std::numeric_limits<int>::max());

    std::vector<uint8_t> interleaved_codes(gpuListSizeInBytes);
    {
        // Translate the codes as needed to our preferred form
        std::vector<uint8_t> codesV(cpuListSizeInBytes);
        std::memcpy(codesV.data(), codes, cpuListSizeInBytes);

        std::vector<uint8_t> up = unpackNonInterleaved(
                std::move(codesV),
                numVecs,
                numSubQuantizers_,
                bitsPerSubQuantizer_);

        for (int i = 0; i < numVecs * numSubQuantizers_; i++) {
            printf("%d ", int(up[i]));
        }

        // RaftIVFPQCodePackerInterleaved packer(
        //         (size_t)numVecs, numSubQuantizers_, bitsPerSubQuantizer_);

        // packer.pack_all(
        //         reinterpret_cast<const uint8_t*>(up.data()),
        //         interleaved_codes.data());

        auto codes = raft::make_device_matrix<uint8_t, uint32_t>(
                raft_handle, numVecs, numSubQuantizers_);
        raft::update_device(codes.data_handle(), up.data(), up.size(), stream);

        printf("Flat codes updated\n");
        raft::neighbors::ivf_pq::helpers::pack_list_data(
                raft_handle,
                &(raft_knn_index.value()),
                codes.view(),
                listId,
                0);

        raft::update_host(
                interleaved_codes.data(),
                raft_knn_index.value().lists()[listId]->data.data_handle(),
                gpuListSizeInBytes,
                stream);
        raft_handle.sync_stream();

        printf("\n now printing interleaved_codes\n");

        // raft::print_device_vector("interleaved_codes",
        // raft_knn_index.value().lists()[listId]->data.data_handle(), 32 *
        // bitsPerSubQuantizer_, std::cout);

        // if (listId == 0) {
        //     for (int i = 0; i < 32 * numSubQuantizers_; i++) {
        //         printf("%d ", int(interleaved_codes[i]));
        //     }
        // }
        // raft::print_host_vector("interleaved_codes",
        // interleaved_codes.data(), 10, std::cout);
    }

    uint8_t* list_data_ptr;

    /// fetch the list data ptr on host
    //     raft::update_host(
    //             &list_data_ptr,
    //             raft_knn_index.value().data_ptrs().data_handle() + listId,
    //             1,
    //             stream);
    //     raft_handle.sync_stream();

    //     raft::update_device(
    //             list_data_ptr,
    //             interleaved_codes.data(),
    //             gpuListSizeInBytes,
    //             stream);

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

void RaftIVFPQ::setPQCentroids_(float* data) {
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    auto data_buf = raft::make_temporary_device_buffer<float>(
            raft_handle,
            data,
            raft::make_extents<uint32_t>(
                    numSubQuantizers_,
                    numSubQuantizerCodes_,
                    dimPerSubQuantizer_));

    raft::neighbors::ivf_pq::helpers::set_pq_centers(
            raft_handle,
            &(raft_knn_index.value()),
            data_buf.view().data_handle());
}

RaftIVFPQCodePackerInterleaved::RaftIVFPQCodePackerInterleaved(
        size_t list_size,
        int numSubQuantizers,
        int bitsPerSubQuantizer) {
    this->bitsPerSubQuantizer_ = bitsPerSubQuantizer;
    nvec = list_size;
    code_size = numSubQuantizers;
}

void RaftIVFPQCodePackerInterleaved::pack_1(
        const uint8_t* flat_code,
        size_t offset,
        uint8_t* block) const {
    switch (bitsPerSubQuantizer_) {
        case 4:
            raft::neighbors::ivf_pq::codepacker::pack_1<4>(
                    flat_code,
                    block,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        case 5:
            raft::neighbors::ivf_pq::codepacker::pack_1<5>(
                    flat_code,
                    block,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        case 6:
            raft::neighbors::ivf_pq::codepacker::pack_1<6>(
                    flat_code,
                    block,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        case 7:
            raft::neighbors::ivf_pq::codepacker::pack_1<7>(
                    flat_code,
                    block,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        case 8:
            raft::neighbors::ivf_pq::codepacker::pack_1<8>(
                    flat_code,
                    block,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        default:
            FAISS_THROW_FMT(
                    "Invalid bits per sub quantizer (%d), the value must be within [4, 8]",
                    bitsPerSubQuantizer_);
    }
}

void RaftIVFPQCodePackerInterleaved::unpack_1(
        const uint8_t* block,
        size_t offset,
        uint8_t* flat_code) const {
    switch (bitsPerSubQuantizer_) {
        case 4:
            raft::neighbors::ivf_pq::codepacker::unpack_1<4>(
                    block,
                    flat_code,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        case 5:
            raft::neighbors::ivf_pq::codepacker::unpack_1<5>(
                    block,
                    flat_code,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        case 6:
            raft::neighbors::ivf_pq::codepacker::unpack_1<6>(
                    block,
                    flat_code,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        case 7:
            raft::neighbors::ivf_pq::codepacker::unpack_1<7>(
                    block,
                    flat_code,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        case 8:
            raft::neighbors::ivf_pq::codepacker::unpack_1<8>(
                    block,
                    flat_code,
                    static_cast<uint32_t>(code_size),
                    static_cast<uint32_t>(offset));
            break;
        default:
            FAISS_THROW_FMT(
                    "Invalid bits per sub quantizer (%d), the value must be within [4, 8]",
                    bitsPerSubQuantizer_);
    }
}

} // namespace gpu
} // namespace faiss