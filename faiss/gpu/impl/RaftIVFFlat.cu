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
#include <raft/core/handle.hpp>
#include <cstdint>
#include <raft/neighbors/ivf_flat.cuh>

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

#include <raft/core/logger.hpp>

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
                  space) {}

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
    printf("Inside RaftIVFFlat search()\n");

    // TODO: We probably don't want to ignore the coarse quantizer here...

    std::uint32_t n = queries.getSize(0);
    std::uint32_t cols = queries.getSize(1);
    std::uint32_t k_ = k;

    // Device is already set in GpuIndex::search
    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= numLists_);

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();
    raft::neighbors::ivf_flat::search_params pams;
    pams.n_probes = nprobe;

    auto queries_view =
            raft::make_device_matrix_view<const float>(queries.data(), n, cols);
    auto out_inds_view =
            raft::make_device_matrix_view<idx_t>(outIndices.data(), n, k_);
    auto out_dists_view =
            raft::make_device_matrix_view<float>(outDistances.data(), n, k_);
    raft::neighbors::ivf_flat::search<float, idx_t>(
            raft_handle,
	    pams,
            *raft_knn_index,
            queries_view,
            out_inds_view,
            out_dists_view);

    raft_handle.sync_stream();
}

/// Classify and encode/add vectors to our IVF lists.
/// The input data must be on our current device.
/// Returns the number of vectors successfully added. Vectors may
/// not be able to be added because they contain NaNs.
idx_t RaftIVFFlat::addVectors(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
    printf("Inside RaftIVFFlat addVectors()\n");

    raft::print_device_vector("add_vectors", vecs.data(), 50, std::cout);

    auto vecs_view = raft::make_device_matrix_view<const float, idx_t>(
            vecs.data(), vecs.getSize(0), dim_);
    auto inds_view = raft::make_device_vector_view<const idx_t, idx_t>(
            indices.data(), (idx_t)indices.getSize(0));

    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    printf("About to call extend on index\n");
    // TODO: We probably don't want to ignore the coarse quantizer here

    if (raft_knn_index.has_value()) {
        raft_knn_index.emplace(raft::neighbors::ivf_flat::extend(
                raft_handle,
                vecs_view,
                std::make_optional<
                        raft::device_vector_view<const idx_t, idx_t>>(
                        inds_view),
	        raft_knn_index.value()));

    } else {
        printf("Index has not been trained!\n");
    }
    printf("Done.\n");
    return vecs.getSize(0);
}

void RaftIVFFlat::reset() {
    printf("Inside RaftIVFFlat reset()\n");
    raft_knn_index.reset();
}

idx_t RaftIVFFlat::getListLength(idx_t listId) const {
    printf("Inside RaftIVFFlat getListLength\n");

    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    uint32_t size;
    raft::copy(
            &size,
            raft_knn_index.value().list_sizes().data_handle() + listId,
            1,
            raft_handle.get_stream());
    raft_handle.sync_stream();
    return int(size);
}

/// Return the list indices of a particular list back to the CPU
std::vector<idx_t> RaftIVFFlat::getListIndices(idx_t listId) const {
    printf("Inside RaftIVFFlat getListIndices\n");

    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    uint32_t size;

    raft::copy(
            &size,
            raft_knn_index.value().list_sizes().data_handle() + listId,
            1,
            raft_handle.get_stream());
    raft_handle.sync_stream();

    std::vector<idx_t> vec(size);
    raft::copy(
            vec.data(),
            *(raft_knn_index.value().inds_ptrs().data_handle() + listId),
            size,
            raft_handle.get_stream());
    return vec;
}

/// Return the encoded vectors of a particular list back to the CPU
std::vector<uint8_t> RaftIVFFlat::getListVectorData(idx_t listId, bool gpuFormat)
        const {
    printf("Inside RaftIVFFlat getListVectorData\n");

    FAISS_ASSERT(raft_knn_index.has_value());
    const raft::device_resources& raft_handle =
            resources_->getRaftHandleCurrentDevice();

    std::cout << "Calling getListVectorData for " << listId << std::endl;

    using elem_t = decltype(raft_knn_index.value().data_ptrs())::element_type;
    size_t dim = raft_knn_index.value().dim();
    uint32_t list_size;
    
    raft::copy(&list_size, raft_knn_index.value().list_sizes().data_handle() + listId, 1, raft_handle.get_stream());


    // the interleaved block can be slightly larger than the list size (it's
    // rounded up)
    size_t byte_size = size_t(list_size) * sizeof(elem_t) * dim;
    std::vector<uint8_t> vec(byte_size);
    raft::copy(
            vec.data(),
            reinterpret_cast<const uint8_t*>(
                    raft_knn_index.value().data_ptrs().data_handle()+listId),
            byte_size,
            raft_handle.get_stream());
    return vec;
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
    printf("Inside RaftIVFFlat searchPreassigned\n");

    // TODO: Fill this in!
}

void RaftIVFFlat::updateQuantizer(Index* quantizer) {
    idx_t quantizer_ntotal = quantizer->ntotal;

    std::cout << "Calling RAFT updateQuantizer with trained index with "
              << quantizer_ntotal << " items" << std::endl;
    const raft::device_resources& handle = resources_->getRaftHandleCurrentDevice();
    auto stream = handle.get_stream();

    auto total_elems = size_t(quantizer_ntotal) * size_t(quantizer->d);

    raft::logger::get().set_level(RAFT_LEVEL_TRACE);

    raft::neighbors::ivf_flat::index_params pams;
    pams.add_data_on_build = false;

    switch (this->metric_) {
        case faiss::METRIC_L2:
            printf("Using L2!\n");
            pams.metric = raft::distance::DistanceType::L2Expanded;
            break;
        case faiss::METRIC_INNER_PRODUCT:
            printf("Using Inner product!\n");
            pams.metric = raft::distance::DistanceType::InnerProduct;
            break;
        default:
            FAISS_THROW_MSG("Metric is not supported.");
    }

    raft_knn_index.emplace(
            handle,
            pams.metric,
            (uint32_t)this->numLists_,
            false,
	    false,
            (uint32_t)this->dim_);

    printf("Reconstructing\n");
    // Copy (reconstructed) centroids over, rather than re-training
    rmm::device_uvector<float> buf_dev(total_elems, stream);
    std::vector<float> buf_host(total_elems);
    quantizer->reconstruct_n(0, quantizer_ntotal, buf_host.data());

    printf("Copying...\n");

    raft::update_device(
            raft_knn_index.value().centers().data_handle(),
            buf_host.data(),
            total_elems,
            stream);

    raft::print_device_vector(
            "raft centers",
            raft_knn_index.value().centers().data_handle(),
            this->dim_,
            std::cout);
}

//
//
// void RaftIVFFlat::copyInvertedListsFrom(const InvertedLists* ivf) {
//    size_t nlist = ivf ? ivf->nlist : 0;
//    size_t ntotal = ivf ? ivf->compute_ntotal() : 0;
//
//    printf("Inside RAFT copyInvertedListsFrom\n");
//    raft::device_resources &handle = resources_->getRaftHandleCurrentDevice();
//    // We need to allocate the IVF
//    printf("nlist=%ld, ntotal=%ld\n", nlist, ntotal);
//
//    std::vector<std::uint32_t> list_sizes_(nlist);
//    std::vector<Index::idx_t> list_offsets_(nlist+1);
//    std::vector<Index::idx_t> indices_(ntotal);
//
//    raft::neighbors::ivf_flat::index_params raft_idx_params;
//    raft_idx_params.n_lists = nlist;
//    raft_idx_params.metric = raft::distance::DistanceType::L2Expanded;
//    raft_idx_params.add_data_on_build = false;
//    raft_idx_params.kmeans_n_iters = 100;
//
//    raft_knn_index.emplace(handle, raft_idx_params, dim_);
//    raft_knn_index.value().allocate(handle, ntotal, true);
//
//    for (size_t i = 0; i < nlist; ++i) {
//        size_t listSize = ivf->list_size(i);
//
//        // GPU index can only support max int entries per list
//        FAISS_THROW_IF_NOT_FMT(
//                listSize <= (size_t)std::numeric_limits<int>::max(),
//                "GPU inverted list can only support "
//                "%zu entries; %zu found",
//                (size_t)std::numeric_limits<int>::max(),
//                listSize);
//
//        addEncodedVectorsToList_(
//                i, ivf->get_codes(i), ivf->get_ids(i), listSize);
//    }
//
//    raft::update_device(raft_knn_index.value().list_sizes().data_handle(),
//    list_sizes_.data(), nlist, handle.get_stream());
//    raft::update_device(raft_knn_index.value().list_offsets().data_handle(),
//    list_offsets_.data(), nlist+1, handle.get_stream());
//
//}

// void RaftIVFFlat::addEncodedVectorsToList_(
//        int listId,
//        const void* codes,
//        const Index::idx_t* indices,
//        size_t numVecs) {
//    auto stream = resources_->getDefaultStreamCurrentDevice();
//
//    // This list must already exist
////    FAISS_ASSERT(listId < deviceListData_.size());
//
//    // This list must currently be empty
////    auto& listCodes = deviceListData_[listId];
////    FAISS_ASSERT(listCodes->data.size() == 0);
////    FAISS_ASSERT(listCodes->numVecs == 0);
//
//    // If there's nothing to add, then there's nothing we have to do
//    if (numVecs == 0) {
//        return;
//    }
//
//    // The GPU might have a different layout of the memory
//    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
//    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);
//
//    // We only have int32 length representations on the GPU per each
//    // list; the length is in sizeof(char)
//    FAISS_ASSERT(gpuListSizeInBytes <=
//    (size_t)std::numeric_limits<int>::max());
//
//    // Translate the codes as needed to our preferred form
//    std::vector<uint8_t> codesV(cpuListSizeInBytes);
//    std::memcpy(codesV.data(), codes, cpuListSizeInBytes);
//    auto translatedCodes = translateCodesToGpu_(std::move(codesV), numVecs);
//
//    std::cout << "numVecs=" << numVecs << "gpuListSizeInBytes=" <<
//    gpuListSizeInBytes << std::endl;
//
////
/// RAFT_CUDA_TRY(cudaMemcpyAsync(raft_knn_index.value().data().data_handle()+(),
/// translatedCodes.data(), ))
//
////    listCodes->data.append(
////            translatedCodes.data(),
////            gpuListSizeInBytes,
////            stream,
////            true /* exact reserved size */);
////    listCodes->numVecs = numVecs;
////
////    // Handle the indices as well
////    addIndicesFromCpu_(listId, indices, numVecs);
////
//
//      // We should problay consider using this...
////    deviceListDataPointers_.setAt(
////            listId, (void*)listCodes->data.data(), stream);
////    deviceListLengths_.setAt(listId, (int)numVecs, stream);
////
////    // We update this as well, since the multi-pass algorithm uses it
////    maxListLength_ = std::max(maxListLength_, (int)numVecs);
//}

///// Copy all inverted lists from ourselves to a CPU representation
// void RaftIVFFlat::copyInvertedListsTo(InvertedLists* ivf) {
//    printf("Inside RaftIVFFlat copyInvertedListsTo\n");
//
//    // TODO: Need to replicate copyInvertedListsTo() in IVFBase.cu
//}

} // namespace gpu
} // namespace faiss
