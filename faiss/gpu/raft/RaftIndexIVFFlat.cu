/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/raft/RaftIndexIVFFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Float16.cuh>

#include <raft/spatial/knn/ivf_flat.cuh>

#include <limits>

namespace faiss {
namespace gpu {

RaftIndexIVFFlat::RaftIndexIVFFlat(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFFlat* index,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVFFlat(
        provider,
        index,
        config), raft_handle(resources_->getDefaultStream(config_.device)) {

    copyFrom(index);
}

RaftIndexIVFFlat::RaftIndexIVFFlat(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVFFlat(provider, dims, nlist, metric, config),
          raft_handle(resources_->getDefaultStream(config_.device)) {

    this->is_trained = false;
}

RaftIndexIVFFlat::~RaftIndexIVFFlat() {}

void RaftIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat* index) {

    printf("Copying from...\n");

    // TODO: Need to copy necessary memory from the index and set any needed params.
    DeviceScope scope(config_.device);

    GpuIndex::copyFrom(index);

    FAISS_ASSERT(index->nlist > 0);
    FAISS_THROW_IF_NOT_FMT(
            index->nlist <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports %zu inverted lists",
            (size_t)std::numeric_limits<int>::max());
    nlist = index->nlist;

    FAISS_THROW_IF_NOT_FMT(
            index->nprobe > 0 && index->nprobe <= getMaxKSelection(),
            "GPU index only supports nprobe <= %zu; passed %zu",
            (size_t)getMaxKSelection(),
            index->nprobe);
    nprobe = index->nprobe;

    FAISS_ASSERT(metric_type != faiss::METRIC_L2 &&
                 metric_type != faiss::METRIC_INNER_PRODUCT);

    if (!index->is_trained) {
        // copied in GpuIndex::copyFrom
        FAISS_ASSERT(!is_trained && ntotal == 0);
        return;
    }

    // copied in GpuIndex::copyFrom
    // ntotal can exceed max int, but the number of vectors per inverted
    // list cannot exceed this. We check this in the subclasses.
    FAISS_ASSERT(is_trained && (ntotal == index->ntotal));

    // Since we're trained, the quantizer must have data
    FAISS_ASSERT(index->quantizer->ntotal > 0);


//    // Copy our lists as well
//    index_.reset(new IVFFlat(
//            resources_.get(),
//            quantizer->getGpuData(),  // FlatIndex instance- contains the vectors in index
//            index->metric_type,
//            index->metric_arg,
//            false,   // no residual
//            nullptr, // no scalar quantizer
//            ivfFlatConfig_.interleavedLayout,
//            ivfFlatConfig_.indicesOptions,
//            config_.memorySpace));
//
//    // Copy all of the IVF data
//    index_->copyInvertedListsFrom(index->invlists);  // xcopy


    raft::spatial::knn::ivf_flat::index_params raft_idx_params;
    raft_idx_params.n_lists = nlist;
    raft_idx_params.metric = raft::distance::DistanceType::L2Expanded;

    raft_knn_index.emplace(raft_handle, raft_idx_params, (uint32_t)d);

    /**
     * TODO: Copy centers and center norms from quantizer
     * Things to do:
     *    1. Copy index_->quantizer->vectors_ to raft_index->centers
     *    2. Copy index_->quantizer->norms_ to raft_index->center_norms
     */

    raft::copy(raft_knn_index.value().centers(),


    /**
     * TODO: Copy IVF data, indices, list_sizes, list_offsets from index->invlists
     *
     * Things to do:
     *    1. index->ivflists->data() is going to need to be translated over to our format
     *       (even the interleaved format is a little different)
     *
     *       The GpuIndexIVFFlat has a function translateCodesToGpu_() for this
     *
     *    2. We will need to copy  list_sizes, indices, and list_offsets
     */

}

void RaftIndexIVFFlat::reserveMemory(size_t numVecs) {

    std::cout << "Reserving memory for " << numVecs << " vectors." << std::endl;
    reserveMemoryVecs_ = numVecs;
    if (raft_knn_index.has_value()) {
        DeviceScope scope(config_.device);

        // TODO: Need to figure out if this is absolutely necessary.

        /**
         * For example:
         * raft::spatial::knn::ivf_flat::allocate_ivf_lists(
         *      raft_handle, *raft_knn_index, numVecs);
         *
         * raft::spatial::knn::ivf_flat::populate(
         *      raft_handle, *raft_knn_index,
         *      n_centroids, centroids,
         *      n_vectors, ivf);
         *
         */
    }
}

size_t RaftIndexIVFFlat::reclaimMemory() {
    std::cout << "Reclaiming memory" << std::endl;

    // TODO: Need to figure out if this is absolutely necessary
    /**
     * For example:
     * raft::spatial::knn::ivf_flat::reclaim_ivf_lists(
     *      raft_handle, *raft_knn_index, numVecs);
     */
    return 0;
}

void RaftIndexIVFFlat::train(Index::idx_t n, const float* x) {
    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    DeviceScope scope(config_.device);

    if (this->is_trained) {
        FAISS_ASSERT(raft_knn_index.has_value());
        return;
    }

    raft::spatial::knn::ivf_flat::index_params raft_idx_params;
    raft_idx_params.n_lists = nlist;
    raft_idx_params.metric = raft::distance::DistanceType::L2Expanded;

    raft_knn_index.emplace(
        raft::spatial::knn::ivf_flat::build(raft_handle, raft_idx_params,
                                            const_cast<float*>(x),
                                            n, (faiss::Index::idx_t)d));

    raft_handle.sync_stream();
}

int RaftIndexIVFFlat::getListLength(int listId) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    DeviceScope scope(config_.device);

    // TODO: Call function in RAFT to do this.
    /**
     * For example:
     * raft::spatial::knn::ivf_flat::get_list_length(
     *    raft_handle, *raft_knn_index, listId);
     */
    return 0;
}

std::vector<uint8_t> RaftIndexIVFFlat::getListVectorData(
        int listId,
        bool gpuFormat) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    DeviceScope scope(config_.device);

    // TODO: Invoke corresponding call in raft::ivf_flat
    /**
     * For example:
     * raft::spatial::knn::ivf_flat::get_list_vector_data(
     *    raft_handle, *raft_knn_index, listId, gpuFormat);
     */
    std::vector<uint8_t> vec;
    return vec;
}

void RaftIndexIVFFlat::reset() {
    std::cout << "Calling reset()" << std::endl;
    raft_knn_index.reset();
}

std::vector<Index::idx_t> RaftIndexIVFFlat::getListIndices(int listId) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    DeviceScope scope(config_.device);

    // TODO: Need to invoke corresponding call in raft::ivf_flat
    /**
     * For example:
     * raft::spatial::knn::ivf_flat::get_list_indices(
     *    raft_handle, *raft_knn_index, listId);
     */
    Index::idx_t start_offset, stop_offset;
    std::vector<Index::idx_t> vec;
    return vec;
}

void RaftIndexIVFFlat::addImpl_(
        int n,
        const float* x,
        const Index::idx_t* xids) {
    // Device is already set in GpuIndex::add
    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(n > 0);

//    // Not all vectors may be able to be added (some may contain NaNs etc)
//    index_->addVectors(data, labels);
//
//    // but keep the ntotal based on the total number of vectors that we
//    // attempted to add

    std::cout << "Calling addImpl_ with " << n << " vectors." << std::endl;

    /**
     * For example:
     * raft::spatial::knn::ivf_flat::add_vectors(
     *      raft_handle, *raft_knn_index, n, x, xids);
     */
    raft_knn_index.emplace(raft::spatial::knn::ivf_flat::extend(
            raft_handle, raft_knn_index.value(), x, xids, (Index::idx_t)n));
    this->ntotal += n;
}

void RaftIndexIVFFlat::searchImpl_(
        int n,
        const float* x,
        int k,
        float* distances,
        Index::idx_t* labels) const {
    // Device is already set in GpuIndex::search
    FAISS_ASSERT(raft_knn_index.has_value());
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= nlist);

    // Data is already resident on the GPU
    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, (int)this->d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<Index::idx_t, 2, true> outLabels(
            const_cast<Index::idx_t*>(labels), {n, k});

    // TODO: Populate the rest of the params properly.
    raft::spatial::knn::ivf_flat::search_params raft_idx_params;
    raft_idx_params.n_probes = nprobe;

    raft::spatial::knn::ivf_flat::search<float, faiss::Index::idx_t>(
                                         raft_handle,
                                         raft_idx_params,
                                         *raft_knn_index,
                                         const_cast<float*>(x),
                                         static_cast<std::uint32_t>(n),
                                         static_cast<std::uint32_t>(k),
                                         static_cast<faiss::Index::idx_t *>(labels),
                                         distances);

    raft_handle.sync_stream();
}

} // namespace gpu
} // namespace faiss
