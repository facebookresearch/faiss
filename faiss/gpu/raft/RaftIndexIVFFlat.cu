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

    config.device = config_.device;

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

    raft::spatial::knn::ivf_flat::index_params raft_idx_params;
    raft_idx_params.n_lists = nlist;
    raft_idx_params.metric = raft::distance::DistanceType::L2Expanded;

    // TODO: Invoke corresponding call on the RAFT side to copy quantizer
    /**
     * For example:
     * raft_knn_index.emplace(raft::spatial::knn::ivf_flat::make_ivf_flat_index<T>(
     *      raft_handle, raft_idx_params, (faiss::Index::idx_t)d);
     */
}

void RaftIndexIVFFlat::reserveMemory(size_t numVecs) {

    std::cout << "Reserving memory for " << numVecs << " vectors." << std::endl;
    reserveMemoryVecs_ = numVecs;
    if (raft_knn_index.has_value()) {
        DeviceScope scope(config_.device);

        // TODO: We need to reserve memory on the raft::ivf_flat::index
        /**
         * For example:
         * raft::spatial::knn::ivf_flat::ivf_flat_allocate_ivf_lists(
         *      raft_handle, *raft_knn_index, numVecs);
         */
    }
}

size_t RaftIndexIVFFlat::reclaimMemory() {
    std::cout << "Reclaiming memory" << std::endl;

    // TODO: We need to reclaim memory on the raft::ivf_flat::index
    /**
     * For example:
     * raft::spatial::knn::ivf_flat::ivf_flat_reclaim_ivf_lists(
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


    // TODO: This should only train the quantizer portion of the index
    /**
     * For example:
     *
     * raft_knn_index.emplace(raft::spatial::knn::ivf_flat::make_ivf_flat_index<T>(
     *      raft_handle, raft_idx_params, (faiss::Index::idx_t)d);

     * raft::spatial::knn::ivf_flat::ivf_flat_train_quantizer(
     *      raft_handle, *raft_knn_index, const_cast<float*>(x), n);
     */

    raft_knn_index.emplace(
        raft::spatial::knn::ivf_flat::build(raft_handle, raft_idx_params,
                                            const_cast<float*>(x),
                                            n, (faiss::Index::idx_t)d,
                                            raft_handle.get_stream()));

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

      // Data is already resident on the GPU
    Tensor<float, 2, true> data(const_cast<float*>(x), {n, (int)this->d});
    Tensor<Index::idx_t, 1, true> labels(const_cast<Index::idx_t*>(xids), {n});

//    // Not all vectors may be able to be added (some may contain NaNs etc)
//    index_->addVectors(data, labels);
//
//    // but keep the ntotal based on the total number of vectors that we
//    // attempted to add
    ntotal += n;

    std::cout << "Calling addImpl_ with " << n << " vectors." << std::endl;

    // TODO: Invoke corresponding call in raft::ivf_flat
    /**
     * For example:
     * raft::spatial::knn::ivf_flat::ivf_flat_add_vectors(
     *      raft_handle, *raft_knn_index, n, x, xids);
     */

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

    raft::spatial::knn::ivf_flat::search<float, faiss::Index::idx_t>(raft_handle,
                                         raft_idx_params,
                                         *raft_knn_index,
                                         const_cast<float*>(x),
                                         static_cast<std::uint32_t>(n),
                                         static_cast<std::uint32_t>(k),
                                         static_cast<faiss::Index::idx_t *>(labels),
                                         distances, raft_handle.get_stream());

    raft_handle.sync_stream();
}

} // namespace gpu
} // namespace faiss
