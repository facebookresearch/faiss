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
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/raft/RaftIndexIVFFlat.h>
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
        : GpuIndexIVFFlat(provider, index, config),
          raft_handle(resources_->getDefaultStream(config_.device)) {
    copyFrom(index);
}

RaftIndexIVFFlat::RaftIndexIVFFlat(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVFFlat(provider, dims, nlist, metric, config),
          raft_handle(resources_->getDefaultStream(config_.device)) {}

RaftIndexIVFFlat::~RaftIndexIVFFlat() {
    RaftIndexIVFFlat::reset();
}

void RaftIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat* index) {
    DeviceScope scope(config_.device);
    GpuIndex::copyFrom(index);
    FAISS_ASSERT(index->nlist > 0);
    FAISS_THROW_IF_NOT_FMT(
            index->nlist <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports %zu inverted lists",
            (size_t)std::numeric_limits<int>::max());
    FAISS_THROW_IF_NOT_FMT(
            index->nprobe > 0 && index->nprobe <= getMaxKSelection(),
            "GPU index only supports nprobe <= %zu; passed %zu",
            (size_t)getMaxKSelection(),
            index->nprobe);

    if (index->is_trained && index->ntotal > 0) {
        // TODO: A proper copy of the index without retraining
        // For now, just get all the data from the index, and train our index
        // anew.
        auto stream = raft_handle.get_stream();
        auto total_elems = size_t(index->ntotal) * size_t(index->d);
        rmm::device_uvector<float> buf_dev(total_elems, stream);
        {
            std::vector<float> buf_host(total_elems);
            index->reconstruct_n(0, index->ntotal, buf_host.data());
            raft::copy(buf_dev.data(), buf_host.data(), total_elems, stream);
        }
        FAISS_ASSERT(index->d == this->d);
        FAISS_ASSERT(index->metric_arg == this->metric_arg);
        FAISS_ASSERT(index->metric_type == this->metric_type);
        FAISS_ASSERT(index->nlist == this->nlist);
        RaftIndexIVFFlat::rebuildRaftIndex(buf_dev.data(), index->ntotal);
    } else {
        // index is not trained, so we can remove ours as well (if there was
        // any)
        raft_knn_index.reset();
    }
    this->is_trained = index->is_trained;
}

void RaftIndexIVFFlat::reserveMemory(size_t numVecs) {
    std::cout << "Reserving memory for " << numVecs << " vectors." << std::endl;
    reserveMemoryVecs_ = numVecs;
    if (raft_knn_index.has_value()) {
        DeviceScope scope(config_.device);

        // TODO: We need to reserve memory on the raft::ivf_flat::index
        /**
         * For example:
         * raft::spatial::knn::ivf_flat::allocate_ivf_lists(
         *      raft_handle, *raft_knn_index, numVecs);
         *
         * raft::spatial::knn::ivf_flat::populate(
         *      raft_handle, *raft_knn_index,
         *      n_centroids, centroids,
         *      n_vectors, ivf);
         */
    }
}

size_t RaftIndexIVFFlat::reclaimMemory() {
    std::cout << "Reclaiming memory" << std::endl;

    // TODO: We need to reclaim memory on the raft::ivf_flat::index
    /**
     * For example:
     * raft::spatial::knn::ivf_flat::reclaim_ivf_lists(
     *      raft_handle, *raft_knn_index, numVecs);
     */
    return 0;
}

void RaftIndexIVFFlat::train(Index::idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    // TODO: This should only train the quantizer portion of the index
    /**
     * For example:
     *
     * raft_knn_index.emplace(raft::spatial::knn::ivf_flat::make_index<T>(
     *      raft_handle, raft_idx_params, (faiss::Index::idx_t)d);

     * raft::spatial::knn::ivf_flat::train_quantizer(
     *      raft_handle, *raft_knn_index, const_cast<float*>(x), n);
     *
     * NB: ivf_flat does not have a quantizer. Training here imply kmeans?
     */

    RaftIndexIVFFlat::rebuildRaftIndex(x, n);
}

int RaftIndexIVFFlat::getListLength(int listId) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    DeviceScope scope(config_.device);

    return int(raft_knn_index->list_sizes(listId));
}

std::vector<uint8_t> RaftIndexIVFFlat::getListVectorData(
        int listId,
        bool gpuFormat) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    DeviceScope scope(config_.device);

    using elem_t = decltype(raft_knn_index->data)::element_type;
    size_t dim = raft_knn_index->dim();
    size_t byte_offset =
            size_t(raft_knn_index->list_offsets(listId)) * sizeof(elem_t) * dim;
    // the interleaved block can be slightly larger than the list size (it's
    // rounded up)
    size_t byte_size = size_t(raft_knn_index->list_offsets(listId + 1)) *
                    sizeof(elem_t) * dim -
            byte_offset;
    std::vector<uint8_t> vec(byte_size);
    raft::copy(
            vec.data(),
            reinterpret_cast<const uint8_t*>(raft_knn_index->data.data()) +
                    byte_offset,
            byte_size,
            raft_handle.get_stream());
    return vec;
}

void RaftIndexIVFFlat::reset() {
    raft_knn_index.reset();
    this->ntotal = 0;
}

std::vector<Index::idx_t> RaftIndexIVFFlat::getListIndices(int listId) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    DeviceScope scope(config_.device);

    size_t offset = raft_knn_index->list_offsets(listId);
    size_t size = raft_knn_index->list_sizes(listId);
    std::vector<Index::idx_t> vec(size);
    raft::copy(
            vec.data(),
            raft_knn_index->indices.data() + offset,
            size,
            raft_handle.get_stream());
    return vec;
}

void RaftIndexIVFFlat::addImpl_(
        int n,
        const float* x,
        const Index::idx_t* xids) {
    // Device is already set in GpuIndex::add
    FAISS_ASSERT(is_trained);
    FAISS_ASSERT(n > 0);
    /* TODO:
      At the moment, raft does not support adding vectors, and does not support
      providing indices with the vectors even in training

      For now, just do the training anew
     */
    raft_knn_index.reset();

    // Not all vectors may be able to be added (some may contain NaNs etc)
    // but keep the ntotal based on the total number of vectors that we
    // attempted to add index_->addVectors(data, labels);
    RaftIndexIVFFlat::rebuildRaftIndex(x, n);
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

    raft::spatial::knn::ivf_flat::search_params pams;
    pams.n_probes = nprobe;
    raft::spatial::knn::ivf_flat::search<float, faiss::Index::idx_t>(
            raft_handle,
            pams,
            *raft_knn_index,
            const_cast<float*>(x),
            static_cast<std::uint32_t>(n),
            static_cast<std::uint32_t>(k),
            labels,
            distances);

    raft_handle.sync_stream();
}

void RaftIndexIVFFlat::rebuildRaftIndex(const float* x, Index::idx_t n_rows) {
    raft::spatial::knn::ivf_flat::index_params pams;

    pams.n_lists = this->nlist;
    switch (this->metric_type) {
        case faiss::METRIC_L2:
            pams.metric = raft::distance::DistanceType::L2Expanded;
            break;
        case faiss::METRIC_INNER_PRODUCT:
            pams.metric = raft::distance::DistanceType::InnerProduct;
            break;
        default:
            FAISS_THROW_MSG("Metric is not supported.");
    }
    pams.metric_arg = this->metric_arg;
    pams.kmeans_trainset_fraction = 1.0;

    raft_knn_index.emplace(raft::spatial::knn::ivf_flat::build(
            this->raft_handle, pams, x, n_rows, uint32_t(this->d)));
    this->raft_handle.sync_stream();
    this->is_trained = true;
    this->ntotal = n_rows;
}

} // namespace gpu
} // namespace faiss
