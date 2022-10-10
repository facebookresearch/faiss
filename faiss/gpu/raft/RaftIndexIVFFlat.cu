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
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/raft/RaftIndexIVFFlat.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/Float16.cuh>

#include <raft/core/nvtx.hpp>
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
          raft_handle(resources_->getDefaultStream(config_.device)) {

    std::cout << "In raft index constructor" << std::endl;
}

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

    /**
     * TODO: Copy centers and center norms from quantizer
     * Things to do:
     *    1. Copy index_->quantizer->vectors_ to raft_index->centers
     *    2. Copy index_->quantizer->norms_ to raft_index->center_norms
     */
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
    if (index->is_trained) {
        // TODO: A proper copy of the index without retraining
        // For now, just get all the data from the index, and train our index
        // anew.
        FAISS_ASSERT(index->d == this->d);
        FAISS_ASSERT(index->metric_arg == this->metric_arg);
        FAISS_ASSERT(index->metric_type == this->metric_type);
        FAISS_ASSERT(index->nlist == this->nlist);

        Index::idx_t quantizer_ntotal = index->quantizer->ntotal;
        Index::idx_t index_ntotal = index->ntotal;

        std::cout << "Calling copyFrom with trained index with "  << quantizer_ntotal << " items" << std::endl;
        auto stream = raft_handle.get_stream();

        auto total_elems = size_t(quantizer_ntotal) * size_t(index->quantizer->d);

        raft::spatial::knn::ivf_flat::index_params pams;

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

        raft_knn_index.emplace(raft_handle, pams.metric, (uint32_t)this->nlist, (uint32_t)this->d);

        // Copy (reconstructed) centroids over, rather than re-training
        rmm::device_uvector<float> buf_dev(total_elems, stream);
        {
            std::vector<float> buf_host(total_elems);
            index->quantizer->reconstruct_n(0, quantizer_ntotal, buf_host.data());
            raft::copy(raft_knn_index.value().centers().data_handle(), buf_host.data(), total_elems, stream);
        }

        raft::print_device_vector("raft centers", raft_knn_index.value().centers().data_handle(), total_elems, std::cout);

        // Add (reconstructed) vectors to index if needed
        if(index_ntotal > 0) {
            std::cout << "Adding " << index_ntotal << " vectors to index" << std::endl;
            total_elems = size_t(index_ntotal) * size_t(index->d);
            buf_dev.resize(total_elems, stream);
            {
                std::vector<float> buf_host(total_elems);
                index->reconstruct_n(0, index_ntotal, buf_host.data());
                raft::copy(buf_dev.data(), buf_host.data(), total_elems, stream);
            }

            // TODO: We might want to consider moving the centroid norm computation
            // outside of the incremental add on the RAFT side.
            RaftIndexIVFFlat::addImpl_(index_ntotal, buf_dev.data(), nullptr);
        }
    } else {
        // index is not trained, so we can remove ours as well (if there was
        // any)
        std::cout << "Calling copyFrom with index that hasn't been trained" << std::endl;
        raft_knn_index.reset();
    }
    this->is_trained = index->is_trained;
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
    DeviceScope scope(config_.device);


    raft::common::nvtx::range<raft::common::nvtx::domain::raft> fun_scope(
            "RaftIndexIVFFlat::train (%ld)", n);

    std::cout << "Calling train() with " << n << " rows" << std::endl;

    uint32_t start = raft::curTimeMillis();
    if (this->is_trained) {
        FAISS_ASSERT(raft_knn_index.has_value());
        return;
    }

    raft::spatial::knn::ivf_flat::index_params raft_idx_params;
    raft_idx_params.n_lists = nlist;
    raft_idx_params.metric = raft::distance::DistanceType::L2Expanded;
    raft_idx_params.add_data_on_build = false;
    raft_idx_params.kmeans_n_iters = 100;

    raft_knn_index.emplace(
        raft::spatial::knn::ivf_flat::build(raft_handle, raft_idx_params,
                                            const_cast<float*>(x),
                                            n, (faiss::Index::idx_t)d));

    raft_handle.sync_stream();
    uint32_t stop = raft::curTimeMillis();

    std::cout << "train took " << (stop - start) << "ms. " << std::endl;
    this->is_trained = true;
}

int RaftIndexIVFFlat::getListLength(int listId) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    DeviceScope scope(config_.device);

    uint32_t size;
    raft::copy(&size, raft_knn_index.value().list_sizes().data_handle() + listId,
               1, raft_handle.get_stream());
    raft_handle.sync_stream();
    return int(size);
}

std::vector<uint8_t> RaftIndexIVFFlat::getListVectorData(
        int listId,
        bool gpuFormat) const {
    FAISS_ASSERT(raft_knn_index.has_value());
    DeviceScope scope(config_.device);

    std::cout << "Calling getListVectorData for " << listId << std::endl;

    using elem_t = decltype(raft_knn_index.value().data())::element_type;
    size_t dim = raft_knn_index.value().dim();
    Index::idx_t offsets[2];
    raft::copy(offsets, raft_knn_index.value().list_offsets().data_handle() + listId, 2, raft_handle.get_stream());

    raft_handle.sync_stream();
    size_t byte_offset = offsets[0] * sizeof(elem_t) * dim;
    // the interleaved block can be slightly larger than the list size (it's
    // rounded up)
    size_t byte_size = size_t(offsets[1]) *
                    sizeof(elem_t) * dim -
            byte_offset;
    std::vector<uint8_t> vec(byte_size);
    raft::copy(
            vec.data(),
            reinterpret_cast<const uint8_t*>(raft_knn_index.value().data().data_handle()) +
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

    Index::idx_t offset;
    uint32_t size;

    raft::copy(&offset, raft_knn_index.value().list_offsets().data_handle() + listId, 1, raft_handle.get_stream());
    raft::copy(&size, raft_knn_index.value().list_sizes().data_handle() + listId, 1, raft_handle.get_stream());
    raft_handle.sync_stream();

    std::vector<Index::idx_t> vec(size);
    raft::copy(
            vec.data(),
            raft_knn_index.value().indices().data_handle() + offset,
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

//    // Not all vectors may be able to be added (some may contain NaNs etc)
//    index_->addVectors(data, labels);

    // but keep the ntotal based on the total number of vectors that we
    // attempted to add

    std::cout << "Calling addImpl_ with " << n << " vectors." << std::endl;

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

    raft::common::nvtx::range<raft::common::nvtx::domain::raft> fun_scope(
            "RaftIndexIVFFlat::searchImpl_ (%ld)", n);

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

    std::cout << "Calling rebuildRaftIndex with " << n_rows << " rows" << std::endl;
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
    pams.add_data_on_build = false;

    raft_knn_index.emplace(raft::spatial::knn::ivf_flat::build(
            this->raft_handle, pams, x, n_rows, uint32_t(this->d)));

    this->raft_handle.sync_stream();
    this->is_trained = true;
    this->ntotal = n_rows;
}

} // namespace gpu
} // namespace faiss
