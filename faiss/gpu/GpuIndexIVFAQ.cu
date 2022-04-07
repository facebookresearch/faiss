/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFAQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFAQ.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/utils/utils.h>
#include <faiss/gpu/impl/IVFAQ.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>

#include <limits>

namespace faiss {
namespace gpu {

GpuIndexIVFAQ::GpuIndexIVFAQ(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFAQ* index,
        GpuIndexIVFAQConfig config)
        : GpuIndexIVF(
                  provider,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  index->nlist,
                  config),
          aq(index->aq),
          ivfaqConfig_(config),
          usePrecomputedTables_(config.usePrecomputedTables),
          subQuantizers_(0),
          bitsPerCode_(0),
          reserveMemoryVecs_(0) {
    copyFrom(index);
}

GpuIndexIVFAQ::GpuIndexIVFAQ(
        GpuResourcesProvider* provider,
        int dims,
        int nlist,
        int subQuantizers,
        int bitsPerCode,
        faiss::MetricType metric,
        GpuIndexIVFAQConfig config)
        : GpuIndexIVF(provider, dims, metric, 0, nlist, config),
          aq(dims, subQuantizers, bitsPerCode),
          ivfaqConfig_(config),
          usePrecomputedTables_(config.usePrecomputedTables),
          subQuantizers_(subQuantizers),
          bitsPerCode_(bitsPerCode),
          reserveMemoryVecs_(0) {
    verifySettings_();

    this->is_trained = false;
}

GpuIndexIVFAQ::~GpuIndexIVFAQ() {}

void GpuIndexIVFAQ::copyFrom(const faiss::IndexIVFAQ* index) {
    DeviceScope scope(config_.device);

    GpuIndexIVF::copyFrom(index);

    // Clear out our old data
    index_.reset();

    aq = index->aq;
    subQuantizers_ = index->aq.M;
    bitsPerCode_ = index->aq.nbits;

    // We only support this
    FAISS_THROW_IF_NOT_MSG(
            index->pq.nbits == 8, "GPU: only pq.nbits == 8 is supported");
    FAISS_THROW_IF_NOT_MSG(
            index->by_residual, "GPU: only by_residual = true is supported");
    FAISS_THROW_IF_NOT_MSG(
            index->polysemous_ht == 0, "GPU: polysemous codes not supported");

    verifySettings_();

    // The other index might not be trained
    if (!index->is_trained) {
        // copied in GpuIndex::copyFrom
        FAISS_ASSERT(!is_trained);
        return;
    }

    // Copy our lists as well
    // The additive quantizer must have data in it
    FAISS_ASSERT(index->aq.centroids.size() > 0);
    index_.reset(new IVFAQ(
            resources_.get(),
            index->metric_type,
            index->metric_arg,
            quantizer->getGpuData(),
            subQuantizers_,
            bitsPerCode_,
            ivfaqConfig_.useFloat16LookupTables,
            ivfaqConfig_.useMMCodeDistance,
            ivfaqConfig_.interleavedLayout,
            (float*)index->aq.centroids.data(),
            ivfaqConfig_.indicesOptions,
            config_.memorySpace));
    // Doesn't make sense to reserve memory here
    index_->setPrecomputedCodes(usePrecomputedTables_);

    // Copy all of the IVF data
    index_->copyInvertedListsFrom(index->invlists);
}

void GpuIndexIVFAQ::copyTo(faiss::IndexIVFAQ* index) const {
    DeviceScope scope(config_.device);

    // We must have the indices in order to copy to ourselves
    FAISS_THROW_IF_NOT_MSG(
            ivfaqConfig_.indicesOptions != INDICES_IVF,
            "Cannot copy to CPU as GPU index doesn't retain "
            "indices (INDICES_IVF)");

    GpuIndexIVF::copyTo(index);

    //
    // IndexIVFAQ information
    //
    index->by_residual = true;
    index->use_precomputed_table = 0;
    index->code_size = subQuantizers_;
    index->aq = faiss::AdditiveQuantizer(this->d, subQuantizers_, bitsPerCode_);

    index->do_polysemous_training = false;
    index->polysemous_training = nullptr;

    index->scan_table_threshold = 0;
    index->max_codes = 0;
    index->polysemous_ht = 0;
    index->precomputed_table.clear();

    auto ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (index_) {
        // Copy IVF lists
        index_->copyInvertedListsTo(ivf);

        // Copy AQ centroids
        auto devAQCentroids = index_->getAQCentroids();
        index->aq.centroids.resize(devAQCentroids.numElements());

        fromDevice<float, 3>(
                devAQCentroids,
                index->aq.centroids.data(),
                resources_->getDefaultStream(config_.device));

        if (usePrecomputedTables_) {
            index->precompute_table();
        }
    }
}

void GpuIndexIVFAQ::reserveMemory(size_t numVecs) {
    reserveMemoryVecs_ = numVecs;
    if (index_) {
        // to set the current devece
        // and restore the previous device
        // upon destruction
        DeviceScope scope(config_.device);
        index_->reserveMemory(numVecs);
    }
}

void GpuIndexIVFAQ::setPrecomputedCodes(bool enable) {
    usePrecomputedTables_ = enable;
    if (index_) {
        DeviceScope scope(config_.device);
        index_->setPrecomputedCodes(enable);
    }
    verifySettigs_();
}

bool GpuIndexIVFAQ::getPrecomputedCodes() const {
    return usePrecomputedTables_;
}

int GpuIndexIVFAQ::getNumSubQuantizers() const {
    return subQuantizers_;
}

int GpuIndexIVFAQ::getBitsPerCode() const {
    return bitsPerCode_;
}

int GpuIndexIVFAQ::getCentroidsPerSubQuantizer() const {
    return utils::pow2(bitsPerCode_);
}

size_t GpuIndexIVFAQ::reclaimMemory() {
    if (index_) {
        DeviceScope scope(config_.device);
        return index_->reclaimMemory();
    }
    return 0;
}

void GpuIndexIVFPQ::reset() {
    if (index_) {
        DeviceScope scope(config_.device);

        index_->reset();
        this->ntotal = 0;
    } else {
        FAISS_ASSERT(this->ntotal == 0);
    }
}

void GpuIndexIVFAQ::trainResiduaQuantizer_(Index::idx_t n, const float* x) {
    // Code largely copied from faiss::IndexIVFAQ
    auto x_in = x;

    size_t max_train_points = 1024 * ((size_t)1 << aq->nbits[0]);
    // we need more data to train LSQ
    if (dynamic_cast<LocalSearchQuantizer*>(aq)) {
        max_train_points = 1024 * aq->M * ((size_t)1 << aq->nbits[0]);
    }
    x = fvecs_maybe_subsample(
            d, (size_t*)&n, max_train_points, x, verbose, 1234);

    ScopeDeleter<float> del_x(x_in == x ? nullptr : x);

    if (this->verbose) {
        printf("computing residuals\n");
    }

    std::vector<Index::idx_t> idx(n);
    quantizer->assign(n, x, idx.data());

    std::vector<float> residuals(n * d);
    quantizer->compute_residual_n(n, x, residuals.data(), idx.data());

    if (this->verbose) {
        printf("training %d x %d additive quantizer on %ld vectors in %dD\n",
               subQuantizers_,
               getCentroidsPerSubQuantizer(),
               n,
               this->d);
    }

    // accelerate aq training by using a Gpu clustering index
    // if a clutering index has not already been assigned
    if (!aq.assign_index) {
        // TODO:wxx
        try {
            aq.train(n, residuals.data());
        } catch (...) {
            aq.assign_index = nullptr;
            throw;
        }
        aq.assign_index = nullptr;
    } else {
        aq.train(n, residuals.data());
    }
    index_.reset(new IVFAQ(
            resources_.get(),
            metric_type,
            metric_arg,
            quantizer->getGpuData(),
            subQuantizers_,
            bitsPerCode_,
            ivfaqConfig_.useFloat16LookupTables,
            ivfaqConfig_.useMMCodeDistance,
            ivfaqConfig_.interleavedLayout,
            aq.centroids.data(),
            ivfaqConfig_.indicesOptions,
            config_.memorySpace));
    if (reserveMemoryVecs_) {
        index_->reserveMemory(reserveMemoryVecs_);
    }

    index_->setPrecomputedCodes(usePrecomputedTables_);
}

void GpuIndexIVFAQ::train(index::idx_t n, const float* x) {
    // For now, only support <= max int results
    /*
    FAISS_THROW_IF_NOT_FMT(
            n <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());
        */
    DeviceScope scope(config_.device);

    if (this->is_trained) {
        FAISS_ASSERT(quantizer->is_trained);
        FAISS_ASSERT(quantizer->ntotal == nlist);
        FAISS_ASSERT(index_);
        return;
    }

    FAISS_ASSERT(!index_);

    // FIXME: GPUize more of this
    auto hostData = toHost<float, 2>(
            (float*)x,
            resources_->getDefaultStream(config_.device),
            {(int)n, (int)this->d});

    trainQuantizer_(n, hostData.data());
    trainResidualQuantizer_(n, hostData.data());

    FAISS_ASSERT(index_);
    this->is_trained = true;
}

void GpuIndexIVFPQ::addImpl_(int n, const float* x, const Index::idx_t* xids) {
    // Device is already set in GpuIndex::add
    FAISS_ASSERT(index_);
    FAISS_ASSERT(n > 0);

    // Data is already resident on the GPU
    Tensor<float, 2, true> data(const_cast<float*>(x), {n, (int)this->d});
    Tensor<Index::idx_t, 1, true> labels(const_cast<Index::idx_t*>(xids), {n});

    // Not all vectors may be able to be added (some may contain NaNs etc)
    index_->addVectors(data, labels);

    // but keep the ntotal based on the total number of vectors that we
    // attempted to add
    ntotal += n;
}

void GpuIndex void GpuIndexIVFAQ::searchImpl_(
        int n,
        const float* x,
        int k,
        float* distances,
        Index::idx_t* labels) const {
    // Device is already set in GpuIndex::search
    FAISS_ASSERT(index_);
    FAISS_ASSERT(n > 0);
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= nlist);

    // Data is already resident on the GPU
    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, (int)this->d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<Index::idx_t, 2, true> outLabels(
            const_cast<Index::idx_t*>(labels), {n, k});

    index_->query(queries, nprobe, k, outDistances, outLabels);
}

} // namespace gpu
} // namespace faiss