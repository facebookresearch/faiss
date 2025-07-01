/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/clone_index.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/IVFBase.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>

namespace faiss {
namespace gpu {

GpuIndexIVF::GpuIndexIVF(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        idx_t nlistIn,
        GpuIndexIVFConfig config)
        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
          IndexIVFInterface(nullptr, nlistIn),
          ivfConfig_(config) {
    // Only IP and L2 are supported for now
    if (!(metric_type == faiss::METRIC_L2 ||
          metric_type == faiss::METRIC_INNER_PRODUCT)) {
        FAISS_THROW_FMT("unsupported metric type %d", (int)metric_type);
    }

    init_();
}

GpuIndexIVF::GpuIndexIVF(
        GpuResourcesProvider* provider,
        Index* coarseQuantizer,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        idx_t nlistIn,
        GpuIndexIVFConfig config)
        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
          IndexIVFInterface(coarseQuantizer, nlistIn),
          ivfConfig_(config) {
    FAISS_THROW_IF_NOT_MSG(
            quantizer, "expecting a coarse quantizer object; none provided");

    // We are passed an external quantizer object that we do not own
    own_fields = false;

    // Only IP and L2 are supported for now
    if (!(metric_type == faiss::METRIC_L2 ||
          metric_type == faiss::METRIC_INNER_PRODUCT)) {
        FAISS_THROW_FMT("unsupported metric type %d", (int)metric_type);
    }

    init_();
}

void GpuIndexIVF::init_() {
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be > 0");

    // Spherical by default if the metric is inner_product
    // (copying IndexIVF.cpp)
    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }

    // here we set a low # iterations because this is typically used
    // for large clusterings (copying IndexIVF.cpp's Level1Quantizer
    cp.niter = 10;

    cp.verbose = verbose;

    if (quantizer) {
        // The passed in quantizer may be either a CPU or GPU index
        // Same work as IndexIVF's constructor
        is_trained = quantizer->is_trained && quantizer->ntotal == nlist;
    } else {
        // we have not yet been trained
        is_trained = false;

        // Construct a GPU empty flat quantizer as our coarse quantizer
        GpuIndexFlatConfig config = ivfConfig_.flatConfig;
        // inherit our same device
        config.device = config_.device;
        config.use_cuvs = config_.use_cuvs;

        if (metric_type == faiss::METRIC_L2) {
            quantizer = new GpuIndexFlatL2(resources_, d, config);
        } else if (metric_type == faiss::METRIC_INNER_PRODUCT) {
            quantizer = new GpuIndexFlatIP(resources_, d, config);
        } else {
            // unknown metric type
            FAISS_THROW_FMT("unsupported metric type %d", (int)metric_type);
        }

        // we instantiated the coarse quantizer here, so we destroy it as well
        own_fields = true;
    }

    verifyIVFSettings_();
}

GpuIndexIVF::~GpuIndexIVF() {}

void GpuIndexIVF::verifyIVFSettings_() const {
    // We should always have a quantizer instance
    FAISS_THROW_IF_NOT(quantizer);
    FAISS_THROW_IF_NOT(d == quantizer->d);

    if (is_trained) {
        FAISS_THROW_IF_NOT(quantizer->is_trained);

        // IVF quantizer should correspond to our set of lists
        FAISS_THROW_IF_NOT_FMT(
                quantizer->ntotal == nlist,
                "IVF nlist count (%zu) does not match trained coarse quantizer size (%zu)",
                nlist,
                quantizer->ntotal);
    } else {
        // The coarse quantizer may or may not be trained, but if we are
        // trained, then the coarse quantizer must also be trained (the check
        // above)
        FAISS_THROW_IF_NOT(ntotal == 0);
    }

    // If the quantizer is a GPU index, then it must be resident on the same
    // device as us
    auto gpuQuantizer = tryCastGpuIndex(quantizer);
    if (gpuQuantizer && gpuQuantizer->getDevice() != getDevice()) {
        FAISS_THROW_FMT(
                "GpuIndexIVF: not allowed to instantiate a GPU IVF "
                "index that is resident on a different GPU (%d) "
                "than its GPU coarse quantizer (%d)",
                getDevice(),
                gpuQuantizer->getDevice());
    }
}

void GpuIndexIVF::copyFrom(const faiss::IndexIVF* index) {
    DeviceScope scope(config_.device);

    GpuIndex::copyFrom(index);

    FAISS_ASSERT(index->nlist > 0);
    nlist = index->nlist;

    validateNProbe(index->nprobe, should_use_cuvs(config_));
    nprobe = index->nprobe;

    // The metric type may have changed as well, so we might have to
    // change our quantizer
    if (own_fields) {
        delete quantizer;
    }
    quantizer = nullptr;

    // IVF index that we are copying from must have a coarse quantizer
    FAISS_THROW_IF_NOT(index->quantizer);

    if (!isGpuIndex(index->quantizer)) {
        // The coarse quantizer used in the IndexIVF is non-GPU.
        // If it is something that we support on the GPU, we wish to copy it
        // over to the GPU, on the same device that we are on.
        GpuResourcesProviderFromInstance pfi(getResources());

        // Attempt to clone the index to GPU. If it fails because the coarse
        // quantizer is not implemented on GPU and the flag to allow CPU
        // fallback is set, retry it with CPU cloner and re-throw errors.
        try {
            GpuClonerOptions options;
            auto cloner = ToGpuCloner(&pfi, getDevice(), options);
            quantizer = cloner.clone_Index(index->quantizer);
        } catch (const std::exception& e) {
            if (strstr(e.what(), "not implemented on GPU")) {
                if (ivfConfig_.allowCpuCoarseQuantizer) {
                    Cloner cpuCloner;
                    quantizer = cpuCloner.clone_Index(index->quantizer);
                } else {
                    FAISS_THROW_MSG(
                            "This index type is not implemented on "
                            "GPU and allowCpuCoarseQuantizer is set to false. "
                            "Please set the flag to true to allow the CPU "
                            "fallback in cloning.");
                }
            } else {
                throw;
            }
        }
        own_fields = true;
    } else {
        // Otherwise, this is a GPU coarse quantizer index instance found in a
        // CPU instance. It is unclear what we should do here, but for now we'll
        // flag this as an error (we're expecting a pure CPU index)
        FAISS_THROW_MSG(
                "GpuIndexIVF::copyFrom: copying a CPU IVF index to GPU "
                "that already contains a GPU coarse (level 1) quantizer "
                "is not currently supported");
    }

    // Validate equality
    FAISS_ASSERT(is_trained == index->is_trained);
    FAISS_ASSERT(ntotal == index->ntotal);
    FAISS_ASSERT(nlist == index->nlist);
    FAISS_ASSERT(quantizer->is_trained == index->quantizer->is_trained);
    FAISS_ASSERT(quantizer->ntotal == index->quantizer->ntotal);

    // Validate IVF/quantizer settings
    verifyIVFSettings_();
}

void GpuIndexIVF::copyTo(faiss::IndexIVF* index) const {
    DeviceScope scope(config_.device);

    //
    // Index information
    //
    GpuIndex::copyTo(index);

    //
    // IndexIVF information
    //
    index->nlist = nlist;
    index->nprobe = nprobe;

    FAISS_ASSERT(quantizer);
    if (index->own_fields) {
        delete index->quantizer;
        index->quantizer = nullptr;
    }

    index->quantizer = index_gpu_to_cpu(quantizer);
    FAISS_THROW_IF_NOT(index->quantizer);

    // Validate consistency between the coarse quantizer and the index
    FAISS_ASSERT(
            index->quantizer->is_trained == quantizer->is_trained &&
            index->quantizer->is_trained == is_trained);
    FAISS_ASSERT(index->quantizer->ntotal == quantizer->ntotal);

    index->own_fields = true;
    index->quantizer_trains_alone = 0;
    index->cp = this->cp;
    index->make_direct_map(false);
}

idx_t GpuIndexIVF::getNumLists() const {
    return nlist;
}

idx_t GpuIndexIVF::getListLength(idx_t listId) const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(baseIndex_);

    return baseIndex_->getListLength(listId);
}

std::vector<uint8_t> GpuIndexIVF::getListVectorData(
        idx_t listId,
        bool gpuFormat) const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(baseIndex_);

    return baseIndex_->getListVectorData(listId, gpuFormat);
}

std::vector<idx_t> GpuIndexIVF::getListIndices(idx_t listId) const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(baseIndex_);

    return baseIndex_->getListIndices(listId);
}

void GpuIndexIVF::addImpl_(idx_t n, const float* x, const idx_t* xids) {
    // Device is already set in GpuIndex::add
    FAISS_ASSERT(baseIndex_);
    FAISS_ASSERT(n > 0);

    // Data is already resident on the GPU
    Tensor<float, 2, true> data(const_cast<float*>(x), {n, this->d});
    Tensor<idx_t, 1, true> labels(const_cast<idx_t*>(xids), {n});

    // Not all vectors may be able to be added (some may contain NaNs etc)
    baseIndex_->addVectors(quantizer, data, labels);

    // but keep the ntotal based on the total number of vectors that we
    // attempted to add
    ntotal += n;
}

int GpuIndexIVF::getCurrentNProbe_(const SearchParameters* params) const {
    size_t use_nprobe = nprobe;
    if (params) {
        auto ivfParams = dynamic_cast<const SearchParametersIVF*>(params);
        if (ivfParams) {
            use_nprobe = ivfParams->nprobe;

            FAISS_THROW_IF_NOT_FMT(
                    ivfParams->max_codes == 0,
                    "GPU IVF index does not currently support "
                    "SearchParametersIVF::max_codes (passed %zu, must be 0)",
                    ivfParams->max_codes);
        } else {
            FAISS_THROW_MSG(
                    "GPU IVF index: passed unhandled SearchParameters "
                    "class to search function; only SearchParametersIVF "
                    "implemented at present");
        }
    }

    validateNProbe(use_nprobe, should_use_cuvs(config_));
    // We use int internally for nprobe
    return int(use_nprobe);
}

void GpuIndexIVF::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // Device was already set in GpuIndex::search
    int use_nprobe = getCurrentNProbe_(params);

    // This was previously checked
    FAISS_ASSERT(is_trained && baseIndex_);
    FAISS_ASSERT(n > 0);

    // Data is already resident on the GPU
    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, this->d});
    Tensor<float, 2, true> outDistances(distances, {n, k});
    Tensor<idx_t, 2, true> outLabels(const_cast<idx_t*>(labels), {n, k});

    baseIndex_->search(
            quantizer, queries, use_nprobe, k, outDistances, outLabels);
}

void GpuIndexIVF::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT_MSG(stats == nullptr, "IVF stats not supported");
    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    FAISS_THROW_IF_NOT_MSG(
            !store_pairs,
            "GpuIndexIVF::search_preassigned does not "
            "currently support store_pairs");
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "GpuIndexIVF not trained");
    FAISS_ASSERT(baseIndex_);

    validateKSelect(k, should_use_cuvs(config_));

    if (n == 0 || k == 0) {
        // nothing to search
        return;
    }

    idx_t use_nprobe = params ? params->nprobe : this->nprobe;
    validateNProbe(use_nprobe, should_use_cuvs(config_));

    size_t max_codes = params ? params->max_codes : this->max_codes;
    FAISS_THROW_IF_NOT_FMT(
            max_codes == 0,
            "GPU IVF index does not currently support "
            "SearchParametersIVF::max_codes (passed %zu, must be 0)",
            max_codes);

    // Ensure that all data/output buffers are resident on our desired device
    auto vecsDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(x),
            stream,
            {n, d});

    auto distanceDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(centroid_dis),
            stream,
            {n, use_nprobe});

    auto assignDevice = toDeviceTemporary<idx_t, 2>(
            resources_.get(),
            config_.device,
            const_cast<idx_t*>(assign),
            stream,
            {n, use_nprobe});

    auto outDistancesDevice = toDeviceTemporary<float, 2>(
            resources_.get(), config_.device, distances, stream, {n, k});

    auto outIndicesDevice = toDeviceTemporary<idx_t, 2>(
            resources_.get(), config_.device, labels, stream, {n, k});

    baseIndex_->searchPreassigned(
            quantizer,
            vecsDevice,
            distanceDevice,
            assignDevice,
            k,
            outDistancesDevice,
            outIndicesDevice,
            store_pairs);

    // If the output was not already on the GPU, copy it back
    fromDevice<float, 2>(outDistancesDevice, distances, stream);
    fromDevice<idx_t, 2>(outIndicesDevice, labels, stream);
}

void GpuIndexIVF::range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_MSG("range search not implemented");
}

bool GpuIndexIVF::addImplRequiresIDs_() const {
    // All IVF indices have storage for IDs
    return true;
}

void GpuIndexIVF::trainQuantizer_(idx_t n, const float* x) {
    DeviceScope scope(config_.device);

    if (n == 0) {
        // nothing to do
        return;
    }

    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (this->verbose) {
            printf("IVF quantizer does not need training.\n");
        }

        return;
    }

    if (this->verbose) {
        printf("Training IVF quantizer on %ld vectors in %dD\n", n, d);
    }

    quantizer->reset();

    // leverage the CPU-side k-means code, which works for the GPU
    // flat index as well
    Clustering clus(this->d, nlist, this->cp);
    clus.verbose = verbose;
    clus.train(n, x, *quantizer);

    quantizer->is_trained = true;
    FAISS_ASSERT(quantizer->ntotal == nlist);
}

} // namespace gpu
} // namespace faiss
