/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
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
        int nlistIn,
        GpuIndexIVFConfig config)
        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
          nlist(nlistIn),
          nprobe(1),
          quantizer(nullptr),
          own_fields(false),
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
        int nlistIn,
        GpuIndexIVFConfig config)
        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
          nlist(nlistIn),
          nprobe(1),
          quantizer(coarseQuantizer),
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
    // for large clusterings
    // (copying IndexIVF.cpp's Level1Quantizer
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

GpuIndexIVF::~GpuIndexIVF() {
    if (own_fields) {
        delete quantizer;
    }
}

void GpuIndexIVF::verifyIVFSettings_() const {
    // We should always have a quantizer instance
    FAISS_THROW_IF_NOT(quantizer);
    FAISS_THROW_IF_NOT(d == quantizer->d);

    if (is_trained) {
        FAISS_THROW_IF_NOT(quantizer->is_trained);

        // IVF quantizer should correspond to our set of lists
        FAISS_THROW_IF_NOT_FMT(
                quantizer->ntotal == nlist,
                "IVF nlist count (%d) does not match trained coarse quantizer size (%zu)",
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
    FAISS_THROW_IF_NOT_FMT(
            index->nlist <= (idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports %zu inverted lists",
            (size_t)std::numeric_limits<int>::max());
    nlist = index->nlist;

    FAISS_THROW_IF_NOT_FMT(
            index->nprobe > 0 && index->nprobe <= getMaxKSelection(),
            "GPU index only supports nprobe <= %zu; passed %zu",
            (size_t)getMaxKSelection(),
            index->nprobe);
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

        GpuClonerOptions options;
        auto cloner = ToGpuCloner(&pfi, getDevice(), options);

        quantizer = cloner.clone_Index(index->quantizer);
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

int GpuIndexIVF::getNumLists() const {
    return nlist;
}

void GpuIndexIVF::setNumProbes(int nprobe) {
    FAISS_THROW_IF_NOT_FMT(
            nprobe > 0 && nprobe <= getMaxKSelection(),
            "GPU index only supports nprobe <= %d; passed %d",
            getMaxKSelection(),
            nprobe);
    this->nprobe = nprobe;
}

int GpuIndexIVF::getNumProbes() const {
    return nprobe;
}

int GpuIndexIVF::getListLength(int listId) const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(baseIndex_);

    return baseIndex_->getListLength(listId);
}

std::vector<uint8_t> GpuIndexIVF::getListVectorData(int listId, bool gpuFormat)
        const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(baseIndex_);

    return baseIndex_->getListVectorData(listId, gpuFormat);
}

std::vector<idx_t> GpuIndexIVF::getListIndices(int listId) const {
    DeviceScope scope(config_.device);
    FAISS_ASSERT(baseIndex_);

    return baseIndex_->getListIndices(listId);
}

void GpuIndexIVF::addImpl_(int n, const float* x, const idx_t* xids) {
    // Device is already set in GpuIndex::add
    FAISS_ASSERT(baseIndex_);
    FAISS_ASSERT(n > 0);

    // Data is already resident on the GPU
    Tensor<float, 2, true> data(const_cast<float*>(x), {n, (int)this->d});
    Tensor<idx_t, 1, true> labels(const_cast<idx_t*>(xids), {n});

    // Not all vectors may be able to be added (some may contain NaNs etc)
    baseIndex_->addVectors(quantizer, data, labels);

    // but keep the ntotal based on the total number of vectors that we
    // attempted to add
    ntotal += n;
}

void GpuIndexIVF::searchImpl_(
        int n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // Device was already set in GpuIndex::search
    idx_t use_nprobe = nprobe;
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

    validateNProbe(use_nprobe);

    // This was previously checked
    FAISS_ASSERT(is_trained && baseIndex_);
    FAISS_ASSERT(n > 0);

    // Data is already resident on the GPU
    Tensor<float, 2, true> queries(const_cast<float*>(x), {n, (int)this->d});
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
        const IVFSearchParameters* params) const {
    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    FAISS_THROW_IF_NOT_MSG(
            !store_pairs,
            "GpuIndexIVF::search_preassigned does not "
            "currently support store_pairs");
    FAISS_THROW_IF_NOT_MSG(this->is_trained, "GpuIndexIVF not trained");
    FAISS_ASSERT(baseIndex_);

    validateNumVectors(n);
    validateKSelect(k);

    if (n == 0 || k == 0) {
        // nothing to search
        return;
    }

    idx_t use_nprobe = params ? params->nprobe : this->nprobe;
    validateNProbe(use_nprobe);

    if (params) {
        FAISS_THROW_IF_NOT_FMT(
                params->max_codes == 0,
                "GPU IVF index does not currently support "
                "SearchParametersIVF::max_codes (passed %zu, must be 0)",
                params->max_codes);
    }

    // Ensure that all data/output buffers are resident on our desired device
    auto vecsDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(x),
            stream,
            {(int)n, (int)d});

    auto distanceDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(centroid_dis),
            stream,
            {(int)n, (int)use_nprobe});

    auto assignDevice = toDeviceTemporary<idx_t, 2>(
            resources_.get(),
            config_.device,
            const_cast<idx_t*>(assign),
            stream,
            {(int)n, (int)use_nprobe});

    auto outDistancesDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            distances,
            stream,
            {(int)n, (int)k});

    auto outIndicesDevice = toDeviceTemporary<idx_t, 2>(
            resources_.get(), config_.device, labels, stream, {(int)n, (int)k});

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

    // leverage the CPU-side k-means code, which works for the GPU
    // flat index as well
    quantizer->reset();
    Clustering clus(this->d, nlist, this->cp);
    clus.verbose = verbose;
    clus.train(n, x, *quantizer);
    quantizer->is_trained = true;

    FAISS_ASSERT(quantizer->ntotal == nlist);
}

} // namespace gpu
} // namespace faiss
