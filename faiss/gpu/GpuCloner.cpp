/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuCloner.h>
#include <faiss/impl/FaissAssert.h>
#include <memory>
#include <typeinfo>

#include <faiss/gpu/StandardGpuResources.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexFlat.h>
#if defined USE_NVIDIA_CUVS
#include <faiss/IndexHNSW.h>
#endif
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexReplicas.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexShardsIVF.h>
#include <faiss/MetaIndexes.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexBinaryFlat.h>
#if defined USE_NVIDIA_CUVS
#include <faiss/gpu/GpuIndexBinaryCagra.h>
#include <faiss/gpu/GpuIndexCagra.h>
#endif
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/index_io.h>

namespace faiss {
namespace gpu {

/**********************************************************
 * Cloning to CPU
 **********************************************************/

void ToCPUCloner::merge_index(Index* dst, Index* src, bool successive_ids) {
    if (auto ifl = dynamic_cast<IndexFlat*>(dst)) {
        auto ifl2 = dynamic_cast<const IndexFlat*>(src);
        FAISS_ASSERT(ifl2);
        FAISS_ASSERT(successive_ids);
        ifl->add(ifl2->ntotal, ifl2->get_xb());
    } else if (auto ifl = dynamic_cast<IndexIVFFlat*>(dst)) {
        auto ifl2 = dynamic_cast<IndexIVFFlat*>(src);
        FAISS_ASSERT(ifl2);
        ifl->merge_from(*ifl2, successive_ids ? ifl->ntotal : 0);
    } else if (auto ifl = dynamic_cast<IndexIVFScalarQuantizer*>(dst)) {
        auto ifl2 = dynamic_cast<IndexIVFScalarQuantizer*>(src);
        FAISS_ASSERT(ifl2);
        ifl->merge_from(*ifl2, successive_ids ? ifl->ntotal : 0);
    } else if (auto ifl = dynamic_cast<IndexIVFPQ*>(dst)) {
        auto ifl2 = dynamic_cast<IndexIVFPQ*>(src);
        FAISS_ASSERT(ifl2);
        ifl->merge_from(*ifl2, successive_ids ? ifl->ntotal : 0);
    } else {
        FAISS_ASSERT(!"merging not implemented for this type of class");
    }
}

Index* ToCPUCloner::clone_Index(const Index* index) {
    if (auto ifl = dynamic_cast<const GpuIndexFlat*>(index)) {
        IndexFlat* res = new IndexFlat();
        ifl->copyTo(res);
        return res;
    } else if (auto ifl = dynamic_cast<const GpuIndexIVFFlat*>(index)) {
        IndexIVFFlat* res = new IndexIVFFlat();
        ifl->copyTo(res);
        return res;
    } else if (
            auto ifl = dynamic_cast<const GpuIndexIVFScalarQuantizer*>(index)) {
        IndexIVFScalarQuantizer* res = new IndexIVFScalarQuantizer();
        ifl->copyTo(res);
        return res;
    } else if (auto ipq = dynamic_cast<const GpuIndexIVFPQ*>(index)) {
        IndexIVFPQ* res = new IndexIVFPQ();
        ipq->copyTo(res);
        return res;

        // for IndexShards and IndexReplicas we assume that the
        // objective is to make a single component out of them
        // (inverse op of ToGpuClonerMultiple)

    }
#if defined USE_NVIDIA_CUVS
    else if (auto icg = dynamic_cast<const GpuIndexCagra*>(index)) {
        IndexHNSWCagra* res = new IndexHNSWCagra();
        if (icg->get_numeric_type() == faiss::NumericType::Float16) {
            res->base_level_only = true;
        }
        icg->copyTo(res);
        return res;
    }
#endif
    else if (auto ish = dynamic_cast<const IndexShards*>(index)) {
        int nshard = ish->count();
        FAISS_ASSERT(nshard > 0);
        Index* res = clone_Index(ish->at(0));
        for (int i = 1; i < ish->count(); i++) {
            Index* res_i = clone_Index(ish->at(i));
            merge_index(res, res_i, ish->successive_ids);
            delete res_i;
        }
        return res;
    } else if (auto ipr = dynamic_cast<const IndexReplicas*>(index)) {
        // just clone one of the replicas
        FAISS_ASSERT(ipr->count() > 0);
        return clone_Index(ipr->at(0));
    } else {
        return Cloner::clone_Index(index);
    }
}

faiss::Index* index_gpu_to_cpu(const faiss::Index* gpu_index) {
    ToCPUCloner cl;
    return cl.clone_Index(gpu_index);
}

/**********************************************************
 * Cloning to 1 GPU
 **********************************************************/

ToGpuCloner::ToGpuCloner(
        GpuResourcesProvider* prov,
        int device,
        const GpuClonerOptions& options)
        : GpuClonerOptions(options), provider(prov), device(device) {}

Index* ToGpuCloner::clone_Index(const Index* index) {
    if (auto ifl = dynamic_cast<const IndexFlat*>(index)) {
        GpuIndexFlatConfig config;
        config.device = device;
        config.useFloat16 = useFloat16;
        config.use_cuvs = use_cuvs;
        return new GpuIndexFlat(provider, ifl, config);
    } else if (
            dynamic_cast<const IndexScalarQuantizer*>(index) &&
            static_cast<const IndexScalarQuantizer*>(index)->sq.qtype ==
                    ScalarQuantizer::QT_fp16) {
        GpuIndexFlatConfig config;
        config.device = device;
        config.useFloat16 = true;
        FAISS_THROW_IF_NOT_MSG(
                !use_cuvs, "this type of index is not implemented for cuVS");
        GpuIndexFlat* gif = new GpuIndexFlat(
                provider, index->d, index->metric_type, config);
        // transfer data by blocks
        idx_t bs = 1024 * 1024;
        for (idx_t i0 = 0; i0 < index->ntotal; i0 += bs) {
            idx_t i1 = std::min(i0 + bs, index->ntotal);
            std::vector<float> buffer((i1 - i0) * index->d);
            index->reconstruct_n(i0, i1 - i0, buffer.data());
            gif->add(i1 - i0, buffer.data());
        }
        assert(gif->getNumVecs() == index->ntotal);
        return gif;
    } else if (auto ifl = dynamic_cast<const faiss::IndexIVFFlat*>(index)) {
        GpuIndexIVFFlatConfig config;
        config.device = device;
        config.indicesOptions = indicesOptions;
        config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;
        config.use_cuvs = use_cuvs;
        config.allowCpuCoarseQuantizer = allowCpuCoarseQuantizer;

        GpuIndexIVFFlat* res = new GpuIndexIVFFlat(
                provider, ifl->d, ifl->nlist, ifl->metric_type, config);
        if (reserveVecs > 0 && ifl->ntotal == 0) {
            res->reserveMemory(reserveVecs);
        }

        res->copyFrom(ifl);
        return res;
    } else if (
            auto ifl = dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(
                    index)) {
        GpuIndexIVFScalarQuantizerConfig config;
        config.device = device;
        config.indicesOptions = indicesOptions;
        config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;
        FAISS_THROW_IF_NOT_MSG(
                !use_cuvs, "this type of index is not implemented for cuVS");

        GpuIndexIVFScalarQuantizer* res = new GpuIndexIVFScalarQuantizer(
                provider,
                ifl->d,
                ifl->nlist,
                ifl->sq.qtype,
                ifl->metric_type,
                ifl->by_residual,
                config);
        if (reserveVecs > 0 && ifl->ntotal == 0) {
            res->reserveMemory(reserveVecs);
        }

        res->copyFrom(ifl);
        return res;
    } else if (auto ipq = dynamic_cast<const faiss::IndexIVFPQ*>(index)) {
        if (verbose) {
            printf("  IndexIVFPQ size %ld -> GpuIndexIVFPQ "
                   "indicesOptions=%d "
                   "usePrecomputed=%d useFloat16=%d reserveVecs=%ld\n",
                   ipq->ntotal,
                   indicesOptions,
                   usePrecomputed,
                   useFloat16,
                   reserveVecs);
        }
        GpuIndexIVFPQConfig config;
        config.device = device;
        config.indicesOptions = indicesOptions;
        config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;
        config.useFloat16LookupTables = useFloat16;
        config.usePrecomputedTables = usePrecomputed;
        config.use_cuvs = use_cuvs;
        config.interleavedLayout = use_cuvs;
        config.allowCpuCoarseQuantizer = allowCpuCoarseQuantizer;

        GpuIndexIVFPQ* res = new GpuIndexIVFPQ(provider, ipq, config);

        if (reserveVecs > 0 && ipq->ntotal == 0) {
            res->reserveMemory(reserveVecs);
        }

        return res;
    }
#if defined USE_NVIDIA_CUVS
    else if (auto icg = dynamic_cast<const faiss::IndexHNSWCagra*>(index)) {
        GpuIndexCagraConfig config;
        config.device = device;
        GpuIndexCagra* res =
                new GpuIndexCagra(provider, icg->d, icg->metric_type, config);
        res->copyFrom(icg, icg->get_numeric_type());
        return res;
    }
#endif
    else {
        // use CPU cloner for IDMap and PreTransform
        auto index_idmap = dynamic_cast<const IndexIDMap*>(index);
        auto index_pt = dynamic_cast<const IndexPreTransform*>(index);
        if (index_idmap || index_pt) {
            return Cloner::clone_Index(index);
        }
        FAISS_THROW_MSG("This index type is not implemented on GPU.");
    }
}

faiss::Index* index_cpu_to_gpu(
        GpuResourcesProvider* provider,
        int device,
        const faiss::Index* index,
        const GpuClonerOptions* options) {
    GpuClonerOptions defaults;
    ToGpuCloner cl(provider, device, options ? *options : defaults);
    return cl.clone_Index(index);
}

/**********************************************************
 * Cloning to multiple GPUs
 **********************************************************/

ToGpuClonerMultiple::ToGpuClonerMultiple(
        std::vector<GpuResourcesProvider*>& provider,
        std::vector<int>& devices,
        const GpuMultipleClonerOptions& options)
        : GpuMultipleClonerOptions(options) {
    FAISS_THROW_IF_NOT(provider.size() == devices.size());
    for (size_t i = 0; i < provider.size(); i++) {
        sub_cloners.emplace_back(provider[i], devices[i], options);
    }
}

ToGpuClonerMultiple::ToGpuClonerMultiple(
        const std::vector<ToGpuCloner>& sub_cloners,
        const GpuMultipleClonerOptions& options)
        : GpuMultipleClonerOptions(options), sub_cloners(sub_cloners) {}

void ToGpuClonerMultiple::copy_ivf_shard(
        const IndexIVF* index_ivf,
        IndexIVF* idx2,
        idx_t n,
        idx_t i) {
    if (shard_type == 2) {
        idx_t i0 = i * index_ivf->ntotal / n;
        idx_t i1 = (i + 1) * index_ivf->ntotal / n;

        if (verbose)
            printf("IndexShards shard %ld indices %ld:%ld\n", i, i0, i1);
        index_ivf->copy_subset_to(
                *idx2, InvertedLists::SUBSET_TYPE_ID_RANGE, i0, i1);
        FAISS_ASSERT(idx2->ntotal == i1 - i0);
    } else if (shard_type == 1) {
        if (verbose)
            printf("IndexShards shard %ld select modulo %ld = %ld\n", i, n, i);
        index_ivf->copy_subset_to(
                *idx2, InvertedLists::SUBSET_TYPE_ID_MOD, n, i);
    } else if (shard_type == 4) {
        idx_t i0 = i * index_ivf->nlist / n;
        idx_t i1 = (i + 1) * index_ivf->nlist / n;
        if (verbose) {
            printf("IndexShards %ld/%ld select lists %d:%d\n",
                   i,
                   n,
                   int(i0),
                   int(i1));
        }
        index_ivf->copy_subset_to(
                *idx2, InvertedLists::SUBSET_TYPE_INVLIST, i0, i1);
    } else {
        FAISS_THROW_FMT("shard_type %d not implemented", shard_type);
    }
}

Index* ToGpuClonerMultiple::clone_Index_to_shards(const Index* index) {
    idx_t n = sub_cloners.size();

    auto index_ivf = dynamic_cast<const faiss::IndexIVF*>(index);
    auto index_ivfpq = dynamic_cast<const faiss::IndexIVFPQ*>(index);
    auto index_ivfflat = dynamic_cast<const faiss::IndexIVFFlat*>(index);
    auto index_ivfsq =
            dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(index);
    auto index_flat = dynamic_cast<const faiss::IndexFlat*>(index);
    FAISS_THROW_IF_NOT_MSG(
            index_ivfpq || index_ivfflat || index_flat || index_ivfsq,
            "IndexShards implemented only for "
            "IndexIVFFlat, IndexIVFScalarQuantizer, "
            "IndexFlat and IndexIVFPQ");

    // decide what coarse quantizer the sub-indexes are going to have
    const Index* quantizer = nullptr;
    std::unique_ptr<Index> new_quantizer;
    if (index_ivf) {
        quantizer = index_ivf->quantizer;
        if (common_ivf_quantizer &&
            !dynamic_cast<const IndexFlat*>(quantizer)) {
            // then we flatten the coarse quantizer so that everything remains
            // on GPU
            new_quantizer = std::make_unique<IndexFlat>(
                    quantizer->d, quantizer->metric_type);
            std::vector<float> centroids(quantizer->d * quantizer->ntotal);
            quantizer->reconstruct_n(0, quantizer->ntotal, centroids.data());
            new_quantizer->add(quantizer->ntotal, centroids.data());
            quantizer = new_quantizer.get();
        }
    }

    std::vector<faiss::Index*> shards(n);

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        // make a shallow copy
        if (reserveVecs) {
            sub_cloners[i].reserveVecs = (reserveVecs + n - 1) / n;
        }
        // note: const_casts here are harmless because the indexes build here
        // are short-lived, translated immediately to GPU indexes.
        if (index_ivfpq) {
            faiss::IndexIVFPQ idx2(
                    const_cast<Index*>(quantizer),
                    index_ivfpq->d,
                    index_ivfpq->nlist,
                    index_ivfpq->pq.M,
                    index_ivfpq->pq.nbits);
            idx2.metric_type = index_ivfpq->metric_type;
            idx2.pq = index_ivfpq->pq;
            idx2.nprobe = index_ivfpq->nprobe;
            idx2.use_precomputed_table = 0;
            idx2.is_trained = index->is_trained;
            copy_ivf_shard(index_ivfpq, &idx2, n, i);
            shards[i] = sub_cloners[i].clone_Index(&idx2);
        } else if (index_ivfflat) {
            faiss::IndexIVFFlat idx2(
                    const_cast<Index*>(quantizer),
                    index->d,
                    index_ivfflat->nlist,
                    index_ivfflat->metric_type);
            idx2.nprobe = index_ivfflat->nprobe;
            idx2.is_trained = index->is_trained;
            copy_ivf_shard(index_ivfflat, &idx2, n, i);
            shards[i] = sub_cloners[i].clone_Index(&idx2);
        } else if (index_ivfsq) {
            faiss::IndexIVFScalarQuantizer idx2(
                    const_cast<Index*>(quantizer),
                    index->d,
                    index_ivfsq->nlist,
                    index_ivfsq->sq.qtype,
                    index_ivfsq->metric_type,
                    index_ivfsq->by_residual);

            idx2.nprobe = index_ivfsq->nprobe;
            idx2.is_trained = index->is_trained;
            idx2.sq = index_ivfsq->sq;
            copy_ivf_shard(index_ivfsq, &idx2, n, i);
            shards[i] = sub_cloners[i].clone_Index(&idx2);
        } else if (index_flat) {
            faiss::IndexFlat idx2(index->d, index->metric_type);
            shards[i] = sub_cloners[i].clone_Index(&idx2);
            if (index->ntotal > 0) {
                idx_t i0 = index->ntotal * i / n;
                idx_t i1 = index->ntotal * (i + 1) / n;
                shards[i]->add(i1 - i0, index_flat->get_xb() + i0 * index->d);
            }
        }
    }

    bool successive_ids = index_flat != nullptr;
    faiss::IndexShards* res;
    if (common_ivf_quantizer && index_ivf) {
        this->shard = false;
        Index* common_quantizer = clone_Index(index_ivf->quantizer);
        this->shard = true;
        IndexShardsIVF* idx = new faiss::IndexShardsIVF(
                common_quantizer, index_ivf->nlist, true, false);
        idx->own_fields = true;
        idx->own_indices = true;
        res = idx;
    } else {
        res = new faiss::IndexShards(index->d, true, successive_ids);
        res->own_indices = true;
    }

    for (int i = 0; i < n; i++) {
        res->add_shard(shards[i]);
    }
    FAISS_ASSERT(index->ntotal == res->ntotal);
    return res;
}

Index* ToGpuClonerMultiple::clone_Index(const Index* index) {
    idx_t n = sub_cloners.size();
    if (n == 1) {
        return sub_cloners[0].clone_Index(index);
    }

    if (dynamic_cast<const IndexFlat*>(index) ||
        dynamic_cast<const IndexIVFFlat*>(index) ||
        dynamic_cast<const IndexIVFScalarQuantizer*>(index) ||
        dynamic_cast<const IndexIVFPQ*>(index)) {
        if (!shard) {
            IndexReplicas* res = new IndexReplicas();
            for (auto& sub_cloner : sub_cloners) {
                res->addIndex(sub_cloner.clone_Index(index));
            }
            res->own_indices = true;
            return res;
        } else {
            return clone_Index_to_shards(index);
        }
    } else if (auto miq = dynamic_cast<const MultiIndexQuantizer*>(index)) {
        if (verbose) {
            printf("cloning MultiIndexQuantizer: "
                   "will be valid only for search k=1\n");
        }
        const ProductQuantizer& pq = miq->pq;
        IndexSplitVectors* splitv = new IndexSplitVectors(pq.d, true);
        splitv->own_fields = true;

        for (int m = 0; m < pq.M; m++) {
            // which GPU(s) will be assigned to this sub-quantizer

            idx_t i0 = m * n / pq.M;
            idx_t i1 = pq.M <= n ? (m + 1) * n / pq.M : i0 + 1;
            std::vector<ToGpuCloner> sub_cloners_2;
            sub_cloners_2.insert(
                    sub_cloners_2.begin(),
                    sub_cloners.begin() + i0,
                    sub_cloners.begin() + i1);
            ToGpuClonerMultiple cm(sub_cloners_2, *this);
            IndexFlatL2 idxc(pq.dsub);
            idxc.add(pq.ksub, pq.centroids.data() + m * pq.d * pq.ksub);
            Index* idx2 = cm.clone_Index(&idxc);
            splitv->add_sub_index(idx2);
        }
        return splitv;
    } else {
        return Cloner::clone_Index(index);
    }
}

faiss::Index* index_cpu_to_gpu_multiple(
        std::vector<GpuResourcesProvider*>& provider,
        std::vector<int>& devices,
        const faiss::Index* index,
        const GpuMultipleClonerOptions* options) {
    GpuMultipleClonerOptions defaults;
    ToGpuClonerMultiple cl(provider, devices, options ? *options : defaults);
    return cl.clone_Index(index);
}

GpuProgressiveDimIndexFactory::GpuProgressiveDimIndexFactory(int ngpu) {
    FAISS_THROW_IF_NOT(ngpu >= 1);
    devices.resize(ngpu);
    vres.resize(ngpu);

    for (int i = 0; i < ngpu; i++) {
        vres[i] = new StandardGpuResources();
        devices[i] = i;
    }
    ncall = 0;
}

GpuProgressiveDimIndexFactory::~GpuProgressiveDimIndexFactory() {
    for (int i = 0; i < vres.size(); i++) {
        delete vres[i];
    }
}

Index* GpuProgressiveDimIndexFactory::operator()(int dim) {
    IndexFlatL2 index(dim);
    ncall++;
    return index_cpu_to_gpu_multiple(vres, devices, &index, &options);
}

/*********************************************
 * Cloning binary indexes
 *********************************************/

faiss::IndexBinary* index_binary_gpu_to_cpu(
        const faiss::IndexBinary* gpu_index) {
    if (auto ii = dynamic_cast<const GpuIndexBinaryFlat*>(gpu_index)) {
        IndexBinaryFlat* ret = new IndexBinaryFlat();
        ii->copyTo(ret);
        return ret;
    }
#if defined USE_NVIDIA_CUVS
    else if (auto ii = dynamic_cast<const GpuIndexBinaryCagra*>(gpu_index)) {
        IndexBinaryHNSW* ret = new IndexBinaryHNSW();
        ii->copyTo(ret);
        return ret;
    }
#endif
    else {
        FAISS_THROW_MSG("cannot clone this type of index");
    }
}

faiss::IndexBinary* index_binary_cpu_to_gpu(
        GpuResourcesProvider* provider,
        int device,
        const faiss::IndexBinary* index,
        const GpuClonerOptions* options) {
    if (auto ii = dynamic_cast<const IndexBinaryFlat*>(index)) {
        GpuIndexBinaryFlatConfig config;
        config.device = device;
        return new GpuIndexBinaryFlat(provider, ii, config);
    }
#if defined USE_NVIDIA_CUVS
    else if (auto ii = dynamic_cast<const faiss::IndexBinaryHNSW*>(index)) {
        GpuIndexCagraConfig config;
        config.device = device;
        GpuIndexBinaryCagra* res =
                new GpuIndexBinaryCagra(provider, ii->d, config);
        res->copyFrom(ii);
        return res;
    }
#endif
    else {
        FAISS_THROW_MSG("cannot clone this type of index");
    }
}

faiss::IndexBinary* index_binary_cpu_to_gpu_multiple(
        std::vector<GpuResourcesProvider*>& provider,
        std::vector<int>& devices,
        const faiss::IndexBinary* index,
        const GpuMultipleClonerOptions* options) {
    GpuMultipleClonerOptions defaults;
    FAISS_THROW_IF_NOT(devices.size() == provider.size());
    int n = devices.size();
    if (n == 1) {
        return index_binary_cpu_to_gpu(provider[0], devices[0], index, options);
    }
    if (!options) {
        options = &defaults;
    }
    if (options->shard) {
        auto* fi = dynamic_cast<const IndexBinaryFlat*>(index);
        FAISS_THROW_IF_NOT_MSG(fi, "only flat index cloning supported");
        IndexBinaryShards* ret = new IndexBinaryShards(true, true);
        for (int i = 0; i < n; i++) {
            IndexBinaryFlat fig(fi->d);
            size_t i0 = i * fi->ntotal / n;
            size_t i1 = (i + 1) * fi->ntotal / n;
            fig.add(i1 - i0, fi->xb.data() + i0 * fi->code_size);
            ret->addIndex(index_binary_cpu_to_gpu(
                    provider[i], devices[i], &fig, options));
        }
        ret->own_indices = true;
        return ret;
    } else { // replicas
        IndexBinaryReplicas* ret = new IndexBinaryReplicas(true);
        for (int i = 0; i < n; i++) {
            ret->addIndex(index_binary_cpu_to_gpu(
                    provider[i], devices[i], index, options));
        }
        ret->own_indices = true;
        return ret;
    }
}

} // namespace gpu
} // namespace faiss
