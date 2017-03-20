
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "GpuAutoTune.h"


#include "GpuIndex.h"
#include "../FaissAssert.h"
#include "../index_io.h"
#include "../IndexFlat.h"
#include "../IndexIVF.h"
#include "../IndexIVFPQ.h"
#include "../VectorTransform.h"
#include "../MetaIndexes.h"
#include "GpuIndexFlat.h"
#include "GpuIndexIVFFlat.h"
#include "GpuIndexIVFPQ.h"
#include "IndexProxy.h"

namespace faiss { namespace gpu {

/**********************************************************
 * Cloning from/to GPU
 **********************************************************/


struct ToCPUCloner: Cloner {

    Index *clone_Index(const Index *index) override {
        if(auto ifl = dynamic_cast<const GpuIndexFlat *>(index)) {
            IndexFlat *res = new IndexFlat();
            ifl->copyTo(res);
            return res;
        } else if(auto ifl = dynamic_cast<const GpuIndexIVFFlat *>(index)) {
            IndexIVFFlat *res = new IndexIVFFlat();
            ifl->copyTo(res);
            return res;
        } else if(auto ipq = dynamic_cast<const GpuIndexIVFPQ *>(index)) {
            IndexIVFPQ *res = new IndexIVFPQ();
            ipq->copyTo(res);
            return res;
        } else {
            return Cloner::clone_Index(index);
        }
    }
};

faiss::Index * index_gpu_to_cpu(const faiss::Index *gpu_index)
{
    ToCPUCloner cl;
    return cl.clone_Index(gpu_index);
}



GpuClonerOptions::GpuClonerOptions():
    indicesOptions(INDICES_64_BIT),
    useFloat16CoarseQuantizer(false),
    useFloat16(false),
    usePrecomputed(true),
    reserveVecs(0),
    storeTransposed(false),
    verbose(0)
{}


struct ToGpuCloner: faiss::Cloner, GpuClonerOptions {
    GpuResources *resources;
    int device;

    ToGpuCloner(GpuResources *resources, int device, const GpuClonerOptions &options):
        GpuClonerOptions(options), resources(resources), device(device)
    {}

    Index *clone_Index(const Index *index) override {
        if(auto ifl = dynamic_cast<const IndexFlat *>(index)) {
          GpuIndexFlatConfig config;
          config.device = device;
          config.useFloat16 = useFloat16;
          config.storeTransposed = storeTransposed;

          return new GpuIndexFlat(resources, ifl, config);
        } else if(auto ifl = dynamic_cast<const faiss::IndexIVFFlat *>(index)) {
          GpuIndexIVFFlat *res =
            new GpuIndexIVFFlat(resources,
                                device,
                                useFloat16CoarseQuantizer,
                                useFloat16,
                                ifl->d,
                                ifl->nlist,
                                indicesOptions,
                                ifl->metric_type);
          if(reserveVecs > 0 && ifl->ntotal == 0)
              res->reserveMemory(reserveVecs);
          res->copyFrom(ifl);
          return res;
        } else if(auto ipq = dynamic_cast<const faiss::IndexIVFPQ *>(index)) {
            if(verbose)
                printf("  IndexIVFPQ size %ld -> GpuIndexIVFPQ indicesOptions=%d "
                       "usePrecomputed=%d useFloat16=%d reserveVecs=%ld\n",
                       ipq->ntotal, indicesOptions, usePrecomputed,
                       useFloat16, reserveVecs);
            GpuIndexIVFPQ *res = new GpuIndexIVFPQ(
                resources, device, indicesOptions, useFloat16,
                ipq);
            res->setPrecomputedCodes(usePrecomputed);
            if(reserveVecs > 0 && ipq->ntotal == 0)
                res->reserveMemory(reserveVecs);
            return res;
        } else {
            return Cloner::clone_Index(index);
        }
    }

};


faiss::Index * index_cpu_to_gpu(
       GpuResources* resources, int device,
       const faiss::Index *index,
       const GpuClonerOptions *options)
{
    GpuClonerOptions defaults;
    ToGpuCloner cl(resources, device, options ? *options : defaults);
    return cl.clone_Index(index);
}

GpuMultipleClonerOptions::GpuMultipleClonerOptions(): shard(false)
{}

struct ToGpuClonerMultiple: faiss::Cloner, GpuMultipleClonerOptions {
    std::vector<ToGpuCloner> sub_cloners;

    ToGpuClonerMultiple(std::vector<GpuResources *> & resources,
                        std::vector<int>& devices,
                        const GpuMultipleClonerOptions &options):
        GpuMultipleClonerOptions(options)
    {
        FAISS_ASSERT(resources.size() == devices.size());
        for(int i = 0; i < resources.size(); i++) {
            sub_cloners.push_back(ToGpuCloner(
                     resources[i], devices[i], options));
        }
    }


    ToGpuClonerMultiple(const std::vector<ToGpuCloner> & sub_cloners,
                        const GpuMultipleClonerOptions &options):
        GpuMultipleClonerOptions(options),
        sub_cloners(sub_cloners)
    {}


    Index *clone_Index(const Index *index) override {
        long n = sub_cloners.size();

        if (n == 1)
            return sub_cloners[0].clone_Index(index);

        if(dynamic_cast<const IndexFlat *>(index) ||
           dynamic_cast<const faiss::IndexIVFFlat *>(index) ||
           dynamic_cast<const faiss::IndexIVFPQ *>(index)) {
            if(!shard) {
                IndexProxy * res = new IndexProxy();
                for(auto & sub_cloner: sub_cloners) {
                    res->addIndex(sub_cloner.clone_Index(index));
                }
                res->own_fields = true;
                return res;
            } else {
                auto index_ivfpq =
                    dynamic_cast<const faiss::IndexIVFPQ *>(index);
                auto index_ivfflat =
                    dynamic_cast<const faiss::IndexIVFFlat *>(index);
                FAISS_ASSERT (index_ivfpq || index_ivfflat ||
                              !"IndexShards implemented only for "
                              "IndexIVFFlat or IndexIVFPQ");
                std::vector<faiss::Index*> shards(n);

                for(long i = 0; i < n; i++) {
                    // make a shallow copy
                    long i0 = i * index->ntotal / n;
                    long i1 = (i + 1) * index->ntotal / n;
                    if(verbose)
                        printf("IndexShards shard %ld indices %ld:%ld\n",
                               i, i0, i1);

                    if(reserveVecs)
                        sub_cloners[i].reserveVecs =
                            (reserveVecs + n - 1) / n;

                    if (index_ivfpq) {
                        faiss::IndexIVFPQ idx2(
                              index_ivfpq->quantizer, index_ivfpq->d,
                              index_ivfpq->nlist, index_ivfpq->code_size,
                              index_ivfpq->pq.nbits);
                        idx2.pq = index_ivfpq->pq;
                        idx2.use_precomputed_table = 0;
                        idx2.is_trained = index->is_trained;
                        index_ivfpq->copy_subset_to(idx2, 0, i0, i1);
                        shards[i] = sub_cloners[i].clone_Index(&idx2);
                    } else if (index_ivfflat) {
                        faiss::IndexIVFFlat idx2(
                              index_ivfflat->quantizer, index->d,
                              index_ivfflat->nlist, index_ivfflat->metric_type);
                        index_ivfflat->copy_subset_to(idx2, 0, i0, i1);
                        shards[i] = sub_cloners[i].clone_Index(&idx2);
                    }
                }
                faiss::IndexShards *res =
                    new faiss::IndexShards(index->d, true, false);

                for (int i = 0; i < n; i++) {
                    res->add_shard(shards[i]);
                }
                res->own_fields = true;
                assert(index->ntotal == res->ntotal);
                return res;
            }
        } else if(auto miq = dynamic_cast<const MultiIndexQuantizer *>(index)) {
            if (verbose) {
                printf("cloning MultiIndexQuantizer: "
                       "will be valid only for search k=1\n");
            }
            const ProductQuantizer & pq = miq->pq;
            IndexSplitVectors *splitv = new IndexSplitVectors(pq.d, true);
            splitv->own_fields = true;

            for (int m = 0; m < pq.M; m++) {
                // which GPU(s) will be assigned to this sub-quantizer

                long i0 = m * n / pq.M;
                long i1 = pq.M <= n ? (m + 1) * n / pq.M : i0 + 1;
                std::vector<ToGpuCloner> sub_cloners_2;
                sub_cloners_2.insert(
                      sub_cloners_2.begin(), sub_cloners.begin() + i0,
                      sub_cloners.begin() + i1);
                ToGpuClonerMultiple cm(sub_cloners_2, *this);
                IndexFlatL2 idxc (pq.dsub);
                idxc.add (pq.ksub, pq.centroids.data() + m * pq.d * pq.ksub);
                Index *idx2 = cm.clone_Index(&idxc);
                splitv->add_sub_index(idx2);
            }
            return splitv;
        } else {
            return Cloner::clone_Index(index);
        }
    }


};



faiss::Index * index_cpu_to_gpu_multiple(
       std::vector<GpuResources*> & resources,
       std::vector<int> &devices,
       const faiss::Index *index,
       const GpuMultipleClonerOptions *options)
{
    GpuMultipleClonerOptions defaults;
    ToGpuClonerMultiple cl(resources, devices, options ? *options : defaults);
    return cl.clone_Index(index);
}



/**********************************************************
 * Parameters to auto-tune on GpuIndex'es
 **********************************************************/

#define DC(classname) auto ix = dynamic_cast<const classname *>(index)


void GpuParameterSpace::initialize (const Index * index)
{
    if (DC (IndexPreTransform)) {
        index = ix->index;
    }
    if (DC (IndexProxy)) {
        if (ix->count() == 0) return;
        index = ix->at(0);
    }
    if (DC (faiss::IndexShards)) {
        if (ix->shard_indexes.size() == 0) return;
        index = ix->shard_indexes[0];
    }
    if (DC (GpuIndexIVF)) {
        ParameterRange & pr = add_range("nprobe");
        for (int i = 0; i < 12; i++) {
            size_t nprobe = 1 << i;
            if (nprobe >= ix->getNumLists() ||
                nprobe > 1024) break;
            pr.values.push_back (nprobe);
        }
    }
    // not sure we should call the parent initializer
}



#undef DC
// non-const version
#define DC(classname) auto *ix = dynamic_cast<classname *>(index)



void GpuParameterSpace::set_index_parameter (
        Index * index, const std::string & name, double val) const
{
    if (DC (IndexPreTransform)) {
        index = ix->index;
    }
    if (DC (IndexProxy)) {
        for (int i = 0; i < ix->count(); i++)
            set_index_parameter (ix->at(i), name, val);
        return;
    }
    if (DC (faiss::IndexShards)) {
        for (auto sub_index : ix->shard_indexes)
            set_index_parameter (sub_index, name, val);
        return;
    }
    if (name == "nprobe") {
        DC (GpuIndexIVF);
        FAISS_ASSERT(ix);
        ix->setNumProbes (int (val));
        return;
    }
    FAISS_ASSERT (!"unknown parameter");
}




} } // namespace
