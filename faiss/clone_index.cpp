/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/clone_index.h>

#include <cstdio>
#include <cstdlib>

#include <faiss/impl/FaissAssert.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexAdditiveQuantizerFastScan.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFAdditiveQuantizerFastScan.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexIVFPQR.h>
#include <faiss/IndexIVFSpectralHash.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexLattice.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexRowwiseMinMax.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <faiss/VectorTransform.h>

#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>

#include <faiss/invlists/BlockInvertedLists.h>

namespace faiss {

/*************************************************************
 * cloning functions
 **************************************************************/

Index* clone_index(const Index* index) {
    Cloner cl;
    return cl.clone_Index(index);
}

// assumes there is a copy constructor ready. Always try from most
// specific to most general. Most indexes don't have complicated
// structs, the default copy constructor often just works.
#define TRYCLONE(classname, obj)                                      \
    if (const classname* clo = dynamic_cast<const classname*>(obj)) { \
        return new classname(*clo);                                   \
    } else

VectorTransform* Cloner::clone_VectorTransform(const VectorTransform* vt) {
    TRYCLONE(RemapDimensionsTransform, vt)
    TRYCLONE(OPQMatrix, vt)
    TRYCLONE(PCAMatrix, vt)
    TRYCLONE(ITQMatrix, vt)
    TRYCLONE(RandomRotationMatrix, vt)
    TRYCLONE(LinearTransform, vt) {
        FAISS_THROW_MSG("clone not supported for this type of VectorTransform");
    }
    return nullptr;
}

IndexIVF* Cloner::clone_IndexIVF(const IndexIVF* ivf) {
    TRYCLONE(IndexIVFPQR, ivf)
    TRYCLONE(IndexIVFPQ, ivf)

    TRYCLONE(IndexIVFLocalSearchQuantizer, ivf)
    TRYCLONE(IndexIVFProductLocalSearchQuantizer, ivf)
    TRYCLONE(IndexIVFProductResidualQuantizer, ivf)
    TRYCLONE(IndexIVFResidualQuantizer, ivf)

    TRYCLONE(IndexIVFLocalSearchQuantizerFastScan, ivf)
    TRYCLONE(IndexIVFProductLocalSearchQuantizerFastScan, ivf)
    TRYCLONE(IndexIVFProductResidualQuantizerFastScan, ivf)
    TRYCLONE(IndexIVFResidualQuantizerFastScan, ivf)
    TRYCLONE(IndexIVFPQFastScan, ivf)

    TRYCLONE(IndexIVFFlatDedup, ivf)
    TRYCLONE(IndexIVFFlat, ivf)

    TRYCLONE(IndexIVFSpectralHash, ivf)

    TRYCLONE(IndexIVFScalarQuantizer, ivf) {
        FAISS_THROW_MSG("clone not supported for this type of IndexIVF");
    }
    return nullptr;
}

IndexRefine* clone_IndexRefine(const IndexRefine* ir) {
    TRYCLONE(IndexRefineFlat, ir)
    TRYCLONE(IndexRefine, ir) {
        FAISS_THROW_MSG("clone not supported for this type of IndexRefine");
    }
}

IndexIDMap* clone_IndexIDMap(const IndexIDMap* im) {
    TRYCLONE(IndexIDMap2, im)
    TRYCLONE(IndexIDMap, im) {
        FAISS_THROW_MSG("clone not supported for this type of IndexIDMap");
    }
}

IndexHNSW* clone_IndexHNSW(const IndexHNSW* ihnsw) {
    TRYCLONE(IndexHNSW2Level, ihnsw)
    TRYCLONE(IndexHNSWFlat, ihnsw)
    TRYCLONE(IndexHNSWPQ, ihnsw)
    TRYCLONE(IndexHNSWSQ, ihnsw)
    TRYCLONE(IndexHNSW, ihnsw) {
        FAISS_THROW_MSG("clone not supported for this type of IndexHNSW");
    }
}

IndexNNDescent* clone_IndexNNDescent(const IndexNNDescent* innd) {
    TRYCLONE(IndexNNDescentFlat, innd)
    TRYCLONE(IndexNNDescent, innd) {
        FAISS_THROW_MSG("clone not supported for this type of IndexNNDescent");
    }
}

IndexNSG* clone_IndexNSG(const IndexNSG* insg) {
    TRYCLONE(IndexNSGFlat, insg)
    TRYCLONE(IndexNSGPQ, insg)
    TRYCLONE(IndexNSGSQ, insg)
    TRYCLONE(IndexNSG, insg) {
        FAISS_THROW_MSG("clone not supported for this type of IndexNNDescent");
    }
}

IndexRowwiseMinMaxBase* clone_IndexRowwiseMinMax(
        const IndexRowwiseMinMaxBase* irmmb) {
    TRYCLONE(IndexRowwiseMinMaxFP16, irmmb)
    TRYCLONE(IndexRowwiseMinMax, irmmb) {
        FAISS_THROW_MSG(
                "clone not supported for this type of IndexRowwiseMinMax");
    }
}

#define TRYCAST(classname) classname* res = dynamic_cast<classname*>(index)

void reset_AdditiveQuantizerIndex(Index* index) {
    auto clone_ProductQuantizers =
            [](std::vector<AdditiveQuantizer*>& quantizers) {
                for (auto& q : quantizers) {
                    q = dynamic_cast<AdditiveQuantizer*>(clone_Quantizer(q));
                }
            };
    if (TRYCAST(IndexIVFLocalSearchQuantizerFastScan)) {
        res->aq = &res->lsq;
    } else if (TRYCAST(IndexIVFResidualQuantizerFastScan)) {
        res->aq = &res->rq;
    } else if (TRYCAST(IndexIVFProductLocalSearchQuantizerFastScan)) {
        res->aq = &res->plsq;
        clone_ProductQuantizers(res->plsq.quantizers);
    } else if (TRYCAST(IndexIVFProductResidualQuantizerFastScan)) {
        res->aq = &res->prq;
        clone_ProductQuantizers(res->prq.quantizers);
    } else if (TRYCAST(IndexIVFLocalSearchQuantizer)) {
        res->aq = &res->lsq;
    } else if (TRYCAST(IndexIVFResidualQuantizer)) {
        res->aq = &res->rq;
    } else if (TRYCAST(IndexIVFProductLocalSearchQuantizer)) {
        res->aq = &res->plsq;
        clone_ProductQuantizers(res->plsq.quantizers);
    } else if (TRYCAST(IndexIVFProductResidualQuantizer)) {
        res->aq = &res->prq;
        clone_ProductQuantizers(res->prq.quantizers);
    } else if (TRYCAST(IndexLocalSearchQuantizerFastScan)) {
        res->aq = &res->lsq;
    } else if (TRYCAST(IndexResidualQuantizerFastScan)) {
        res->aq = &res->rq;
    } else if (TRYCAST(IndexProductLocalSearchQuantizerFastScan)) {
        res->aq = &res->plsq;
        clone_ProductQuantizers(res->plsq.quantizers);
    } else if (TRYCAST(IndexProductResidualQuantizerFastScan)) {
        res->aq = &res->prq;
        clone_ProductQuantizers(res->prq.quantizers);
    } else if (TRYCAST(IndexLocalSearchQuantizer)) {
        res->aq = &res->lsq;
    } else if (TRYCAST(IndexResidualQuantizer)) {
        res->aq = &res->rq;
    } else if (TRYCAST(IndexProductLocalSearchQuantizer)) {
        res->aq = &res->plsq;
        clone_ProductQuantizers(res->plsq.quantizers);
    } else if (TRYCAST(IndexProductResidualQuantizer)) {
        res->aq = &res->prq;
        clone_ProductQuantizers(res->prq.quantizers);
    } else if (TRYCAST(LocalSearchCoarseQuantizer)) {
        res->aq = &res->lsq;
    } else if (TRYCAST(ResidualCoarseQuantizer)) {
        res->aq = &res->rq;
    } else {
        FAISS_THROW_MSG(
                "clone not supported for this type of additive quantizer index");
    }
}

Index* clone_AdditiveQuantizerIndex(const Index* index) {
    // IndexAdditiveQuantizer
    TRYCLONE(IndexResidualQuantizer, index)
    TRYCLONE(IndexProductResidualQuantizer, index)
    TRYCLONE(IndexLocalSearchQuantizer, index)
    TRYCLONE(IndexProductLocalSearchQuantizer, index)

    // IndexFastScan
    TRYCLONE(IndexResidualQuantizerFastScan, index)
    TRYCLONE(IndexLocalSearchQuantizerFastScan, index)
    TRYCLONE(IndexProductResidualQuantizerFastScan, index)
    TRYCLONE(IndexProductLocalSearchQuantizerFastScan, index)

    // AdditiveCoarseQuantizer
    TRYCLONE(ResidualCoarseQuantizer, index)
    TRYCLONE(LocalSearchCoarseQuantizer, index) {
        FAISS_THROW_MSG(
                "clone not supported for this type of additive quantizer index");
    }
}

namespace {

IndexHNSW* clone_HNSW(const IndexHNSW* ihnsw) {
    TRYCLONE(IndexHNSWFlat, ihnsw)
    TRYCLONE(IndexHNSWPQ, ihnsw)
    TRYCLONE(IndexHNSWSQ, ihnsw)
    return new IndexHNSW(*ihnsw);
}

} // anonymous namespace

Index* Cloner::clone_Index(const Index* index) {
    TRYCLONE(IndexPQ, index)
    TRYCLONE(IndexLSH, index)

    // IndexFlat
    TRYCLONE(IndexFlat1D, index)
    TRYCLONE(IndexFlatL2, index)
    TRYCLONE(IndexFlatIP, index)
    TRYCLONE(IndexFlat, index)

    TRYCLONE(IndexLattice, index)
    TRYCLONE(IndexRandom, index)
    TRYCLONE(IndexPQFastScan, index)

    TRYCLONE(IndexScalarQuantizer, index)
    TRYCLONE(MultiIndexQuantizer, index)

    if (const IndexIVF* ivf = dynamic_cast<const IndexIVF*>(index)) {
        IndexIVF* res = clone_IndexIVF(ivf);
        if (ivf->invlists == nullptr) {
            res->invlists = nullptr;
        } else if (
                auto* ails = dynamic_cast<const ArrayInvertedLists*>(
                        ivf->invlists)) {
            res->invlists = new ArrayInvertedLists(*ails);
            res->own_invlists = true;
        } else if (
                auto* bils = dynamic_cast<const BlockInvertedLists*>(
                        ivf->invlists)) {
            res->invlists = new BlockInvertedLists(*bils);
            res->own_invlists = true;
        } else {
            FAISS_THROW_MSG(
                    "clone not supported for this type of inverted lists");
        }
        res->own_fields = true;
        res->quantizer = clone_Index(ivf->quantizer);

        if (dynamic_cast<const IndexIVFAdditiveQuantizerFastScan*>(res) ||
            dynamic_cast<const IndexIVFAdditiveQuantizer*>(res)) {
            reset_AdditiveQuantizerIndex(res);
        }
        return res;
    } else if (
            const IndexPreTransform* ipt =
                    dynamic_cast<const IndexPreTransform*>(index)) {
        IndexPreTransform* res = new IndexPreTransform();
        res->d = ipt->d;
        res->ntotal = ipt->ntotal;
        res->is_trained = ipt->is_trained;
        res->metric_type = ipt->metric_type;
        res->metric_arg = ipt->metric_arg;

        res->index = clone_Index(ipt->index);
        for (int i = 0; i < ipt->chain.size(); i++)
            res->chain.push_back(clone_VectorTransform(ipt->chain[i]));
        res->own_fields = true;
        return res;
    } else if (
            const IndexIDMap* idmap = dynamic_cast<const IndexIDMap*>(index)) {
        IndexIDMap* res = clone_IndexIDMap(idmap);
        res->own_fields = true;
        res->index = clone_Index(idmap->index);
        return res;
    } else if (const IndexHNSW* ihnsw = dynamic_cast<const IndexHNSW*>(index)) {
        IndexHNSW* res = clone_IndexHNSW(ihnsw);
        res->own_fields = true;
        // make sure we don't get a GPU index here
        res->storage = Cloner::clone_Index(ihnsw->storage);
        return res;
    } else if (const IndexNSG* insg = dynamic_cast<const IndexNSG*>(index)) {
        IndexNSG* res = clone_IndexNSG(insg);

        // copy the dynamic allocated graph
        auto& new_graph = res->nsg.final_graph;
        auto& old_graph = insg->nsg.final_graph;
        new_graph = std::make_shared<nsg::Graph<int>>(*old_graph);

        res->own_fields = true;
        res->storage = clone_Index(insg->storage);
        return res;
    } else if (
            const IndexNNDescent* innd =
                    dynamic_cast<const IndexNNDescent*>(index)) {
        IndexNNDescent* res = clone_IndexNNDescent(innd);
        res->own_fields = true;
        res->storage = clone_Index(innd->storage);
        return res;
    } else if (
            const Index2Layer* i2l = dynamic_cast<const Index2Layer*>(index)) {
        Index2Layer* res = new Index2Layer(*i2l);
        res->q1.own_fields = true;
        res->q1.quantizer = clone_Index(i2l->q1.quantizer);
        return res;
    } else if (
            const IndexRefine* ir = dynamic_cast<const IndexRefine*>(index)) {
        IndexRefine* res = clone_IndexRefine(ir);
        res->own_fields = true;
        res->base_index = clone_Index(ir->base_index);
        if (ir->refine_index != nullptr) {
            res->own_refine_index = true;
            res->refine_index = clone_Index(ir->refine_index);
        }
        return res;
    } else if (
            const IndexRowwiseMinMaxBase* irmmb =
                    dynamic_cast<const IndexRowwiseMinMaxBase*>(index)) {
        IndexRowwiseMinMaxBase* res = clone_IndexRowwiseMinMax(irmmb);
        res->own_fields = true;
        res->index = clone_Index(irmmb->index);
    } else if (
            dynamic_cast<const IndexAdditiveQuantizerFastScan*>(index) ||
            dynamic_cast<const IndexAdditiveQuantizer*>(index) ||
            dynamic_cast<const AdditiveCoarseQuantizer*>(index)) {
        Index* res = clone_AdditiveQuantizerIndex(index);
        reset_AdditiveQuantizerIndex(res);
        return res;
    } else {
        FAISS_THROW_FMT(
                "clone not supported for this Index type %s",
                typeid(*index).name());
    }
    return nullptr;
}

Quantizer* clone_Quantizer(const Quantizer* quant) {
    TRYCLONE(ResidualQuantizer, quant)
    TRYCLONE(LocalSearchQuantizer, quant)
    TRYCLONE(ProductQuantizer, quant)
    TRYCLONE(ScalarQuantizer, quant)
    FAISS_THROW_MSG("Did not recognize quantizer to clone");
}

} // namespace faiss
