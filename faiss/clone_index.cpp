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
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQR.h>
#include <faiss/IndexIVFSpectralHash.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexLattice.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <faiss/VectorTransform.h>

#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>

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
    TRYCLONE(IndexIVFFlat, ivf)
    TRYCLONE(IndexIVFScalarQuantizer, ivf) {
        FAISS_THROW_MSG("clone not supported for this type of IndexIVF");
    }
    return nullptr;
}

Index* Cloner::clone_Index(const Index* index) {
    TRYCLONE(IndexPQ, index)
    TRYCLONE(IndexLSH, index)
    TRYCLONE(IndexFlatL2, index)
    TRYCLONE(IndexFlatIP, index)
    TRYCLONE(IndexFlat, index)
    TRYCLONE(IndexLattice, index)
    TRYCLONE(IndexResidualQuantizer, index)
    TRYCLONE(IndexScalarQuantizer, index)
    TRYCLONE(MultiIndexQuantizer, index)
    TRYCLONE(ResidualCoarseQuantizer, index)
    if (const IndexIVF* ivf = dynamic_cast<const IndexIVF*>(index)) {
        IndexIVF* res = clone_IndexIVF(ivf);
        if (ivf->invlists == nullptr) {
            res->invlists = nullptr;
        } else if (
                auto* ails = dynamic_cast<const ArrayInvertedLists*>(
                        ivf->invlists)) {
            res->invlists = new ArrayInvertedLists(*ails);
            res->own_invlists = true;
        } else {
            FAISS_THROW_MSG(
                    "clone not supported for this type of inverted lists");
        }
        res->own_fields = true;
        res->quantizer = clone_Index(ivf->quantizer);
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
        const IndexIDMap2* idmap2 = dynamic_cast<const IndexIDMap2*>(index);
        IndexIDMap* res =
                idmap2 ? new IndexIDMap2(*idmap2) : new IndexIDMap(*idmap);
        res->own_fields = true;
        res->index = clone_Index(idmap->index);
        return res;
    } else if (const IndexHNSW* ihnsw = dynamic_cast<const IndexHNSW*>(index)) {
        IndexHNSW* res = new IndexHNSW(*ihnsw);
        res->own_fields = true;
        res->storage = clone_Index(ihnsw->storage);
        return res;
    } else if (const IndexNSG* insg = dynamic_cast<const IndexNSG*>(index)) {
        IndexNSG* res = new IndexNSG(*insg);

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
        IndexNNDescent* res = new IndexNNDescent(*innd);
        res->own_fields = true;
        res->storage = clone_Index(innd->storage);
        return res;
    } else if (
            const Index2Layer* i2l = dynamic_cast<const Index2Layer*>(index)) {
        Index2Layer* res = new Index2Layer(*i2l);
        res->q1.own_fields = true;
        res->q1.quantizer = clone_Index(i2l->q1.quantizer);
        return res;
    } else {
        FAISS_THROW_MSG("clone not supported for this type of Index");
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
