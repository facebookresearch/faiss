/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/index_io.h>

#include <faiss/impl/io_macros.h>

#include <cstdio>
#include <cstdlib>

#include <sys/stat.h>
#include <sys/types.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
#include <faiss/utils/hamming.h>

#include <faiss/invlists/InvertedListsIOHook.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>
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
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <faiss/VectorTransform.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryFromFloat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/IndexBinaryIVF.h>

namespace faiss {

/*************************************************************
 * Read
 **************************************************************/

static void read_index_header(Index* idx, IOReader* f) {
    READ1(idx->d);
    READ1(idx->ntotal);
    Index::idx_t dummy;
    READ1(dummy);
    READ1(dummy);
    READ1(idx->is_trained);
    READ1(idx->metric_type);
    if (idx->metric_type > 1) {
        READ1(idx->metric_arg);
    }
    idx->verbose = false;
}

VectorTransform* read_VectorTransform(IOReader* f) {
    uint32_t h;
    READ1(h);
    VectorTransform* vt = nullptr;

    if (h == fourcc("rrot") || h == fourcc("PCAm") || h == fourcc("LTra") ||
        h == fourcc("PcAm") || h == fourcc("Viqm") || h == fourcc("Pcam")) {
        LinearTransform* lt = nullptr;
        if (h == fourcc("rrot")) {
            lt = new RandomRotationMatrix();
        } else if (
                h == fourcc("PCAm") || h == fourcc("PcAm") ||
                h == fourcc("Pcam")) {
            PCAMatrix* pca = new PCAMatrix();
            READ1(pca->eigen_power);
            if (h == fourcc("Pcam")) {
                READ1(pca->epsilon);
            }
            READ1(pca->random_rotation);
            if (h != fourcc("PCAm")) {
                READ1(pca->balanced_bins);
            }
            READVECTOR(pca->mean);
            READVECTOR(pca->eigenvalues);
            READVECTOR(pca->PCAMat);
            lt = pca;
        } else if (h == fourcc("Viqm")) {
            ITQMatrix* itqm = new ITQMatrix();
            READ1(itqm->max_iter);
            READ1(itqm->seed);
            lt = itqm;
        } else if (h == fourcc("LTra")) {
            lt = new LinearTransform();
        }
        READ1(lt->have_bias);
        READVECTOR(lt->A);
        READVECTOR(lt->b);
        FAISS_THROW_IF_NOT(lt->A.size() >= lt->d_in * lt->d_out);
        FAISS_THROW_IF_NOT(!lt->have_bias || lt->b.size() >= lt->d_out);
        lt->set_is_orthonormal();
        vt = lt;
    } else if (h == fourcc("RmDT")) {
        RemapDimensionsTransform* rdt = new RemapDimensionsTransform();
        READVECTOR(rdt->map);
        vt = rdt;
    } else if (h == fourcc("VNrm")) {
        NormalizationTransform* nt = new NormalizationTransform();
        READ1(nt->norm);
        vt = nt;
    } else if (h == fourcc("VCnt")) {
        CenteringTransform* ct = new CenteringTransform();
        READVECTOR(ct->mean);
        vt = ct;
    } else if (h == fourcc("Viqt")) {
        ITQTransform* itqt = new ITQTransform();

        READVECTOR(itqt->mean);
        READ1(itqt->do_pca);
        {
            ITQMatrix* itqm = dynamic_cast<ITQMatrix*>(read_VectorTransform(f));
            FAISS_THROW_IF_NOT(itqm);
            itqt->itq = *itqm;
            delete itqm;
        }
        {
            LinearTransform* pi =
                    dynamic_cast<LinearTransform*>(read_VectorTransform(f));
            FAISS_THROW_IF_NOT(pi);
            itqt->pca_then_itq = *pi;
            delete pi;
        }
        vt = itqt;
    } else {
        FAISS_THROW_FMT(
                "fourcc %ud (\"%s\") not recognized in %s",
                h,
                fourcc_inv_printable(h).c_str(),
                f->name.c_str());
    }
    READ1(vt->d_in);
    READ1(vt->d_out);
    READ1(vt->is_trained);
    return vt;
}

static void read_ArrayInvertedLists_sizes(
        IOReader* f,
        std::vector<size_t>& sizes) {
    uint32_t list_type;
    READ1(list_type);
    if (list_type == fourcc("full")) {
        size_t os = sizes.size();
        READVECTOR(sizes);
        FAISS_THROW_IF_NOT(os == sizes.size());
    } else if (list_type == fourcc("sprs")) {
        std::vector<size_t> idsizes;
        READVECTOR(idsizes);
        for (size_t j = 0; j < idsizes.size(); j += 2) {
            FAISS_THROW_IF_NOT(idsizes[j] < sizes.size());
            sizes[idsizes[j]] = idsizes[j + 1];
        }
    } else {
        FAISS_THROW_FMT(
                "list_type %ud (\"%s\") not recognized",
                list_type,
                fourcc_inv_printable(list_type).c_str());
    }
}

InvertedLists* read_InvertedLists(IOReader* f, int io_flags) {
    uint32_t h;
    READ1(h);
    if (h == fourcc("il00")) {
        fprintf(stderr,
                "read_InvertedLists:"
                " WARN! inverted lists not stored with IVF object\n");
        return nullptr;
    } else if (h == fourcc("ilar") && !(io_flags & IO_FLAG_SKIP_IVF_DATA)) {
        auto ails = new ArrayInvertedLists(0, 0);
        READ1(ails->nlist);
        READ1(ails->code_size);
        ails->ids.resize(ails->nlist);
        ails->codes.resize(ails->nlist);
        std::vector<size_t> sizes(ails->nlist);
        read_ArrayInvertedLists_sizes(f, sizes);
        for (size_t i = 0; i < ails->nlist; i++) {
            ails->ids[i].resize(sizes[i]);
            ails->codes[i].resize(sizes[i] * ails->code_size);
        }
        for (size_t i = 0; i < ails->nlist; i++) {
            size_t n = ails->ids[i].size();
            if (n > 0) {
                READANDCHECK(ails->codes[i].data(), n * ails->code_size);
                READANDCHECK(ails->ids[i].data(), n);
            }
        }
        return ails;

    } else if (h == fourcc("ilar") && (io_flags & IO_FLAG_SKIP_IVF_DATA)) {
        // code is always ilxx where xx is specific to the type of invlists we
        // want so we get the 16 high bits from the io_flag and the 16 low bits
        // as "il"
        int h2 = (io_flags & 0xffff0000) | (fourcc("il__") & 0x0000ffff);
        size_t nlist, code_size;
        READ1(nlist);
        READ1(code_size);
        std::vector<size_t> sizes(nlist);
        read_ArrayInvertedLists_sizes(f, sizes);
        return InvertedListsIOHook::lookup(h2)->read_ArrayInvertedLists(
                f, io_flags, nlist, code_size, sizes);
    } else {
        return InvertedListsIOHook::lookup(h)->read(f, io_flags);
    }
}

static void read_InvertedLists(IndexIVF* ivf, IOReader* f, int io_flags) {
    InvertedLists* ils = read_InvertedLists(f, io_flags);
    if (ils) {
        FAISS_THROW_IF_NOT(ils->nlist == ivf->nlist);
        FAISS_THROW_IF_NOT(
                ils->code_size == InvertedLists::INVALID_CODE_SIZE ||
                ils->code_size == ivf->code_size);
    }
    ivf->invlists = ils;
    ivf->own_invlists = true;
}

static void read_ProductQuantizer(ProductQuantizer* pq, IOReader* f) {
    READ1(pq->d);
    READ1(pq->M);
    READ1(pq->nbits);
    pq->set_derived_values();
    READVECTOR(pq->centroids);
}

static void read_ResidualQuantizer_old(ResidualQuantizer* rq, IOReader* f) {
    READ1(rq->d);
    READ1(rq->M);
    READVECTOR(rq->nbits);
    READ1(rq->is_trained);
    READ1(rq->train_type);
    READ1(rq->max_beam_size);
    READVECTOR(rq->codebooks);
    READ1(rq->search_type);
    READ1(rq->norm_min);
    READ1(rq->norm_max);
    rq->set_derived_values();
}

static void read_AdditiveQuantizer(AdditiveQuantizer* aq, IOReader* f) {
    READ1(aq->d);
    READ1(aq->M);
    READVECTOR(aq->nbits);
    READ1(aq->is_trained);
    READVECTOR(aq->codebooks);
    READ1(aq->search_type);
    READ1(aq->norm_min);
    READ1(aq->norm_max);
    if (aq->search_type == AdditiveQuantizer::ST_norm_cqint8 ||
        aq->search_type == AdditiveQuantizer::ST_norm_cqint4) {
        READXBVECTOR(aq->qnorm.codes);
    }
    aq->set_derived_values();
}

static void read_ResidualQuantizer(ResidualQuantizer* rq, IOReader* f) {
    read_AdditiveQuantizer(rq, f);
    READ1(rq->train_type);
    READ1(rq->max_beam_size);
    if (!(rq->train_type & ResidualQuantizer::Skip_codebook_tables)) {
        rq->compute_codebook_tables();
    }
}

static void read_LocalSearchQuantizer(LocalSearchQuantizer* lsq, IOReader* f) {
    read_AdditiveQuantizer(lsq, f);
    READ1(lsq->K);
    READ1(lsq->train_iters);
    READ1(lsq->encode_ils_iters);
    READ1(lsq->train_ils_iters);
    READ1(lsq->icm_iters);
    READ1(lsq->p);
    READ1(lsq->lambd);
    READ1(lsq->chunk_size);
    READ1(lsq->random_seed);
    READ1(lsq->nperts);
    READ1(lsq->update_codebooks_with_double);
}

static void read_ScalarQuantizer(ScalarQuantizer* ivsc, IOReader* f) {
    READ1(ivsc->qtype);
    READ1(ivsc->rangestat);
    READ1(ivsc->rangestat_arg);
    READ1(ivsc->d);
    READ1(ivsc->code_size);
    READVECTOR(ivsc->trained);
    ivsc->set_derived_sizes();
}

static void read_HNSW(HNSW* hnsw, IOReader* f) {
    READVECTOR(hnsw->assign_probas);
    READVECTOR(hnsw->cum_nneighbor_per_level);
    READVECTOR(hnsw->levels);
    READVECTOR(hnsw->offsets);
    READVECTOR(hnsw->neighbors);

    READ1(hnsw->entry_point);
    READ1(hnsw->max_level);
    READ1(hnsw->efConstruction);
    READ1(hnsw->efSearch);
    READ1(hnsw->upper_beam);
}

static void read_NSG(NSG* nsg, IOReader* f) {
    READ1(nsg->ntotal);
    READ1(nsg->R);
    READ1(nsg->L);
    READ1(nsg->C);
    READ1(nsg->search_L);
    READ1(nsg->enterpoint);
    READ1(nsg->is_built);

    if (!nsg->is_built) {
        return;
    }

    constexpr int EMPTY_ID = -1;
    int N = nsg->ntotal;
    int R = nsg->R;
    auto& graph = nsg->final_graph;
    graph = std::make_shared<nsg::Graph<int>>(N, R);
    std::fill_n(graph->data, N * R, EMPTY_ID);

    int size = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < R + 1; j++) {
            int id;
            READ1(id);
            if (id != EMPTY_ID) {
                graph->at(i, j) = id;
                size += 1;
            } else {
                break;
            }
        }
    }
}

ProductQuantizer* read_ProductQuantizer(const char* fname) {
    FileIOReader reader(fname);
    return read_ProductQuantizer(&reader);
}

ProductQuantizer* read_ProductQuantizer(IOReader* reader) {
    ProductQuantizer* pq = new ProductQuantizer();
    ScopeDeleter1<ProductQuantizer> del(pq);

    read_ProductQuantizer(pq, reader);
    del.release();
    return pq;
}

static void read_direct_map(DirectMap* dm, IOReader* f) {
    char maintain_direct_map;
    READ1(maintain_direct_map);
    dm->type = (DirectMap::Type)maintain_direct_map;
    READVECTOR(dm->array);
    if (dm->type == DirectMap::Hashtable) {
        using idx_t = Index::idx_t;
        std::vector<std::pair<idx_t, idx_t>> v;
        READVECTOR(v);
        std::unordered_map<idx_t, idx_t>& map = dm->hashtable;
        map.reserve(v.size());
        for (auto it : v) {
            map[it.first] = it.second;
        }
    }
}

static void read_ivf_header(
        IndexIVF* ivf,
        IOReader* f,
        std::vector<std::vector<Index::idx_t>>* ids = nullptr) {
    read_index_header(ivf, f);
    READ1(ivf->nlist);
    READ1(ivf->nprobe);
    ivf->quantizer = read_index(f);
    ivf->own_fields = true;
    if (ids) { // used in legacy "Iv" formats
        ids->resize(ivf->nlist);
        for (size_t i = 0; i < ivf->nlist; i++)
            READVECTOR((*ids)[i]);
    }
    read_direct_map(&ivf->direct_map, f);
}

// used for legacy formats
static ArrayInvertedLists* set_array_invlist(
        IndexIVF* ivf,
        std::vector<std::vector<Index::idx_t>>& ids) {
    ArrayInvertedLists* ail =
            new ArrayInvertedLists(ivf->nlist, ivf->code_size);
    std::swap(ail->ids, ids);
    ivf->invlists = ail;
    ivf->own_invlists = true;
    return ail;
}

static IndexIVFPQ* read_ivfpq(IOReader* f, uint32_t h, int io_flags) {
    bool legacy = h == fourcc("IvQR") || h == fourcc("IvPQ");

    IndexIVFPQR* ivfpqr = h == fourcc("IvQR") || h == fourcc("IwQR")
            ? new IndexIVFPQR()
            : nullptr;
    IndexIVFPQ* ivpq = ivfpqr ? ivfpqr : new IndexIVFPQ();

    std::vector<std::vector<Index::idx_t>> ids;
    read_ivf_header(ivpq, f, legacy ? &ids : nullptr);
    READ1(ivpq->by_residual);
    READ1(ivpq->code_size);
    read_ProductQuantizer(&ivpq->pq, f);

    if (legacy) {
        ArrayInvertedLists* ail = set_array_invlist(ivpq, ids);
        for (size_t i = 0; i < ail->nlist; i++)
            READVECTOR(ail->codes[i]);
    } else {
        read_InvertedLists(ivpq, f, io_flags);
    }

    if (ivpq->is_trained) {
        // precomputed table not stored. It is cheaper to recompute it
        ivpq->use_precomputed_table = 0;
        if (ivpq->by_residual)
            ivpq->precompute_table();
        if (ivfpqr) {
            read_ProductQuantizer(&ivfpqr->refine_pq, f);
            READVECTOR(ivfpqr->refine_codes);
            READ1(ivfpqr->k_factor);
        }
    }
    return ivpq;
}

int read_old_fmt_hack = 0;

Index* read_index(IOReader* f, int io_flags) {
    Index* idx = nullptr;
    uint32_t h;
    READ1(h);
    if (h == fourcc("IxFI") || h == fourcc("IxF2") || h == fourcc("IxFl")) {
        IndexFlat* idxf;
        if (h == fourcc("IxFI")) {
            idxf = new IndexFlatIP();
        } else if (h == fourcc("IxF2")) {
            idxf = new IndexFlatL2();
        } else {
            idxf = new IndexFlat();
        }
        read_index_header(idxf, f);
        idxf->code_size = idxf->d * sizeof(float);
        READXBVECTOR(idxf->codes);
        FAISS_THROW_IF_NOT(
                idxf->codes.size() == idxf->ntotal * idxf->code_size);
        // leak!
        idx = idxf;
    } else if (h == fourcc("IxHE") || h == fourcc("IxHe")) {
        IndexLSH* idxl = new IndexLSH();
        read_index_header(idxl, f);
        READ1(idxl->nbits);
        READ1(idxl->rotate_data);
        READ1(idxl->train_thresholds);
        READVECTOR(idxl->thresholds);
        int code_size_i;
        READ1(code_size_i);
        idxl->code_size = code_size_i;
        if (h == fourcc("IxHE")) {
            FAISS_THROW_IF_NOT_FMT(
                    idxl->nbits % 64 == 0,
                    "can only read old format IndexLSH with "
                    "nbits multiple of 64 (got %d)",
                    (int)idxl->nbits);
            // leak
            idxl->code_size *= 8;
        }
        {
            RandomRotationMatrix* rrot = dynamic_cast<RandomRotationMatrix*>(
                    read_VectorTransform(f));
            FAISS_THROW_IF_NOT_MSG(rrot, "expected a random rotation");
            idxl->rrot = *rrot;
            delete rrot;
        }
        READVECTOR(idxl->codes);
        FAISS_THROW_IF_NOT(
                idxl->rrot.d_in == idxl->d && idxl->rrot.d_out == idxl->nbits);
        FAISS_THROW_IF_NOT(
                idxl->codes.size() == idxl->ntotal * idxl->code_size);
        idx = idxl;
    } else if (
            h == fourcc("IxPQ") || h == fourcc("IxPo") || h == fourcc("IxPq")) {
        // IxPQ and IxPo were merged into the same IndexPQ object
        IndexPQ* idxp = new IndexPQ();
        read_index_header(idxp, f);
        read_ProductQuantizer(&idxp->pq, f);
        idxp->code_size = idxp->pq.code_size;
        READVECTOR(idxp->codes);
        if (h == fourcc("IxPo") || h == fourcc("IxPq")) {
            READ1(idxp->search_type);
            READ1(idxp->encode_signs);
            READ1(idxp->polysemous_ht);
        }
        // Old versoins of PQ all had metric_type set to INNER_PRODUCT
        // when they were in fact using L2. Therefore, we force metric type
        // to L2 when the old format is detected
        if (h == fourcc("IxPQ") || h == fourcc("IxPo")) {
            idxp->metric_type = METRIC_L2;
        }
        idx = idxp;
    } else if (h == fourcc("IxRQ") || h == fourcc("IxRq")) {
        IndexResidualQuantizer* idxr = new IndexResidualQuantizer();
        read_index_header(idxr, f);
        if (h == fourcc("IxRQ")) {
            read_ResidualQuantizer_old(&idxr->rq, f);
        } else {
            read_ResidualQuantizer(&idxr->rq, f);
        }
        READ1(idxr->code_size);
        READVECTOR(idxr->codes);
        idx = idxr;
    } else if (h == fourcc("IxLS")) {
        auto idxr = new IndexLocalSearchQuantizer();
        read_index_header(idxr, f);
        read_LocalSearchQuantizer(&idxr->lsq, f);
        READ1(idxr->code_size);
        READVECTOR(idxr->codes);
        idx = idxr;
    } else if (h == fourcc("ImRQ")) {
        ResidualCoarseQuantizer* idxr = new ResidualCoarseQuantizer();
        read_index_header(idxr, f);
        read_ResidualQuantizer(&idxr->rq, f);
        READ1(idxr->beam_factor);
        idxr->set_beam_factor(idxr->beam_factor);
        idx = idxr;
    } else if (h == fourcc("IvFl") || h == fourcc("IvFL")) { // legacy
        IndexIVFFlat* ivfl = new IndexIVFFlat();
        std::vector<std::vector<Index::idx_t>> ids;
        read_ivf_header(ivfl, f, &ids);
        ivfl->code_size = ivfl->d * sizeof(float);
        ArrayInvertedLists* ail = set_array_invlist(ivfl, ids);

        if (h == fourcc("IvFL")) {
            for (size_t i = 0; i < ivfl->nlist; i++) {
                READVECTOR(ail->codes[i]);
            }
        } else { // old format
            for (size_t i = 0; i < ivfl->nlist; i++) {
                std::vector<float> vec;
                READVECTOR(vec);
                ail->codes[i].resize(vec.size() * sizeof(float));
                memcpy(ail->codes[i].data(), vec.data(), ail->codes[i].size());
            }
        }
        idx = ivfl;
    } else if (h == fourcc("IwFd")) {
        IndexIVFFlatDedup* ivfl = new IndexIVFFlatDedup();
        read_ivf_header(ivfl, f);
        ivfl->code_size = ivfl->d * sizeof(float);
        {
            std::vector<Index::idx_t> tab;
            READVECTOR(tab);
            for (long i = 0; i < tab.size(); i += 2) {
                std::pair<Index::idx_t, Index::idx_t> pair(tab[i], tab[i + 1]);
                ivfl->instances.insert(pair);
            }
        }
        read_InvertedLists(ivfl, f, io_flags);
        idx = ivfl;
    } else if (h == fourcc("IwFl")) {
        IndexIVFFlat* ivfl = new IndexIVFFlat();
        read_ivf_header(ivfl, f);
        ivfl->code_size = ivfl->d * sizeof(float);
        read_InvertedLists(ivfl, f, io_flags);
        idx = ivfl;
    } else if (h == fourcc("IxSQ")) {
        IndexScalarQuantizer* idxs = new IndexScalarQuantizer();
        read_index_header(idxs, f);
        read_ScalarQuantizer(&idxs->sq, f);
        READVECTOR(idxs->codes);
        idxs->code_size = idxs->sq.code_size;
        idx = idxs;
    } else if (h == fourcc("IxLa")) {
        int d, nsq, scale_nbit, r2;
        READ1(d);
        READ1(nsq);
        READ1(scale_nbit);
        READ1(r2);
        IndexLattice* idxl = new IndexLattice(d, nsq, scale_nbit, r2);
        read_index_header(idxl, f);
        READVECTOR(idxl->trained);
        idx = idxl;
    } else if (h == fourcc("IvSQ")) { // legacy
        IndexIVFScalarQuantizer* ivsc = new IndexIVFScalarQuantizer();
        std::vector<std::vector<Index::idx_t>> ids;
        read_ivf_header(ivsc, f, &ids);
        read_ScalarQuantizer(&ivsc->sq, f);
        READ1(ivsc->code_size);
        ArrayInvertedLists* ail = set_array_invlist(ivsc, ids);
        for (int i = 0; i < ivsc->nlist; i++)
            READVECTOR(ail->codes[i]);
        idx = ivsc;
    } else if (h == fourcc("IwSQ") || h == fourcc("IwSq")) {
        IndexIVFScalarQuantizer* ivsc = new IndexIVFScalarQuantizer();
        read_ivf_header(ivsc, f);
        read_ScalarQuantizer(&ivsc->sq, f);
        READ1(ivsc->code_size);
        if (h == fourcc("IwSQ")) {
            ivsc->by_residual = true;
        } else {
            READ1(ivsc->by_residual);
        }
        read_InvertedLists(ivsc, f, io_flags);
        idx = ivsc;
    } else if (h == fourcc("IwLS") || h == fourcc("IwRQ")) {
        bool is_LSQ = h == fourcc("IwLS");
        IndexIVFAdditiveQuantizer* iva;
        if (is_LSQ) {
            iva = new IndexIVFLocalSearchQuantizer();
        } else {
            iva = new IndexIVFResidualQuantizer();
        }
        read_ivf_header(iva, f);
        READ1(iva->code_size);
        if (is_LSQ) {
            read_LocalSearchQuantizer((LocalSearchQuantizer*)iva->aq, f);
        } else {
            read_ResidualQuantizer((ResidualQuantizer*)iva->aq, f);
        }
        READ1(iva->by_residual);
        READ1(iva->use_precomputed_table);
        read_InvertedLists(iva, f, io_flags);
        idx = iva;
    } else if (h == fourcc("IwSh")) {
        IndexIVFSpectralHash* ivsp = new IndexIVFSpectralHash();
        read_ivf_header(ivsp, f);
        ivsp->vt = read_VectorTransform(f);
        ivsp->own_fields = true;
        READ1(ivsp->nbit);
        // not stored by write_ivf_header
        ivsp->code_size = (ivsp->nbit + 7) / 8;
        READ1(ivsp->period);
        READ1(ivsp->threshold_type);
        READVECTOR(ivsp->trained);
        read_InvertedLists(ivsp, f, io_flags);
        idx = ivsp;
    } else if (
            h == fourcc("IvPQ") || h == fourcc("IvQR") || h == fourcc("IwPQ") ||
            h == fourcc("IwQR")) {
        idx = read_ivfpq(f, h, io_flags);

    } else if (h == fourcc("IxPT")) {
        IndexPreTransform* ixpt = new IndexPreTransform();
        ixpt->own_fields = true;
        read_index_header(ixpt, f);
        int nt;
        if (read_old_fmt_hack == 2) {
            nt = 1;
        } else {
            READ1(nt);
        }
        for (int i = 0; i < nt; i++) {
            ixpt->chain.push_back(read_VectorTransform(f));
        }
        ixpt->index = read_index(f, io_flags);
        idx = ixpt;
    } else if (h == fourcc("Imiq")) {
        MultiIndexQuantizer* imiq = new MultiIndexQuantizer();
        read_index_header(imiq, f);
        read_ProductQuantizer(&imiq->pq, f);
        idx = imiq;
    } else if (h == fourcc("IxRF")) {
        IndexRefine* idxrf = new IndexRefine();
        read_index_header(idxrf, f);
        idxrf->base_index = read_index(f, io_flags);
        idxrf->refine_index = read_index(f, io_flags);
        READ1(idxrf->k_factor);
        if (dynamic_cast<IndexFlat*>(idxrf->refine_index)) {
            // then make a RefineFlat with it
            IndexRefine* idxrf_old = idxrf;
            idxrf = new IndexRefineFlat();
            *idxrf = *idxrf_old;
            delete idxrf_old;
        }
        idxrf->own_fields = true;
        idxrf->own_refine_index = true;
        idx = idxrf;
    } else if (h == fourcc("IxMp") || h == fourcc("IxM2")) {
        bool is_map2 = h == fourcc("IxM2");
        IndexIDMap* idxmap = is_map2 ? new IndexIDMap2() : new IndexIDMap();
        read_index_header(idxmap, f);
        idxmap->index = read_index(f, io_flags);
        idxmap->own_fields = true;
        READVECTOR(idxmap->id_map);
        if (is_map2) {
            static_cast<IndexIDMap2*>(idxmap)->construct_rev_map();
        }
        idx = idxmap;
    } else if (h == fourcc("Ix2L")) {
        Index2Layer* idxp = new Index2Layer();
        read_index_header(idxp, f);
        idxp->q1.quantizer = read_index(f, io_flags);
        READ1(idxp->q1.nlist);
        READ1(idxp->q1.quantizer_trains_alone);
        read_ProductQuantizer(&idxp->pq, f);
        READ1(idxp->code_size_1);
        READ1(idxp->code_size_2);
        READ1(idxp->code_size);
        READVECTOR(idxp->codes);
        idx = idxp;
    } else if (
            h == fourcc("IHNf") || h == fourcc("IHNp") || h == fourcc("IHNs") ||
            h == fourcc("IHN2")) {
        IndexHNSW* idxhnsw = nullptr;
        if (h == fourcc("IHNf"))
            idxhnsw = new IndexHNSWFlat();
        if (h == fourcc("IHNp"))
            idxhnsw = new IndexHNSWPQ();
        if (h == fourcc("IHNs"))
            idxhnsw = new IndexHNSWSQ();
        if (h == fourcc("IHN2"))
            idxhnsw = new IndexHNSW2Level();
        read_index_header(idxhnsw, f);
        read_HNSW(&idxhnsw->hnsw, f);
        idxhnsw->storage = read_index(f, io_flags);
        idxhnsw->own_fields = true;
        if (h == fourcc("IHNp")) {
            dynamic_cast<IndexPQ*>(idxhnsw->storage)->pq.compute_sdc_table();
        }
        idx = idxhnsw;
    } else if (h == fourcc("INSf")) {
        IndexNSG* idxnsg = new IndexNSGFlat();
        read_index_header(idxnsg, f);
        READ1(idxnsg->GK);
        READ1(idxnsg->build_type);
        READ1(idxnsg->nndescent_S);
        READ1(idxnsg->nndescent_R);
        READ1(idxnsg->nndescent_L);
        READ1(idxnsg->nndescent_iter);
        read_NSG(&idxnsg->nsg, f);
        idxnsg->storage = read_index(f, io_flags);
        idxnsg->own_fields = true;
        idx = idxnsg;
    } else if (h == fourcc("IPfs")) {
        IndexPQFastScan* idxpqfs = new IndexPQFastScan();
        read_index_header(idxpqfs, f);
        read_ProductQuantizer(&idxpqfs->pq, f);
        READ1(idxpqfs->implem);
        READ1(idxpqfs->bbs);
        READ1(idxpqfs->qbs);
        READ1(idxpqfs->ntotal2);
        READ1(idxpqfs->M2);
        READVECTOR(idxpqfs->codes);
        idx = idxpqfs;

    } else if (h == fourcc("IwPf")) {
        IndexIVFPQFastScan* ivpq = new IndexIVFPQFastScan();
        read_ivf_header(ivpq, f);
        READ1(ivpq->by_residual);
        READ1(ivpq->code_size);
        READ1(ivpq->bbs);
        READ1(ivpq->M2);
        READ1(ivpq->implem);
        READ1(ivpq->qbs2);
        read_ProductQuantizer(&ivpq->pq, f);
        read_InvertedLists(ivpq, f, io_flags);
        ivpq->precompute_table();
        idx = ivpq;
    } else {
        FAISS_THROW_FMT(
                "Index type 0x%08x (\"%s\") not recognized",
                h,
                fourcc_inv_printable(h).c_str());
        idx = nullptr;
    }
    return idx;
}

Index* read_index(FILE* f, int io_flags) {
    FileIOReader reader(f);
    return read_index(&reader, io_flags);
}

Index* read_index(const char* fname, int io_flags) {
    FileIOReader reader(fname);
    Index* idx = read_index(&reader, io_flags);
    return idx;
}

VectorTransform* read_VectorTransform(const char* fname) {
    FileIOReader reader(fname);
    VectorTransform* vt = read_VectorTransform(&reader);
    return vt;
}

/*************************************************************
 * Read binary indexes
 **************************************************************/

static void read_InvertedLists(IndexBinaryIVF* ivf, IOReader* f, int io_flags) {
    InvertedLists* ils = read_InvertedLists(f, io_flags);
    FAISS_THROW_IF_NOT(
            !ils ||
            (ils->nlist == ivf->nlist && ils->code_size == ivf->code_size));
    ivf->invlists = ils;
    ivf->own_invlists = true;
}

static void read_index_binary_header(IndexBinary* idx, IOReader* f) {
    READ1(idx->d);
    READ1(idx->code_size);
    READ1(idx->ntotal);
    READ1(idx->is_trained);
    READ1(idx->metric_type);
    idx->verbose = false;
}

static void read_binary_ivf_header(
        IndexBinaryIVF* ivf,
        IOReader* f,
        std::vector<std::vector<Index::idx_t>>* ids = nullptr) {
    read_index_binary_header(ivf, f);
    READ1(ivf->nlist);
    READ1(ivf->nprobe);
    ivf->quantizer = read_index_binary(f);
    ivf->own_fields = true;
    if (ids) { // used in legacy "Iv" formats
        ids->resize(ivf->nlist);
        for (size_t i = 0; i < ivf->nlist; i++)
            READVECTOR((*ids)[i]);
    }
    read_direct_map(&ivf->direct_map, f);
}

static void read_binary_hash_invlists(
        IndexBinaryHash::InvertedListMap& invlists,
        int b,
        IOReader* f) {
    size_t sz;
    READ1(sz);
    int il_nbit = 0;
    READ1(il_nbit);
    // buffer for bitstrings
    std::vector<uint8_t> buf((b + il_nbit) * sz);
    READVECTOR(buf);
    BitstringReader rd(buf.data(), buf.size());
    invlists.reserve(sz);
    for (size_t i = 0; i < sz; i++) {
        uint64_t hash = rd.read(b);
        uint64_t ilsz = rd.read(il_nbit);
        auto& il = invlists[hash];
        READVECTOR(il.ids);
        FAISS_THROW_IF_NOT(il.ids.size() == ilsz);
        READVECTOR(il.vecs);
    }
}

static void read_binary_multi_hash_map(
        IndexBinaryMultiHash::Map& map,
        int b,
        size_t ntotal,
        IOReader* f) {
    int id_bits;
    size_t sz;
    READ1(id_bits);
    READ1(sz);
    std::vector<uint8_t> buf;
    READVECTOR(buf);
    size_t nbit = (b + id_bits) * sz + ntotal * id_bits;
    FAISS_THROW_IF_NOT(buf.size() == (nbit + 7) / 8);
    BitstringReader rd(buf.data(), buf.size());
    map.reserve(sz);
    for (size_t i = 0; i < sz; i++) {
        uint64_t hash = rd.read(b);
        uint64_t ilsz = rd.read(id_bits);
        auto& il = map[hash];
        for (size_t j = 0; j < ilsz; j++) {
            il.push_back(rd.read(id_bits));
        }
    }
}

IndexBinary* read_index_binary(IOReader* f, int io_flags) {
    IndexBinary* idx = nullptr;
    uint32_t h;
    READ1(h);
    if (h == fourcc("IBxF")) {
        IndexBinaryFlat* idxf = new IndexBinaryFlat();
        read_index_binary_header(idxf, f);
        READVECTOR(idxf->xb);
        FAISS_THROW_IF_NOT(idxf->xb.size() == idxf->ntotal * idxf->code_size);
        // leak!
        idx = idxf;
    } else if (h == fourcc("IBwF")) {
        IndexBinaryIVF* ivf = new IndexBinaryIVF();
        read_binary_ivf_header(ivf, f);
        read_InvertedLists(ivf, f, io_flags);
        idx = ivf;
    } else if (h == fourcc("IBFf")) {
        IndexBinaryFromFloat* idxff = new IndexBinaryFromFloat();
        read_index_binary_header(idxff, f);
        idxff->own_fields = true;
        idxff->index = read_index(f, io_flags);
        idx = idxff;
    } else if (h == fourcc("IBHf")) {
        IndexBinaryHNSW* idxhnsw = new IndexBinaryHNSW();
        read_index_binary_header(idxhnsw, f);
        read_HNSW(&idxhnsw->hnsw, f);
        idxhnsw->storage = read_index_binary(f, io_flags);
        idxhnsw->own_fields = true;
        idx = idxhnsw;
    } else if (h == fourcc("IBMp") || h == fourcc("IBM2")) {
        bool is_map2 = h == fourcc("IBM2");
        IndexBinaryIDMap* idxmap =
                is_map2 ? new IndexBinaryIDMap2() : new IndexBinaryIDMap();
        read_index_binary_header(idxmap, f);
        idxmap->index = read_index_binary(f, io_flags);
        idxmap->own_fields = true;
        READVECTOR(idxmap->id_map);
        if (is_map2) {
            static_cast<IndexBinaryIDMap2*>(idxmap)->construct_rev_map();
        }
        idx = idxmap;
    } else if (h == fourcc("IBHh")) {
        IndexBinaryHash* idxh = new IndexBinaryHash();
        read_index_binary_header(idxh, f);
        READ1(idxh->b);
        READ1(idxh->nflip);
        read_binary_hash_invlists(idxh->invlists, idxh->b, f);
        idx = idxh;
    } else if (h == fourcc("IBHm")) {
        IndexBinaryMultiHash* idxmh = new IndexBinaryMultiHash();
        read_index_binary_header(idxmh, f);
        idxmh->storage = dynamic_cast<IndexBinaryFlat*>(read_index_binary(f));
        FAISS_THROW_IF_NOT(
                idxmh->storage && idxmh->storage->ntotal == idxmh->ntotal);
        idxmh->own_fields = true;
        READ1(idxmh->b);
        READ1(idxmh->nhash);
        READ1(idxmh->nflip);
        idxmh->maps.resize(idxmh->nhash);
        for (int i = 0; i < idxmh->nhash; i++) {
            read_binary_multi_hash_map(
                    idxmh->maps[i], idxmh->b, idxmh->ntotal, f);
        }
        idx = idxmh;
    } else {
        FAISS_THROW_FMT(
                "Index type %08x (\"%s\") not recognized",
                h,
                fourcc_inv_printable(h).c_str());
        idx = nullptr;
    }
    return idx;
}

IndexBinary* read_index_binary(FILE* f, int io_flags) {
    FileIOReader reader(f);
    return read_index_binary(&reader, io_flags);
}

IndexBinary* read_index_binary(const char* fname, int io_flags) {
    FileIOReader reader(fname);
    IndexBinary* idx = read_index_binary(&reader, io_flags);
    return idx;
}

} // namespace faiss
