/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/index_io.h>

#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

#include <cstdio>
#include <cstdlib>

#include <faiss/invlists/InvertedListsIOHook.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexAdditiveQuantizerFastScan.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/IndexIVFAdditiveQuantizerFastScan.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFIndependentQuantizer.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexIVFPQR.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexIVFSpectralHash.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexLattice.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexRowwiseMinMax.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <faiss/VectorTransform.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryFromFloat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/IndexBinaryIVF.h>

/*************************************************************
 * The I/O format is the content of the class. For objects that are
 * inherited, like Index, a 4-character-code (fourcc) indicates which
 * child class this is an instance of.
 *
 * In this case, the fields of the parent class are written first,
 * then the ones for the child classes. Note that this requires
 * classes to be serialized to have a constructor without parameters,
 * so that the fields can be filled in later. The default constructor
 * should set reasonable defaults for all fields.
 *
 * The fourccs are assigned arbitrarily. When the class changed (added
 * or deprecated fields), the fourcc can be replaced. New code should
 * be able to read the old fourcc and fill in new classes.
 *
 * TODO: in this file, the read functions that encouter errors may
 * leak memory.
 **************************************************************/

namespace faiss {

/*************************************************************
 * Write
 **************************************************************/
static void write_index_header(const Index* idx, IOWriter* f) {
    WRITE1(idx->d);
    WRITE1(idx->ntotal);
    idx_t dummy = 1 << 20;
    WRITE1(dummy);
    WRITE1(dummy);
    WRITE1(idx->is_trained);
    WRITE1(idx->metric_type);
    if (idx->metric_type > 1) {
        WRITE1(idx->metric_arg);
    }
}

void write_VectorTransform(const VectorTransform* vt, IOWriter* f) {
    if (const LinearTransform* lt = dynamic_cast<const LinearTransform*>(vt)) {
        if (dynamic_cast<const RandomRotationMatrix*>(lt)) {
            uint32_t h = fourcc("rrot");
            WRITE1(h);
        } else if (const PCAMatrix* pca = dynamic_cast<const PCAMatrix*>(lt)) {
            uint32_t h = fourcc("Pcam");
            WRITE1(h);
            WRITE1(pca->eigen_power);
            WRITE1(pca->epsilon);
            WRITE1(pca->random_rotation);
            WRITE1(pca->balanced_bins);
            WRITEVECTOR(pca->mean);
            WRITEVECTOR(pca->eigenvalues);
            WRITEVECTOR(pca->PCAMat);
        } else if (const ITQMatrix* itqm = dynamic_cast<const ITQMatrix*>(lt)) {
            uint32_t h = fourcc("Viqm");
            WRITE1(h);
            WRITE1(itqm->max_iter);
            WRITE1(itqm->seed);
        } else {
            // generic LinearTransform (includes OPQ)
            uint32_t h = fourcc("LTra");
            WRITE1(h);
        }
        WRITE1(lt->have_bias);
        WRITEVECTOR(lt->A);
        WRITEVECTOR(lt->b);
    } else if (
            const RemapDimensionsTransform* rdt =
                    dynamic_cast<const RemapDimensionsTransform*>(vt)) {
        uint32_t h = fourcc("RmDT");
        WRITE1(h);
        WRITEVECTOR(rdt->map);
    } else if (
            const NormalizationTransform* nt =
                    dynamic_cast<const NormalizationTransform*>(vt)) {
        uint32_t h = fourcc("VNrm");
        WRITE1(h);
        WRITE1(nt->norm);
    } else if (
            const CenteringTransform* ct =
                    dynamic_cast<const CenteringTransform*>(vt)) {
        uint32_t h = fourcc("VCnt");
        WRITE1(h);
        WRITEVECTOR(ct->mean);
    } else if (
            const ITQTransform* itqt = dynamic_cast<const ITQTransform*>(vt)) {
        uint32_t h = fourcc("Viqt");
        WRITE1(h);
        WRITEVECTOR(itqt->mean);
        WRITE1(itqt->do_pca);
        write_VectorTransform(&itqt->itq, f);
        write_VectorTransform(&itqt->pca_then_itq, f);
    } else {
        FAISS_THROW_MSG("cannot serialize this");
    }
    // common fields
    WRITE1(vt->d_in);
    WRITE1(vt->d_out);
    WRITE1(vt->is_trained);
}

void write_ProductQuantizer(const ProductQuantizer* pq, IOWriter* f) {
    WRITE1(pq->d);
    WRITE1(pq->M);
    WRITE1(pq->nbits);
    WRITEVECTOR(pq->centroids);
}

static void write_AdditiveQuantizer(const AdditiveQuantizer* aq, IOWriter* f) {
    WRITE1(aq->d);
    WRITE1(aq->M);
    WRITEVECTOR(aq->nbits);
    WRITE1(aq->is_trained);
    WRITEVECTOR(aq->codebooks);
    WRITE1(aq->search_type);
    WRITE1(aq->norm_min);
    WRITE1(aq->norm_max);
    if (aq->search_type == AdditiveQuantizer::ST_norm_cqint8 ||
        aq->search_type == AdditiveQuantizer::ST_norm_cqint4 ||
        aq->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
        aq->search_type == AdditiveQuantizer::ST_norm_rq2x4) {
        WRITEXBVECTOR(aq->qnorm.codes);
    }

    if (aq->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
        aq->search_type == AdditiveQuantizer::ST_norm_rq2x4) {
        WRITEVECTOR(aq->norm_tabs);
    }
}

static void write_ResidualQuantizer(const ResidualQuantizer* rq, IOWriter* f) {
    write_AdditiveQuantizer(rq, f);
    WRITE1(rq->train_type);
    WRITE1(rq->max_beam_size);
}

static void write_LocalSearchQuantizer(
        const LocalSearchQuantizer* lsq,
        IOWriter* f) {
    write_AdditiveQuantizer(lsq, f);
    WRITE1(lsq->K);
    WRITE1(lsq->train_iters);
    WRITE1(lsq->encode_ils_iters);
    WRITE1(lsq->train_ils_iters);
    WRITE1(lsq->icm_iters);
    WRITE1(lsq->p);
    WRITE1(lsq->lambd);
    WRITE1(lsq->chunk_size);
    WRITE1(lsq->random_seed);
    WRITE1(lsq->nperts);
    WRITE1(lsq->update_codebooks_with_double);
}

static void write_ProductAdditiveQuantizer(
        const ProductAdditiveQuantizer* paq,
        IOWriter* f) {
    write_AdditiveQuantizer(paq, f);
    WRITE1(paq->nsplits);
}

static void write_ProductResidualQuantizer(
        const ProductResidualQuantizer* prq,
        IOWriter* f) {
    write_ProductAdditiveQuantizer(prq, f);
    for (const auto aq : prq->quantizers) {
        auto rq = dynamic_cast<const ResidualQuantizer*>(aq);
        write_ResidualQuantizer(rq, f);
    }
}

static void write_ProductLocalSearchQuantizer(
        const ProductLocalSearchQuantizer* plsq,
        IOWriter* f) {
    write_ProductAdditiveQuantizer(plsq, f);
    for (const auto aq : plsq->quantizers) {
        auto lsq = dynamic_cast<const LocalSearchQuantizer*>(aq);
        write_LocalSearchQuantizer(lsq, f);
    }
}

static void write_ScalarQuantizer(const ScalarQuantizer* ivsc, IOWriter* f) {
    WRITE1(ivsc->qtype);
    WRITE1(ivsc->rangestat);
    WRITE1(ivsc->rangestat_arg);
    WRITE1(ivsc->d);
    WRITE1(ivsc->code_size);
    WRITEVECTOR(ivsc->trained);
}

void write_InvertedLists(const InvertedLists* ils, IOWriter* f) {
    if (ils == nullptr) {
        uint32_t h = fourcc("il00");
        WRITE1(h);
    } else if (
            const auto& ails = dynamic_cast<const ArrayInvertedLists*>(ils)) {
        uint32_t h = fourcc("ilar");
        WRITE1(h);
        WRITE1(ails->nlist);
        WRITE1(ails->code_size);
        // here we store either as a full or a sparse data buffer
        size_t n_non0 = 0;
        for (size_t i = 0; i < ails->nlist; i++) {
            if (ails->ids[i].size() > 0) {
                n_non0++;
            }
        }
        if (n_non0 > ails->nlist / 2) {
            uint32_t list_type = fourcc("full");
            WRITE1(list_type);
            std::vector<size_t> sizes;
            for (size_t i = 0; i < ails->nlist; i++) {
                sizes.push_back(ails->ids[i].size());
            }
            WRITEVECTOR(sizes);
        } else {
            int list_type = fourcc("sprs"); // sparse
            WRITE1(list_type);
            std::vector<size_t> sizes;
            for (size_t i = 0; i < ails->nlist; i++) {
                size_t n = ails->ids[i].size();
                if (n > 0) {
                    sizes.push_back(i);
                    sizes.push_back(n);
                }
            }
            WRITEVECTOR(sizes);
        }
        // make a single contiguous data buffer (useful for mmapping)
        for (size_t i = 0; i < ails->nlist; i++) {
            size_t n = ails->ids[i].size();
            if (n > 0) {
                WRITEANDCHECK(ails->codes[i].data(), n * ails->code_size);
                WRITEANDCHECK(ails->ids[i].data(), n);
            }
        }

    } else {
        InvertedListsIOHook::lookup_classname(typeid(*ils).name())
                ->write(ils, f);
    }
}

void write_ProductQuantizer(const ProductQuantizer* pq, const char* fname) {
    FileIOWriter writer(fname);
    write_ProductQuantizer(pq, &writer);
}

static void write_HNSW(const HNSW* hnsw, IOWriter* f) {
    WRITEVECTOR(hnsw->assign_probas);
    WRITEVECTOR(hnsw->cum_nneighbor_per_level);
    WRITEVECTOR(hnsw->levels);
    WRITEVECTOR(hnsw->offsets);
    WRITEVECTOR(hnsw->neighbors);

    WRITE1(hnsw->entry_point);
    WRITE1(hnsw->max_level);
    WRITE1(hnsw->efConstruction);
    WRITE1(hnsw->efSearch);

    // // deprecated field
    // WRITE1(hnsw->upper_beam);
    constexpr int tmp_upper_beam = 1;
    WRITE1(tmp_upper_beam);
}

static void write_NSG(const NSG* nsg, IOWriter* f) {
    WRITE1(nsg->ntotal);
    WRITE1(nsg->R);
    WRITE1(nsg->L);
    WRITE1(nsg->C);
    WRITE1(nsg->search_L);
    WRITE1(nsg->enterpoint);
    WRITE1(nsg->is_built);

    if (!nsg->is_built) {
        return;
    }

    constexpr int EMPTY_ID = -1;
    auto& graph = nsg->final_graph;
    int K = graph->K;
    int N = graph->N;
    FAISS_THROW_IF_NOT(N == nsg->ntotal);
    FAISS_THROW_IF_NOT(K == nsg->R);
    FAISS_THROW_IF_NOT(true == graph->own_fields);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            int id = graph->at(i, j);
            if (id != EMPTY_ID) {
                WRITE1(id);
            } else {
                break;
            }
        }
        WRITE1(EMPTY_ID);
    }
}

static void write_NNDescent(const NNDescent* nnd, IOWriter* f) {
    WRITE1(nnd->ntotal);
    WRITE1(nnd->d);
    WRITE1(nnd->K);
    WRITE1(nnd->S);
    WRITE1(nnd->R);
    WRITE1(nnd->L);
    WRITE1(nnd->iter);
    WRITE1(nnd->search_L);
    WRITE1(nnd->random_seed);
    WRITE1(nnd->has_built);

    WRITEVECTOR(nnd->final_graph);
}

static void write_RaBitQuantizer(const RaBitQuantizer* rabitq, IOWriter* f) {
    // don't care about rabitq->centroid
    WRITE1(rabitq->d);
    WRITE1(rabitq->code_size);
    WRITE1(rabitq->metric_type);
}

static void write_direct_map(const DirectMap* dm, IOWriter* f) {
    char maintain_direct_map =
            (char)dm->type; // for backwards compatibility with bool
    WRITE1(maintain_direct_map);
    WRITEVECTOR(dm->array);
    if (dm->type == DirectMap::Hashtable) {
        std::vector<std::pair<idx_t, idx_t>> v;
        const std::unordered_map<idx_t, idx_t>& map = dm->hashtable;
        v.resize(map.size());
        std::copy(map.begin(), map.end(), v.begin());
        WRITEVECTOR(v);
    }
}

static void write_ivf_header(const IndexIVF* ivf, IOWriter* f) {
    write_index_header(ivf, f);
    WRITE1(ivf->nlist);
    WRITE1(ivf->nprobe);
    // subclasses write by_residual (some of them support only one setting of
    // by_residual).
    write_index(ivf->quantizer, f);
    write_direct_map(&ivf->direct_map, f);
}

void write_index(const Index* idx, IOWriter* f, int io_flags) {
    if (idx == nullptr) {
        // eg. for a storage component of HNSW that is set to nullptr
        uint32_t h = fourcc("null");
        WRITE1(h);
    } else if (const IndexFlat* idxf = dynamic_cast<const IndexFlat*>(idx)) {
        uint32_t h =
                fourcc(idxf->metric_type == METRIC_INNER_PRODUCT ? "IxFI"
                               : idxf->metric_type == METRIC_L2  ? "IxF2"
                                                                 : "IxFl");
        WRITE1(h);
        write_index_header(idx, f);
        WRITEXBVECTOR(idxf->codes);
    } else if (const IndexLSH* idxl = dynamic_cast<const IndexLSH*>(idx)) {
        uint32_t h = fourcc("IxHe");
        WRITE1(h);
        write_index_header(idx, f);
        WRITE1(idxl->nbits);
        WRITE1(idxl->rotate_data);
        WRITE1(idxl->train_thresholds);
        WRITEVECTOR(idxl->thresholds);
        int code_size_i = idxl->code_size;
        WRITE1(code_size_i);
        write_VectorTransform(&idxl->rrot, f);
        WRITEVECTOR(idxl->codes);
    } else if (const IndexPQ* idxp = dynamic_cast<const IndexPQ*>(idx)) {
        uint32_t h = fourcc("IxPq");
        WRITE1(h);
        write_index_header(idx, f);
        write_ProductQuantizer(&idxp->pq, f);
        WRITEVECTOR(idxp->codes);
        // search params -- maybe not useful to store?
        WRITE1(idxp->search_type);
        WRITE1(idxp->encode_signs);
        WRITE1(idxp->polysemous_ht);
    } else if (
            const IndexResidualQuantizer* idxr =
                    dynamic_cast<const IndexResidualQuantizer*>(idx)) {
        uint32_t h = fourcc("IxRq");
        WRITE1(h);
        write_index_header(idx, f);
        write_ResidualQuantizer(&idxr->rq, f);
        WRITE1(idxr->code_size);
        WRITEVECTOR(idxr->codes);
    } else if (
            auto* idxr_2 =
                    dynamic_cast<const IndexLocalSearchQuantizer*>(idx)) {
        uint32_t h = fourcc("IxLS");
        WRITE1(h);
        write_index_header(idx, f);
        write_LocalSearchQuantizer(&idxr_2->lsq, f);
        WRITE1(idxr_2->code_size);
        WRITEVECTOR(idxr_2->codes);
    } else if (
            const IndexProductResidualQuantizer* idxpr =
                    dynamic_cast<const IndexProductResidualQuantizer*>(idx)) {
        uint32_t h = fourcc("IxPR");
        WRITE1(h);
        write_index_header(idx, f);
        write_ProductResidualQuantizer(&idxpr->prq, f);
        WRITE1(idxpr->code_size);
        WRITEVECTOR(idxpr->codes);
    } else if (
            const IndexProductLocalSearchQuantizer* idxpl =
                    dynamic_cast<const IndexProductLocalSearchQuantizer*>(
                            idx)) {
        uint32_t h = fourcc("IxPL");
        WRITE1(h);
        write_index_header(idx, f);
        write_ProductLocalSearchQuantizer(&idxpl->plsq, f);
        WRITE1(idxpl->code_size);
        WRITEVECTOR(idxpl->codes);
    } else if (
            auto* idxaqfs =
                    dynamic_cast<const IndexAdditiveQuantizerFastScan*>(idx)) {
        auto idxlsqfs =
                dynamic_cast<const IndexLocalSearchQuantizerFastScan*>(idx);
        auto idxrqfs = dynamic_cast<const IndexResidualQuantizerFastScan*>(idx);
        auto idxplsqfs =
                dynamic_cast<const IndexProductLocalSearchQuantizerFastScan*>(
                        idx);
        auto idxprqfs =
                dynamic_cast<const IndexProductResidualQuantizerFastScan*>(idx);
        FAISS_THROW_IF_NOT(idxlsqfs || idxrqfs || idxplsqfs || idxprqfs);

        if (idxlsqfs) {
            uint32_t h = fourcc("ILfs");
            WRITE1(h);
        } else if (idxrqfs) {
            uint32_t h = fourcc("IRfs");
            WRITE1(h);
        } else if (idxplsqfs) {
            uint32_t h = fourcc("IPLf");
            WRITE1(h);
        } else if (idxprqfs) {
            uint32_t h = fourcc("IPRf");
            WRITE1(h);
        }

        write_index_header(idxaqfs, f);

        if (idxlsqfs) {
            write_LocalSearchQuantizer(&idxlsqfs->lsq, f);
        } else if (idxrqfs) {
            write_ResidualQuantizer(&idxrqfs->rq, f);
        } else if (idxplsqfs) {
            write_ProductLocalSearchQuantizer(&idxplsqfs->plsq, f);
        } else if (idxprqfs) {
            write_ProductResidualQuantizer(&idxprqfs->prq, f);
        }
        WRITE1(idxaqfs->implem);
        WRITE1(idxaqfs->bbs);
        WRITE1(idxaqfs->qbs);

        WRITE1(idxaqfs->M);
        WRITE1(idxaqfs->nbits);
        WRITE1(idxaqfs->ksub);
        WRITE1(idxaqfs->code_size);
        WRITE1(idxaqfs->ntotal2);
        WRITE1(idxaqfs->M2);

        WRITE1(idxaqfs->rescale_norm);
        WRITE1(idxaqfs->norm_scale);
        WRITE1(idxaqfs->max_train_points);

        WRITEVECTOR(idxaqfs->codes);
    } else if (
            auto* ivaqfs =
                    dynamic_cast<const IndexIVFAdditiveQuantizerFastScan*>(
                            idx)) {
        auto ivlsqfs =
                dynamic_cast<const IndexIVFLocalSearchQuantizerFastScan*>(idx);
        auto ivrqfs =
                dynamic_cast<const IndexIVFResidualQuantizerFastScan*>(idx);
        auto ivplsqfs = dynamic_cast<
                const IndexIVFProductLocalSearchQuantizerFastScan*>(idx);
        auto ivprqfs =
                dynamic_cast<const IndexIVFProductResidualQuantizerFastScan*>(
                        idx);
        FAISS_THROW_IF_NOT(ivlsqfs || ivrqfs || ivplsqfs || ivprqfs);

        if (ivlsqfs) {
            uint32_t h = fourcc("IVLf");
            WRITE1(h);
        } else if (ivrqfs) {
            uint32_t h = fourcc("IVRf");
            WRITE1(h);
        } else if (ivplsqfs) {
            uint32_t h = fourcc("NPLf"); // N means IV ...
            WRITE1(h);
        } else {
            uint32_t h = fourcc("NPRf");
            WRITE1(h);
        }

        write_ivf_header(ivaqfs, f);

        if (ivlsqfs) {
            write_LocalSearchQuantizer(&ivlsqfs->lsq, f);
        } else if (ivrqfs) {
            write_ResidualQuantizer(&ivrqfs->rq, f);
        } else if (ivplsqfs) {
            write_ProductLocalSearchQuantizer(&ivplsqfs->plsq, f);
        } else {
            write_ProductResidualQuantizer(&ivprqfs->prq, f);
        }

        WRITE1(ivaqfs->by_residual);
        WRITE1(ivaqfs->implem);
        WRITE1(ivaqfs->bbs);
        WRITE1(ivaqfs->qbs);

        WRITE1(ivaqfs->M);
        WRITE1(ivaqfs->nbits);
        WRITE1(ivaqfs->ksub);
        WRITE1(ivaqfs->code_size);
        WRITE1(ivaqfs->qbs2);
        WRITE1(ivaqfs->M2);

        WRITE1(ivaqfs->rescale_norm);
        WRITE1(ivaqfs->norm_scale);
        WRITE1(ivaqfs->max_train_points);

        write_InvertedLists(ivaqfs->invlists, f);
    } else if (
            const ResidualCoarseQuantizer* idxr_2 =
                    dynamic_cast<const ResidualCoarseQuantizer*>(idx)) {
        uint32_t h = fourcc("ImRQ");
        WRITE1(h);
        write_index_header(idx, f);
        write_ResidualQuantizer(&idxr_2->rq, f);
        WRITE1(idxr_2->beam_factor);
    } else if (
            const Index2Layer* idxp_2 = dynamic_cast<const Index2Layer*>(idx)) {
        uint32_t h = fourcc("Ix2L");
        WRITE1(h);
        write_index_header(idx, f);
        write_index(idxp_2->q1.quantizer, f);
        WRITE1(idxp_2->q1.nlist);
        WRITE1(idxp_2->q1.quantizer_trains_alone);
        write_ProductQuantizer(&idxp_2->pq, f);
        WRITE1(idxp_2->code_size_1);
        WRITE1(idxp_2->code_size_2);
        WRITE1(idxp_2->code_size);
        WRITEVECTOR(idxp_2->codes);
    } else if (
            const IndexScalarQuantizer* idxs =
                    dynamic_cast<const IndexScalarQuantizer*>(idx)) {
        uint32_t h = fourcc("IxSQ");
        WRITE1(h);
        write_index_header(idx, f);
        write_ScalarQuantizer(&idxs->sq, f);
        WRITEVECTOR(idxs->codes);
    } else if (
            const IndexLattice* idxl_2 =
                    dynamic_cast<const IndexLattice*>(idx)) {
        uint32_t h = fourcc("IxLa");
        WRITE1(h);
        WRITE1(idxl_2->d);
        WRITE1(idxl_2->nsq);
        WRITE1(idxl_2->scale_nbit);
        WRITE1(idxl_2->zn_sphere_codec.r2);
        write_index_header(idx, f);
        WRITEVECTOR(idxl_2->trained);
    } else if (
            const IndexIVFFlatDedup* ivfl =
                    dynamic_cast<const IndexIVFFlatDedup*>(idx)) {
        uint32_t h = fourcc("IwFd");
        WRITE1(h);
        write_ivf_header(ivfl, f);
        {
            std::vector<idx_t> tab(2 * ivfl->instances.size());
            long i = 0;
            for (auto it = ivfl->instances.begin(); it != ivfl->instances.end();
                 ++it) {
                tab[i++] = it->first;
                tab[i++] = it->second;
            }
            WRITEVECTOR(tab);
        }
        write_InvertedLists(ivfl->invlists, f);
    } else if (
            const IndexIVFFlat* ivfl_2 =
                    dynamic_cast<const IndexIVFFlat*>(idx)) {
        uint32_t h = fourcc("IwFl");
        WRITE1(h);
        write_ivf_header(ivfl_2, f);
        write_InvertedLists(ivfl_2->invlists, f);
    } else if (
            const IndexIVFScalarQuantizer* ivsc =
                    dynamic_cast<const IndexIVFScalarQuantizer*>(idx)) {
        uint32_t h = fourcc("IwSq");
        WRITE1(h);
        write_ivf_header(ivsc, f);
        write_ScalarQuantizer(&ivsc->sq, f);
        WRITE1(ivsc->code_size);
        WRITE1(ivsc->by_residual);
        write_InvertedLists(ivsc->invlists, f);
    } else if (auto iva = dynamic_cast<const IndexIVFAdditiveQuantizer*>(idx)) {
        bool is_LSQ = dynamic_cast<const IndexIVFLocalSearchQuantizer*>(iva);
        bool is_RQ = dynamic_cast<const IndexIVFResidualQuantizer*>(iva);
        bool is_PLSQ =
                dynamic_cast<const IndexIVFProductLocalSearchQuantizer*>(iva);
        uint32_t h;
        if (is_LSQ) {
            h = fourcc("IwLS");
        } else if (is_RQ) {
            h = fourcc("IwRQ");
        } else if (is_PLSQ) {
            h = fourcc("IwPL");
        } else {
            h = fourcc("IwPR");
        }

        WRITE1(h);
        write_ivf_header(iva, f);
        WRITE1(iva->code_size);
        if (is_LSQ) {
            write_LocalSearchQuantizer((LocalSearchQuantizer*)iva->aq, f);
        } else if (is_RQ) {
            write_ResidualQuantizer((ResidualQuantizer*)iva->aq, f);
        } else if (is_PLSQ) {
            write_ProductLocalSearchQuantizer(
                    (ProductLocalSearchQuantizer*)iva->aq, f);
        } else {
            write_ProductResidualQuantizer(
                    (ProductResidualQuantizer*)iva->aq, f);
        }
        WRITE1(iva->by_residual);
        WRITE1(iva->use_precomputed_table);
        write_InvertedLists(iva->invlists, f);
    } else if (
            const IndexIVFSpectralHash* ivsp =
                    dynamic_cast<const IndexIVFSpectralHash*>(idx)) {
        uint32_t h = fourcc("IwSh");
        WRITE1(h);
        write_ivf_header(ivsp, f);
        write_VectorTransform(ivsp->vt, f);
        WRITE1(ivsp->nbit);
        WRITE1(ivsp->period);
        WRITE1(ivsp->threshold_type);
        WRITEVECTOR(ivsp->trained);
        write_InvertedLists(ivsp->invlists, f);
    } else if (const IndexIVFPQ* ivpq = dynamic_cast<const IndexIVFPQ*>(idx)) {
        const IndexIVFPQR* ivfpqr = dynamic_cast<const IndexIVFPQR*>(idx);

        uint32_t h = fourcc(ivfpqr ? "IwQR" : "IwPQ");
        WRITE1(h);
        write_ivf_header(ivpq, f);
        WRITE1(ivpq->by_residual);
        WRITE1(ivpq->code_size);
        write_ProductQuantizer(&ivpq->pq, f);
        write_InvertedLists(ivpq->invlists, f);
        if (ivfpqr) {
            write_ProductQuantizer(&ivfpqr->refine_pq, f);
            WRITEVECTOR(ivfpqr->refine_codes);
            WRITE1(ivfpqr->k_factor);
        }
    } else if (
            auto* indep =
                    dynamic_cast<const IndexIVFIndependentQuantizer*>(idx)) {
        uint32_t h = fourcc("IwIQ");
        WRITE1(h);
        write_index_header(indep, f);
        write_index(indep->quantizer, f);
        bool has_vt = indep->vt != nullptr;
        WRITE1(has_vt);
        if (has_vt) {
            write_VectorTransform(indep->vt, f);
        }
        write_index(indep->index_ivf, f);
        if (auto index_ivfpq = dynamic_cast<IndexIVFPQ*>(indep->index_ivf)) {
            WRITE1(index_ivfpq->use_precomputed_table);
        }
    } else if (
            const IndexPreTransform* ixpt =
                    dynamic_cast<const IndexPreTransform*>(idx)) {
        uint32_t h = fourcc("IxPT");
        WRITE1(h);
        write_index_header(ixpt, f);
        int nt = ixpt->chain.size();
        WRITE1(nt);
        for (int i = 0; i < nt; i++) {
            write_VectorTransform(ixpt->chain[i], f);
        }
        write_index(ixpt->index, f);
    } else if (
            const MultiIndexQuantizer* imiq =
                    dynamic_cast<const MultiIndexQuantizer*>(idx)) {
        uint32_t h = fourcc("Imiq");
        WRITE1(h);
        write_index_header(imiq, f);
        write_ProductQuantizer(&imiq->pq, f);
    } else if (
            const IndexRefine* idxrf = dynamic_cast<const IndexRefine*>(idx)) {
        uint32_t h = fourcc("IxRF");
        WRITE1(h);
        write_index_header(idxrf, f);
        write_index(idxrf->base_index, f);
        write_index(idxrf->refine_index, f);
        WRITE1(idxrf->k_factor);
    } else if (
            const IndexIDMap* idxmap = dynamic_cast<const IndexIDMap*>(idx)) {
        uint32_t h = dynamic_cast<const IndexIDMap2*>(idx) ? fourcc("IxM2")
                                                           : fourcc("IxMp");
        // no need to store additional info for IndexIDMap2
        WRITE1(h);
        write_index_header(idxmap, f);
        write_index(idxmap->index, f);
        WRITEVECTOR(idxmap->id_map);
    } else if (const IndexHNSW* idxhnsw = dynamic_cast<const IndexHNSW*>(idx)) {
        uint32_t h = dynamic_cast<const IndexHNSWFlat*>(idx) ? fourcc("IHNf")
                : dynamic_cast<const IndexHNSWPQ*>(idx)      ? fourcc("IHNp")
                : dynamic_cast<const IndexHNSWSQ*>(idx)      ? fourcc("IHNs")
                : dynamic_cast<const IndexHNSW2Level*>(idx)  ? fourcc("IHN2")
                : dynamic_cast<const IndexHNSWCagra*>(idx)   ? fourcc("IHc2")
                                                             : 0;
        FAISS_THROW_IF_NOT(h != 0);
        WRITE1(h);
        write_index_header(idxhnsw, f);
        if (h == fourcc("IHc2")) {
            WRITE1(idxhnsw->keep_max_size_level0);
            auto idx_hnsw_cagra = dynamic_cast<const IndexHNSWCagra*>(idxhnsw);
            WRITE1(idx_hnsw_cagra->base_level_only);
            WRITE1(idx_hnsw_cagra->num_base_level_search_entrypoints);
            WRITE1(idx_hnsw_cagra->numeric_type_);
        }
        write_HNSW(&idxhnsw->hnsw, f);
        if (io_flags & IO_FLAG_SKIP_STORAGE) {
            uint32_t n4 = fourcc("null");
            WRITE1(n4);
        } else {
            write_index(idxhnsw->storage, f);
        }
    } else if (const IndexNSG* idxnsg = dynamic_cast<const IndexNSG*>(idx)) {
        uint32_t h = dynamic_cast<const IndexNSGFlat*>(idx) ? fourcc("INSf")
                : dynamic_cast<const IndexNSGPQ*>(idx)      ? fourcc("INSp")
                : dynamic_cast<const IndexNSGSQ*>(idx)      ? fourcc("INSs")
                                                            : 0;
        FAISS_THROW_IF_NOT(h != 0);
        WRITE1(h);
        write_index_header(idxnsg, f);
        WRITE1(idxnsg->GK);
        WRITE1(idxnsg->build_type);
        WRITE1(idxnsg->nndescent_S);
        WRITE1(idxnsg->nndescent_R);
        WRITE1(idxnsg->nndescent_L);
        WRITE1(idxnsg->nndescent_iter);
        write_NSG(&idxnsg->nsg, f);
        write_index(idxnsg->storage, f);
    } else if (
            const IndexNNDescent* idxnnd =
                    dynamic_cast<const IndexNNDescent*>(idx)) {
        auto idxnndflat = dynamic_cast<const IndexNNDescentFlat*>(idx);
        FAISS_THROW_IF_NOT(idxnndflat != nullptr);
        uint32_t h = fourcc("INNf");
        FAISS_THROW_IF_NOT(h != 0);
        WRITE1(h);
        write_index_header(idxnnd, f);
        write_NNDescent(&idxnnd->nndescent, f);
        write_index(idxnnd->storage, f);
    } else if (
            const IndexPQFastScan* idxpqfs =
                    dynamic_cast<const IndexPQFastScan*>(idx)) {
        uint32_t h = fourcc("IPfs");
        WRITE1(h);
        write_index_header(idxpqfs, f);
        write_ProductQuantizer(&idxpqfs->pq, f);
        WRITE1(idxpqfs->implem);
        WRITE1(idxpqfs->bbs);
        WRITE1(idxpqfs->qbs);
        WRITE1(idxpqfs->ntotal2);
        WRITE1(idxpqfs->M2);
        WRITEVECTOR(idxpqfs->codes);
    } else if (
            const IndexIVFPQFastScan* ivpq_2 =
                    dynamic_cast<const IndexIVFPQFastScan*>(idx)) {
        uint32_t h = fourcc("IwPf");
        WRITE1(h);
        write_ivf_header(ivpq_2, f);
        WRITE1(ivpq_2->by_residual);
        WRITE1(ivpq_2->code_size);
        WRITE1(ivpq_2->bbs);
        WRITE1(ivpq_2->M2);
        WRITE1(ivpq_2->implem);
        WRITE1(ivpq_2->qbs2);
        write_ProductQuantizer(&ivpq_2->pq, f);
        write_InvertedLists(ivpq_2->invlists, f);
    } else if (
            const IndexRowwiseMinMax* imm =
                    dynamic_cast<const IndexRowwiseMinMax*>(idx)) {
        // IndexRowwiseMinmaxFloat
        uint32_t h = fourcc("IRMf");
        WRITE1(h);
        write_index_header(imm, f);
        write_index(imm->index, f);
    } else if (
            const IndexRowwiseMinMaxFP16* imm_2 =
                    dynamic_cast<const IndexRowwiseMinMaxFP16*>(idx)) {
        // IndexRowwiseMinmaxHalf
        uint32_t h = fourcc("IRMh");
        WRITE1(h);
        write_index_header(imm_2, f);
        write_index(imm_2->index, f);
    } else if (
            const IndexRaBitQ* idxq = dynamic_cast<const IndexRaBitQ*>(idx)) {
        uint32_t h = fourcc("Ixrq");
        WRITE1(h);
        write_index_header(idx, f);
        write_RaBitQuantizer(&idxq->rabitq, f);
        WRITEVECTOR(idxq->codes);
        WRITEVECTOR(idxq->center);
        WRITE1(idxq->qb);
    } else if (
            const IndexIVFRaBitQ* ivrq =
                    dynamic_cast<const IndexIVFRaBitQ*>(idx)) {
        uint32_t h = fourcc("Iwrq");
        WRITE1(h);
        write_ivf_header(ivrq, f);
        write_RaBitQuantizer(&ivrq->rabitq, f);
        WRITE1(ivrq->code_size);
        WRITE1(ivrq->by_residual);
        WRITE1(ivrq->qb);
        write_InvertedLists(ivrq->invlists, f);
    } else {
        FAISS_THROW_MSG("don't know how to serialize this type of index");
    }
}

void write_index(const Index* idx, FILE* f, int io_flags) {
    FileIOWriter writer(f);
    write_index(idx, &writer, io_flags);
}

void write_index(const Index* idx, const char* fname, int io_flags) {
    FileIOWriter writer(fname);
    write_index(idx, &writer, io_flags);
}

void write_VectorTransform(const VectorTransform* vt, const char* fname) {
    FileIOWriter writer(fname);
    write_VectorTransform(vt, &writer);
}

/*************************************************************
 * Write binary indexes
 **************************************************************/

static void write_index_binary_header(const IndexBinary* idx, IOWriter* f) {
    WRITE1(idx->d);
    WRITE1(idx->code_size);
    WRITE1(idx->ntotal);
    WRITE1(idx->is_trained);
    WRITE1(idx->metric_type);
}

static void write_binary_ivf_header(const IndexBinaryIVF* ivf, IOWriter* f) {
    write_index_binary_header(ivf, f);
    WRITE1(ivf->nlist);
    WRITE1(ivf->nprobe);
    write_index_binary(ivf->quantizer, f);
    write_direct_map(&ivf->direct_map, f);
}

static void write_binary_hash_invlists(
        const IndexBinaryHash::InvertedListMap& invlists,
        int b,
        IOWriter* f) {
    size_t sz = invlists.size();
    WRITE1(sz);
    size_t maxil = 0;
    for (auto it = invlists.begin(); it != invlists.end(); ++it) {
        if (it->second.ids.size() > maxil) {
            maxil = it->second.ids.size();
        }
    }
    int il_nbit = 0;
    while (maxil >= ((uint64_t)1 << il_nbit)) {
        il_nbit++;
    }
    WRITE1(il_nbit);

    // first write sizes then data, may be useful if we want to
    // memmap it at some point

    // buffer for bitstrings
    std::vector<uint8_t> buf(((b + il_nbit) * sz + 7) / 8);
    BitstringWriter wr(buf.data(), buf.size());
    for (auto it = invlists.begin(); it != invlists.end(); ++it) {
        wr.write(it->first, b);
        wr.write(it->second.ids.size(), il_nbit);
    }
    WRITEVECTOR(buf);

    for (auto it = invlists.begin(); it != invlists.end(); ++it) {
        WRITEVECTOR(it->second.ids);
        WRITEVECTOR(it->second.vecs);
    }
}

static void write_binary_multi_hash_map(
        const IndexBinaryMultiHash::Map& map,
        int b,
        size_t ntotal,
        IOWriter* f) {
    int id_bits = 0;
    while ((ntotal > ((idx_t)1 << id_bits))) {
        id_bits++;
    }
    WRITE1(id_bits);
    size_t sz = map.size();
    WRITE1(sz);
    size_t nbit = (b + id_bits) * sz + ntotal * id_bits;
    std::vector<uint8_t> buf((nbit + 7) / 8);
    BitstringWriter wr(buf.data(), buf.size());
    for (auto it = map.begin(); it != map.end(); ++it) {
        wr.write(it->first, b);
        wr.write(it->second.size(), id_bits);
        for (auto id : it->second) {
            wr.write(id, id_bits);
        }
    }
    WRITEVECTOR(buf);
}

void write_index_binary(const IndexBinary* idx, IOWriter* f) {
    if (const IndexBinaryFlat* idxf =
                dynamic_cast<const IndexBinaryFlat*>(idx)) {
        uint32_t h = fourcc("IBxF");
        WRITE1(h);
        write_index_binary_header(idx, f);
        WRITEVECTOR(idxf->xb);
    } else if (
            const IndexBinaryIVF* ivf =
                    dynamic_cast<const IndexBinaryIVF*>(idx)) {
        uint32_t h = fourcc("IBwF");
        WRITE1(h);
        write_binary_ivf_header(ivf, f);
        write_InvertedLists(ivf->invlists, f);
    } else if (
            const IndexBinaryFromFloat* idxff =
                    dynamic_cast<const IndexBinaryFromFloat*>(idx)) {
        uint32_t h = fourcc("IBFf");
        WRITE1(h);
        write_index_binary_header(idxff, f);
        write_index(idxff->index, f);
    } else if (
            const IndexBinaryHNSW* idxhnsw =
                    dynamic_cast<const IndexBinaryHNSW*>(idx)) {
        uint32_t h = fourcc("IBHf");
        WRITE1(h);
        write_index_binary_header(idxhnsw, f);
        write_HNSW(&idxhnsw->hnsw, f);
        write_index_binary(idxhnsw->storage, f);
    } else if (
            const IndexBinaryIDMap* idxmap =
                    dynamic_cast<const IndexBinaryIDMap*>(idx)) {
        uint32_t h = dynamic_cast<const IndexBinaryIDMap2*>(idx)
                ? fourcc("IBM2")
                : fourcc("IBMp");
        // no need to store additional info for IndexIDMap2
        WRITE1(h);
        write_index_binary_header(idxmap, f);
        write_index_binary(idxmap->index, f);
        WRITEVECTOR(idxmap->id_map);
    } else if (
            const IndexBinaryHash* idxh =
                    dynamic_cast<const IndexBinaryHash*>(idx)) {
        uint32_t h = fourcc("IBHh");
        WRITE1(h);
        write_index_binary_header(idxh, f);
        WRITE1(idxh->b);
        WRITE1(idxh->nflip);
        write_binary_hash_invlists(idxh->invlists, idxh->b, f);
    } else if (
            const IndexBinaryMultiHash* idxmh =
                    dynamic_cast<const IndexBinaryMultiHash*>(idx)) {
        uint32_t h = fourcc("IBHm");
        WRITE1(h);
        write_index_binary_header(idxmh, f);
        write_index_binary(idxmh->storage, f);
        WRITE1(idxmh->b);
        WRITE1(idxmh->nhash);
        WRITE1(idxmh->nflip);
        for (int i = 0; i < idxmh->nhash; i++) {
            write_binary_multi_hash_map(
                    idxmh->maps[i], idxmh->b, idxmh->ntotal, f);
        }
    } else {
        FAISS_THROW_MSG("don't know how to serialize this type of index");
    }
}

void write_index_binary(const IndexBinary* idx, FILE* f) {
    FileIOWriter writer(f);
    write_index_binary(idx, &writer);
}

void write_index_binary(const IndexBinary* idx, const char* fname) {
    FileIOWriter writer(fname);
    write_index_binary(idx, &writer);
}

} // namespace faiss
