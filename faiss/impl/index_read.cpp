/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/index_read_utils.h>
#include <faiss/index_io.h>

#include <faiss/impl/io_macros.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/utils/hamming.h>

#include <faiss/invlists/InvertedListsIOHook.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexAdditiveQuantizerFastScan.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/IndexIVFAdditiveQuantizerFastScan.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFFlatPanorama.h>
#include <faiss/IndexIVFIndependentQuantizer.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexIVFPQR.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexIVFRaBitQFastScan.h>
#include <faiss/IndexIVFSpectralHash.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexLattice.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/IndexRefine.h>
#include <faiss/IndexRowwiseMinMax.h>
#ifdef FAISS_ENABLE_SVS
#include <faiss/impl/svs_io.h>
#include <faiss/svs/IndexSVSFlat.h>
#include <faiss/svs/IndexSVSVamana.h>
#include <faiss/svs/IndexSVSVamanaLVQ.h>
#include <faiss/svs/IndexSVSVamanaLeanVec.h>
#endif
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <faiss/VectorTransform.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryFromFloat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/IndexBinaryIVF.h>

// mmap-ing and viewing facilities
#include <faiss/impl/maybe_owned_vector.h>

#include <faiss/impl/mapped_io.h>
#include <faiss/impl/zerocopy_io.h>

namespace faiss {

/*************************************************************
 * Mmap-ing and viewing facilities
 **************************************************************/

// This is a baseline functionality for reading mmapped and zerocopied vector.
// * if `beforeknown_size` is defined, then a size of the vector won't be
// read.
// * if `size_multiplier` is defined, then a size will be multiplied by it.
// * returns true is the case was handled; otherwise, false
template <typename VectorT>
bool read_vector_base(
        VectorT& target,
        IOReader* f,
        const std::optional<size_t> beforeknown_size,
        const std::optional<size_t> size_multiplier) {
    // check if the use case is right
    if constexpr (is_maybe_owned_vector_v<VectorT>) {
        // is it a mmap-enabled reader?
        MappedFileIOReader* mf = dynamic_cast<MappedFileIOReader*>(f);
        if (mf != nullptr) {
            // read the size or use a known one
            size_t size = 0;
            if (beforeknown_size.has_value()) {
                size = beforeknown_size.value();
            } else {
                READANDCHECK(&size, 1);
            }

            // perform the size multiplication
            size *= size_multiplier.value_or(1);

            // ok, mmap and check
            char* address = nullptr;
            const size_t nread = mf->mmap(
                    (void**)&address,
                    sizeof(typename VectorT::value_type),
                    size);

            FAISS_THROW_IF_NOT_FMT(
                    nread == (size),
                    "read error in %s: %zd != %zd (%s)",
                    f->name.c_str(),
                    nread,
                    size,
                    strerror(errno));

            VectorT mmapped_view =
                    VectorT::create_view(address, nread, mf->mmap_owner);
            target = std::move(mmapped_view);

            return true;
        }

        // is it a zero-copy reader?
        ZeroCopyIOReader* zr = dynamic_cast<ZeroCopyIOReader*>(f);
        if (zr != nullptr) {
            // read the size or use a known one
            size_t size = 0;
            if (beforeknown_size.has_value()) {
                size = beforeknown_size.value();
            } else {
                READANDCHECK(&size, 1);
            }

            // perform the size multiplication
            size *= size_multiplier.value_or(1);

            // create a view
            char* address = nullptr;
            size_t nread = zr->get_data_view(
                    (void**)&address,
                    sizeof(typename VectorT::value_type),
                    size);

            FAISS_THROW_IF_NOT_FMT(
                    nread == (size),
                    "read error in %s: %zd != %zd (%s)",
                    f->name.c_str(),
                    nread,
                    size_t(size),
                    strerror(errno));

            VectorT view = VectorT::create_view(address, nread, nullptr);
            target = std::move(view);

            return true;
        }
    }

    return false;
}

// a replacement for READANDCHECK for reading data into std::vector
template <typename VectorT>
void read_vector_with_known_size(VectorT& target, IOReader* f, size_t size) {
    // size is known beforehand, no size multiplication
    if (read_vector_base<VectorT>(target, f, size, std::nullopt)) {
        return;
    }

    // the default case
    READANDCHECK(target.data(), size);
}

// a replacement for READVECTOR
template <typename VectorT>
void read_vector(VectorT& target, IOReader* f) {
    // size is not known beforehand, no size multiplication
    if (read_vector_base<VectorT>(target, f, std::nullopt, std::nullopt)) {
        return;
    }

    // the default case
    READVECTOR(target);
}

// a replacement for READXBVECTOR
template <typename VectorT>
void read_xb_vector(VectorT& target, IOReader* f) {
    // size is not known beforehand, multiply the size 4x
    if (read_vector_base<VectorT>(target, f, std::nullopt, 4)) {
        return;
    }

    // the default case
    READXBVECTOR(target);
}

/*************************************************************
 * Read
 **************************************************************/

static void read_index_header(Index& idx, IOReader* f) {
    READ1(idx.d);
    READ1(idx.ntotal);
    FAISS_CHECK_RANGE(idx.d, 0, (1 << 20) + 1);
    FAISS_THROW_IF_NOT_FMT(
            idx.ntotal >= 0,
            "invalid ntotal %" PRId64 " read from index",
            (int64_t)idx.ntotal);
    idx_t dummy;
    READ1(dummy);
    READ1(dummy);
    READ1(idx.is_trained);
    READ1(idx.metric_type);
    if (idx.metric_type > 1) {
        READ1(idx.metric_arg);
    }
    idx.verbose = false;
}

std::unique_ptr<VectorTransform> read_VectorTransform_up(IOReader* f) {
    uint32_t h;
    READ1(h);
    std::unique_ptr<VectorTransform> vt;

    if (h == fourcc("rrot") || h == fourcc("PCAm") || h == fourcc("LTra") ||
        h == fourcc("PcAm") || h == fourcc("Viqm") || h == fourcc("Pcam")) {
        std::unique_ptr<LinearTransform> lt;
        if (h == fourcc("rrot")) {
            lt = std::make_unique<RandomRotationMatrix>();
        } else if (
                h == fourcc("PCAm") || h == fourcc("PcAm") ||
                h == fourcc("Pcam")) {
            auto pca = std::make_unique<PCAMatrix>();
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
            lt = std::move(pca);
        } else if (h == fourcc("Viqm")) {
            auto itqm = std::make_unique<ITQMatrix>();
            READ1(itqm->max_iter);
            READ1(itqm->seed);
            lt = std::move(itqm);
        } else if (h == fourcc("LTra")) {
            lt = std::make_unique<LinearTransform>();
        }
        READ1(lt->have_bias);
        READVECTOR(lt->A);
        READVECTOR(lt->b);
        FAISS_THROW_IF_NOT(lt->A.size() >= lt->d_in * lt->d_out);
        FAISS_THROW_IF_NOT(!lt->have_bias || lt->b.size() >= lt->d_out);
        lt->set_is_orthonormal();
        vt = std::move(lt);
    } else if (h == fourcc("RmDT")) {
        auto rdt = std::make_unique<RemapDimensionsTransform>();
        READVECTOR(rdt->map);
        vt = std::move(rdt);
    } else if (h == fourcc("VNrm")) {
        auto nt = std::make_unique<NormalizationTransform>();
        READ1(nt->norm);
        vt = std::move(nt);
    } else if (h == fourcc("VCnt")) {
        auto ct = std::make_unique<CenteringTransform>();
        READVECTOR(ct->mean);
        vt = std::move(ct);
    } else if (h == fourcc("Viqt")) {
        auto itqt = std::make_unique<ITQTransform>();

        READVECTOR(itqt->mean);
        READ1(itqt->do_pca);
        {
            // Read, dereference, discard.
            auto sub_vt = read_VectorTransform_up(f);
            ITQMatrix* itqm = dynamic_cast<ITQMatrix*>(sub_vt.get());
            FAISS_THROW_IF_NOT(itqm);
            itqt->itq = *itqm;
        }
        {
            // Read, dereference, discard.
            auto sub_vt = read_VectorTransform_up(f);
            LinearTransform* pi = dynamic_cast<LinearTransform*>(sub_vt.get());
            FAISS_THROW_IF_NOT(pi);
            itqt->pca_then_itq = *pi;
        }
        vt = std::move(itqt);
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

VectorTransform* read_VectorTransform(IOReader* f) {
    return read_VectorTransform_up(f).release();
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
        FAISS_THROW_IF_NOT_FMT(
                idsizes.size() % 2 == 0,
                "invalid sparse inverted list size: %zd (must be even)",
                idsizes.size());
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

std::unique_ptr<InvertedLists> read_InvertedLists_up(
        IOReader* f,
        int io_flags) {
    uint32_t h;
    READ1(h);
    if (h == fourcc("il00")) {
        fprintf(stderr,
                "read_InvertedLists:"
                " WARN! inverted lists not stored with IVF object\n");
        return nullptr;
    } else if (h == fourcc("ilpn") && !(io_flags & IO_FLAG_SKIP_IVF_DATA)) {
        size_t nlist, code_size, n_levels;
        READ1(nlist);
        READ1(code_size);
        READ1(n_levels);
        auto ailp = std::make_unique<ArrayInvertedListsPanorama>(
                nlist, code_size, n_levels);
        std::vector<size_t> sizes(nlist);
        read_ArrayInvertedLists_sizes(f, sizes);
        for (size_t i = 0; i < nlist; i++) {
            ailp->ids[i].resize(sizes[i]);
            size_t num_elems =
                    ((sizes[i] + ArrayInvertedListsPanorama::kBatchSize - 1) /
                     ArrayInvertedListsPanorama::kBatchSize) *
                    ArrayInvertedListsPanorama::kBatchSize;
            ailp->codes[i].resize(num_elems * code_size);
            ailp->cum_sums[i].resize(num_elems * (n_levels + 1));
        }
        for (size_t i = 0; i < nlist; i++) {
            size_t n = sizes[i];
            if (n > 0) {
                read_vector_with_known_size(
                        ailp->codes[i], f, ailp->codes[i].size());
                read_vector_with_known_size(ailp->ids[i], f, n);
                read_vector_with_known_size(
                        ailp->cum_sums[i], f, ailp->cum_sums[i].size());
            }
        }
        return ailp;
    } else if (h == fourcc("ilar") && !(io_flags & IO_FLAG_SKIP_IVF_DATA)) {
        auto ails = std::make_unique<ArrayInvertedLists>(0, 0);
        READ1(ails->nlist);
        READ1(ails->code_size);
        ails->ids.resize(ails->nlist);
        ails->codes.resize(ails->nlist);
        std::vector<size_t> sizes(ails->nlist);
        read_ArrayInvertedLists_sizes(f, sizes);
        for (size_t i = 0; i < ails->nlist; i++) {
            ails->ids[i].resize(sizes[i]);
            ails->codes[i].resize(mul_no_overflow(
                    sizes[i], ails->code_size, "inverted list codes"));
        }
        for (size_t i = 0; i < ails->nlist; i++) {
            size_t n = ails->ids[i].size();
            if (n > 0) {
                read_vector_with_known_size(
                        ails->codes[i],
                        f,
                        mul_no_overflow(
                                n, ails->code_size, "inverted list codes"));
                read_vector_with_known_size(ails->ids[i], f, n);
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
        return std::unique_ptr<InvertedLists>(
                InvertedListsIOHook::lookup(h2)->read_ArrayInvertedLists(
                        f, io_flags, nlist, code_size, sizes));
    } else {
        return std::unique_ptr<InvertedLists>(
                InvertedListsIOHook::lookup(h)->read(f, io_flags));
    }
}

InvertedLists* read_InvertedLists(IOReader* f, int io_flags) {
    return read_InvertedLists_up(f, io_flags).release();
}

void read_InvertedLists(IndexIVF& ivf, IOReader* f, int io_flags) {
    InvertedLists* ils = read_InvertedLists(f, io_flags);
    if (ils) {
        FAISS_THROW_IF_NOT(ils->nlist == ivf.nlist);
        FAISS_THROW_IF_NOT(
                ils->code_size == InvertedLists::INVALID_CODE_SIZE ||
                ils->code_size == ivf.code_size);
    }
    ivf.invlists = ils;
    ivf.own_invlists = true;
}

void read_ProductQuantizer(ProductQuantizer* pq, IOReader* f) {
    READ1(pq->d);
    READ1(pq->M);
    READ1(pq->nbits);
    FAISS_THROW_IF_NOT_FMT(
            pq->M > 0, "invalid ProductQuantizer M=%zd (must be > 0)", pq->M);
    pq->set_derived_values();
    READVECTOR(pq->centroids);
}

static void read_ResidualQuantizer_old(ResidualQuantizer& rq, IOReader* f) {
    READ1(rq.d);
    READ1(rq.M);
    READVECTOR(rq.nbits);
    FAISS_THROW_IF_NOT_FMT(
            rq.nbits.size() == rq.M,
            "ResidualQuantizer nbits size %zd != M %zd",
            rq.nbits.size(),
            rq.M);
    READ1(rq.is_trained);
    READ1(rq.train_type);
    READ1(rq.max_beam_size);
    READVECTOR(rq.codebooks);
    READ1(rq.search_type);
    READ1(rq.norm_min);
    READ1(rq.norm_max);
    rq.set_derived_values();
}

static void read_AdditiveQuantizer(AdditiveQuantizer& aq, IOReader* f) {
    READ1(aq.d);
    READ1(aq.M);
    READVECTOR(aq.nbits);
    READ1(aq.is_trained);
    READVECTOR(aq.codebooks);
    FAISS_THROW_IF_NOT_FMT(
            aq.nbits.size() == aq.M,
            "AdditiveQuantizer nbits size %zd != M %zd",
            aq.nbits.size(),
            aq.M);
    READ1(aq.search_type);
    READ1(aq.norm_min);
    READ1(aq.norm_max);
    if (aq.search_type == AdditiveQuantizer::ST_norm_cqint8 ||
        aq.search_type == AdditiveQuantizer::ST_norm_cqint4 ||
        aq.search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
        aq.search_type == AdditiveQuantizer::ST_norm_rq2x4) {
        read_xb_vector(aq.qnorm.codes, f);
        aq.qnorm.ntotal = aq.qnorm.codes.size() / 4;
        aq.qnorm.update_permutation();
    }

    if (aq.search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
        aq.search_type == AdditiveQuantizer::ST_norm_rq2x4) {
        READVECTOR(aq.norm_tabs);
    }

    aq.set_derived_values();
}

static void read_ResidualQuantizer(
        ResidualQuantizer& rq,
        IOReader* f,
        int io_flags) {
    read_AdditiveQuantizer(rq, f);
    READ1(rq.train_type);
    READ1(rq.max_beam_size);
    if ((rq.train_type & ResidualQuantizer::Skip_codebook_tables) ||
        (io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE)) {
        // don't precompute the tables
    } else {
        rq.compute_codebook_tables();
    }
}

static void read_LocalSearchQuantizer(LocalSearchQuantizer& lsq, IOReader* f) {
    read_AdditiveQuantizer(lsq, f);
    READ1(lsq.K);
    READ1(lsq.train_iters);
    READ1(lsq.encode_ils_iters);
    READ1(lsq.train_ils_iters);
    READ1(lsq.icm_iters);
    READ1(lsq.p);
    READ1(lsq.lambd);
    READ1(lsq.chunk_size);
    READ1(lsq.random_seed);
    READ1(lsq.nperts);
    READ1(lsq.update_codebooks_with_double);
}

static void read_ProductAdditiveQuantizer(
        ProductAdditiveQuantizer& paq,
        IOReader* f) {
    read_AdditiveQuantizer(paq, f);
    READ1(paq.nsplits);
}

static void read_ProductResidualQuantizer(
        ProductResidualQuantizer& prq,
        IOReader* f,
        int io_flags) {
    read_ProductAdditiveQuantizer(prq, f);

    for (size_t i = 0; i < prq.nsplits; i++) {
        auto rq = std::make_unique<ResidualQuantizer>();
        read_ResidualQuantizer(*rq, f, io_flags);
        prq.quantizers.push_back(rq.release());
    }
}

static void read_ProductLocalSearchQuantizer(
        ProductLocalSearchQuantizer& plsq,
        IOReader* f) {
    read_ProductAdditiveQuantizer(plsq, f);

    for (size_t i = 0; i < plsq.nsplits; i++) {
        auto lsq = std::make_unique<LocalSearchQuantizer>();
        read_LocalSearchQuantizer(*lsq, f);
        plsq.quantizers.push_back(lsq.release());
    }
}

void read_ScalarQuantizer(ScalarQuantizer* ivsc, IOReader* f) {
    READ1(ivsc->qtype);
    READ1(ivsc->rangestat);
    READ1(ivsc->rangestat_arg);
    READ1(ivsc->d);
    READ1(ivsc->code_size);
    READVECTOR(ivsc->trained);
    ivsc->set_derived_sizes();
}

static void validate_HNSW(const HNSW& hnsw) {
    size_t ntotal = hnsw.levels.size();
    size_t nb_neighbors_size = hnsw.neighbors.size();

    // cum_nneighbor_per_level must be non-empty and monotonically
    // non-decreasing, starting at 0
    if (!hnsw.cum_nneighbor_per_level.empty()) {
        FAISS_THROW_IF_NOT_FMT(
                hnsw.cum_nneighbor_per_level[0] == 0,
                "HNSW cum_nneighbor_per_level[0] = %d, expected 0",
                hnsw.cum_nneighbor_per_level[0]);
        for (size_t i = 1; i < hnsw.cum_nneighbor_per_level.size(); i++) {
            FAISS_THROW_IF_NOT_FMT(
                    hnsw.cum_nneighbor_per_level[i] >=
                            hnsw.cum_nneighbor_per_level[i - 1],
                    "HNSW cum_nneighbor_per_level not monotonic at %zd: "
                    "%d < %d",
                    i,
                    hnsw.cum_nneighbor_per_level[i],
                    hnsw.cum_nneighbor_per_level[i - 1]);
        }
    }

    // offsets must have size ntotal + 1, be monotonically non-decreasing,
    // and all values must be <= neighbors.size()
    FAISS_THROW_IF_NOT_FMT(
            hnsw.offsets.size() == ntotal + 1,
            "HNSW offsets size %zd != levels size %zd + 1",
            hnsw.offsets.size(),
            ntotal);
    for (size_t i = 0; i < hnsw.offsets.size(); i++) {
        FAISS_THROW_IF_NOT_FMT(
                hnsw.offsets[i] <= nb_neighbors_size,
                "HNSW offsets[%zd] = %zd > neighbors.size() = %zd",
                i,
                hnsw.offsets[i],
                nb_neighbors_size);
        if (i > 0) {
            FAISS_THROW_IF_NOT_FMT(
                    hnsw.offsets[i] ==
                            hnsw.offsets[i - 1] +
                                    hnsw.cum_nneighbor_per_level
                                            [hnsw.levels[i - 1]],
                    "HNSW offsets not increasing by cum_neighbor_per_level at %zd: %zd + %d != %zd",
                    i,
                    hnsw.offsets[i - 1],
                    hnsw.cum_nneighbor_per_level[hnsw.levels[i - 1]],
                    hnsw.offsets[i]);
        }
    }

    // max_level must be valid
    FAISS_THROW_IF_NOT_FMT(
            hnsw.max_level < (int)hnsw.cum_nneighbor_per_level.size(),
            "HNSW max_level %d >= cum_nneighbor_per_level size %zd",
            hnsw.max_level,
            hnsw.cum_nneighbor_per_level.size());

    // entry_point must be -1 (empty) or a valid node id
    FAISS_THROW_IF_NOT_FMT(
            hnsw.entry_point >= -1 && hnsw.entry_point < (int)ntotal,
            "HNSW entry_point %d out of range [-1, %zd)",
            (int)hnsw.entry_point,
            ntotal);

    // All neighbor ids must be -1 or in [0, ntotal)
    for (size_t i = 0; i < nb_neighbors_size; i++) {
        auto id = hnsw.neighbors[i];
        FAISS_THROW_IF_NOT_FMT(
                id >= -1 && id < (int)ntotal,
                "HNSW neighbors[%zd] = %d out of range [-1, %zd)",
                i,
                (int)id,
                ntotal);
    }

    // For each node, verify that its level is valid and that
    // offsets[i] + cum_nneighbor_per_level[levels[i]] <= neighbors.size().
    // This ensures neighbor_range() can never produce an out-of-bounds offset
    // into neighbors.
    int cum_levels = (int)hnsw.cum_nneighbor_per_level.size();
    for (size_t i = 0; i < ntotal; i++) {
        int level = hnsw.levels[i];
        FAISS_CHECK_RANGE(level, 1, cum_levels + 1);
        size_t end = hnsw.offsets[i] + hnsw.cum_nneighbor_per_level[level];
        FAISS_THROW_IF_NOT_FMT(
                end <= nb_neighbors_size,
                "HNSW neighbor range overflow for node %zd: "
                "offsets[%zd] (%zd) + cum_nneighbor_per_level[%d] (%d) "
                "= %zd > neighbors.size() (%zd)",
                i,
                i,
                hnsw.offsets[i],
                level,
                hnsw.cum_nneighbor_per_level[level],
                end,
                nb_neighbors_size);
    }
}

static void read_HNSW(HNSW& hnsw, IOReader* f) {
    READVECTOR(hnsw.assign_probas);
    READVECTOR(hnsw.cum_nneighbor_per_level);
    READVECTOR(hnsw.levels);
    READVECTOR(hnsw.offsets);
    read_vector(hnsw.neighbors, f);

    READ1(hnsw.entry_point);
    READ1(hnsw.max_level);
    READ1(hnsw.efConstruction);
    READ1(hnsw.efSearch);

    // // deprecated field
    // READ1(hnsw.upper_beam);
    READ1_DUMMY(int)

    validate_HNSW(hnsw);
}

static void read_NSG(NSG& nsg, IOReader* f) {
    READ1(nsg.ntotal);
    READ1(nsg.R);
    READ1(nsg.L);
    READ1(nsg.C);
    READ1(nsg.search_L);
    READ1(nsg.enterpoint);
    READ1(nsg.is_built);

    FAISS_THROW_IF_NOT_FMT(
            nsg.ntotal >= 0, "invalid NSG ntotal %d", nsg.ntotal);

    if (!nsg.is_built) {
        return;
    }

    constexpr int EMPTY_ID = -1;
    int N = nsg.ntotal;
    int R = nsg.R;

    auto& graph = nsg.final_graph;
    graph = std::make_shared<nsg::Graph<int>>(N, R);
    std::fill_n(graph->data, (size_t)N * R, EMPTY_ID);

    for (int i = 0; i < N; i++) {
        int j;
        for (j = 0; j < R; j++) {
            int id;
            READ1(id);
            if (id != EMPTY_ID) {
                FAISS_CHECK_RANGE(id, 0, N);
                graph->at(i, j) = id;
            } else {
                break;
            }
        }
        if (j == R) {
            // All R neighbor slots were filled; consume the trailing
            // EMPTY_ID sentinel that write_NSG always appends.
            int sentinel;
            READ1(sentinel);
            FAISS_THROW_IF_NOT(sentinel == EMPTY_ID);
        }
    }

    // enterpoint must be a valid node id
    FAISS_CHECK_RANGE(nsg.enterpoint, 0, N);
}

static void read_NNDescent(NNDescent& nnd, IOReader* f) {
    READ1(nnd.ntotal);
    READ1(nnd.d);
    READ1(nnd.K);
    READ1(nnd.S);
    READ1(nnd.R);
    READ1(nnd.L);
    READ1(nnd.iter);
    READ1(nnd.search_L);
    READ1(nnd.random_seed);
    READ1(nnd.has_built);

    FAISS_THROW_IF_NOT_FMT(
            nnd.ntotal >= 0, "invalid NNDescent ntotal %d", nnd.ntotal);

    READVECTOR(nnd.final_graph);
}

std::unique_ptr<ProductQuantizer> read_ProductQuantizer_up(const char* fname) {
    FileIOReader reader(fname);
    return read_ProductQuantizer_up(&reader);
}

ProductQuantizer* read_ProductQuantizer(const char* fname) {
    return read_ProductQuantizer_up(fname).release();
}

std::unique_ptr<ProductQuantizer> read_ProductQuantizer_up(IOReader* reader) {
    auto pq = std::make_unique<ProductQuantizer>();
    read_ProductQuantizer(pq.get(), reader);
    return pq;
}

ProductQuantizer* read_ProductQuantizer(IOReader* reader) {
    return read_ProductQuantizer_up(reader).release();
}

static void read_RaBitQuantizer(
        RaBitQuantizer& rabitq,
        IOReader* f,
        bool multi_bit = true) {
    READ1(rabitq.d);
    READ1(rabitq.code_size);
    READ1(rabitq.metric_type);

    if (multi_bit) {
        READ1(rabitq.nb_bits);
    } else {
        rabitq.nb_bits = 1;
    }
}

void read_direct_map(DirectMap* dm, IOReader* f) {
    char maintain_direct_map;
    READ1(maintain_direct_map);
    dm->type = (DirectMap::Type)maintain_direct_map;
    READVECTOR(dm->array);
    if (dm->type == DirectMap::Hashtable) {
        std::vector<std::pair<idx_t, idx_t>> v;
        READVECTOR(v);
        std::unordered_map<idx_t, idx_t>& map = dm->hashtable;
        map.reserve(v.size());
        for (auto it : v) {
            map[it.first] = it.second;
        }
    }
}

void read_ivf_header(
        IndexIVF* ivf,
        IOReader* f,
        std::vector<std::vector<idx_t>>* ids) {
    read_index_header(*ivf, f);
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
ArrayInvertedLists* set_array_invlist(
        IndexIVF* ivf,
        std::vector<std::vector<idx_t>>& ids) {
    auto ail = std::make_unique<ArrayInvertedLists>(ivf->nlist, ivf->code_size);

    ail->ids.resize(ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        ail->ids[i] = MaybeOwnedVector<idx_t>(std::move(ids[i]));
    }

    ArrayInvertedLists* result = ail.get();
    ivf->invlists = ail.release();
    ivf->own_invlists = true;
    return result;
}

static std::unique_ptr<IndexIVFPQ> read_ivfpq(
        IOReader* f,
        uint32_t h,
        int io_flags) {
    bool legacy = h == fourcc("IvQR") || h == fourcc("IvPQ");

    IndexIVFPQR* ivfpqr = nullptr;
    std::unique_ptr<IndexIVFPQ> ivpq;
    if (h == fourcc("IvQR") || h == fourcc("IwQR")) {
        ivpq = std::make_unique<IndexIVFPQR>();
        ivfpqr = static_cast<IndexIVFPQR*>(ivpq.get());
    } else {
        ivpq = std::make_unique<IndexIVFPQ>();
    }

    std::vector<std::vector<idx_t>> ids;
    read_ivf_header(ivpq.get(), f, legacy ? &ids : nullptr);
    READ1(ivpq->by_residual);
    READ1(ivpq->code_size);
    read_ProductQuantizer(&ivpq->pq, f);

    if (legacy) {
        ArrayInvertedLists* ail = set_array_invlist(ivpq.get(), ids);
        for (size_t i = 0; i < ail->nlist; i++)
            READVECTOR(ail->codes[i]);
    } else {
        read_InvertedLists(*ivpq, f, io_flags);
    }

    if (ivpq->is_trained) {
        // precomputed table not stored. It is cheaper to recompute it.
        // precompute_table() may be disabled with a flag.
        ivpq->use_precomputed_table = 0;
        if (ivpq->by_residual) {
            if ((io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE) == 0) {
                ivpq->precompute_table();
            }
        }
        if (ivfpqr) {
            read_ProductQuantizer(&ivfpqr->refine_pq, f);
            READVECTOR(ivfpqr->refine_codes);
            READ1(ivfpqr->k_factor);
        }
    }
    return ivpq;
}

int read_old_fmt_hack = 0;

std::unique_ptr<Index> read_index_up(IOReader* f, int io_flags) {
    std::unique_ptr<Index> idx;
    uint32_t h;
    READ1(h);
    if (h == fourcc("null")) {
        // denotes a missing index, useful for some cases
        return idx;
    } else if (h == fourcc("IxFP") || h == fourcc("IxFp")) {
        int d;
        size_t n_levels, batch_size;
        READ1(d);
        READ1(n_levels);
        READ1(batch_size);
        std::unique_ptr<IndexFlatPanorama> idxp;
        if (h == fourcc("IxFP")) {
            idxp = std::make_unique<IndexFlatL2Panorama>(
                    d, n_levels, batch_size);
        } else {
            idxp = std::make_unique<IndexFlatIPPanorama>(
                    d, n_levels, batch_size);
        }
        READ1(idxp->ntotal);
        READ1(idxp->is_trained);
        READVECTOR(idxp->codes);
        READVECTOR(idxp->cum_sums);
        idxp->verbose = false;
        idx = std::move(idxp);
    } else if (
            h == fourcc("IxFI") || h == fourcc("IxF2") || h == fourcc("IxFl")) {
        std::unique_ptr<IndexFlat> idxf;
        if (h == fourcc("IxFI")) {
            idxf = std::make_unique<IndexFlatIP>();
        } else if (h == fourcc("IxF2")) {
            idxf = std::make_unique<IndexFlatL2>();
        } else {
            idxf = std::make_unique<IndexFlat>();
        }
        read_index_header(*idxf, f);
        idxf->code_size = idxf->d * sizeof(float);
        read_xb_vector(idxf->codes, f);
        FAISS_THROW_IF_NOT(
                idxf->codes.size() == idxf->ntotal * idxf->code_size);
        idx = std::move(idxf);
    } else if (h == fourcc("IxHE") || h == fourcc("IxHe")) {
        auto idxl = std::make_unique<IndexLSH>();
        read_index_header(*idxl, f);
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
            // Read, dereference, discard.
            auto sub_vt = read_VectorTransform_up(f);
            RandomRotationMatrix* rrot =
                    dynamic_cast<RandomRotationMatrix*>(sub_vt.get());
            FAISS_THROW_IF_NOT_MSG(rrot, "expected a random rotation");
            idxl->rrot = *rrot;
        }
        read_vector(idxl->codes, f);
        FAISS_THROW_IF_NOT(
                idxl->rrot.d_in == idxl->d && idxl->rrot.d_out == idxl->nbits);
        FAISS_THROW_IF_NOT(
                idxl->codes.size() == idxl->ntotal * idxl->code_size);
        idx = std::move(idxl);
    } else if (
            h == fourcc("IxPQ") || h == fourcc("IxPo") || h == fourcc("IxPq")) {
        // IxPQ and IxPo were merged into the same IndexPQ object
        auto idxp = std::make_unique<IndexPQ>();
        read_index_header(*idxp, f);
        read_ProductQuantizer(&idxp->pq, f);
        idxp->code_size = idxp->pq.code_size;
        read_vector(idxp->codes, f);
        if (h == fourcc("IxPo") || h == fourcc("IxPq")) {
            READ1(idxp->search_type);
            READ1(idxp->encode_signs);
            READ1(idxp->polysemous_ht);
        }
        // Old versions of PQ all had metric_type set to INNER_PRODUCT
        // when they were in fact using L2. Therefore, we force metric type
        // to L2 when the old format is detected
        if (h == fourcc("IxPQ") || h == fourcc("IxPo")) {
            idxp->metric_type = METRIC_L2;
        }
        idx = std::move(idxp);
    } else if (h == fourcc("IxRQ") || h == fourcc("IxRq")) {
        auto idxr = std::make_unique<IndexResidualQuantizer>();
        read_index_header(*idxr, f);
        if (h == fourcc("IxRQ")) {
            read_ResidualQuantizer_old(idxr->rq, f);
        } else {
            read_ResidualQuantizer(idxr->rq, f, io_flags);
        }
        READ1(idxr->code_size);
        read_vector(idxr->codes, f);
        idx = std::move(idxr);
    } else if (h == fourcc("IxLS")) {
        auto idxr = std::make_unique<IndexLocalSearchQuantizer>();
        read_index_header(*idxr, f);
        read_LocalSearchQuantizer(idxr->lsq, f);
        READ1(idxr->code_size);
        read_vector(idxr->codes, f);
        idx = std::move(idxr);
    } else if (h == fourcc("IxPR")) {
        auto idxpr = std::make_unique<IndexProductResidualQuantizer>();
        read_index_header(*idxpr, f);
        read_ProductResidualQuantizer(idxpr->prq, f, io_flags);
        READ1(idxpr->code_size);
        read_vector(idxpr->codes, f);
        idx = std::move(idxpr);
    } else if (h == fourcc("IxPL")) {
        auto idxpl = std::make_unique<IndexProductLocalSearchQuantizer>();
        read_index_header(*idxpl, f);
        read_ProductLocalSearchQuantizer(idxpl->plsq, f);
        READ1(idxpl->code_size);
        read_vector(idxpl->codes, f);
        idx = std::move(idxpl);
    } else if (h == fourcc("ImRQ")) {
        auto idxr = std::make_unique<ResidualCoarseQuantizer>();
        read_index_header(*idxr, f);
        read_ResidualQuantizer(idxr->rq, f, io_flags);
        READ1(idxr->beam_factor);
        if (io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE) {
            // then we force the beam factor to -1
            // which skips the table precomputation.
            idxr->beam_factor = -1;
        }
        idxr->set_beam_factor(idxr->beam_factor);
        idx = std::move(idxr);
    } else if (
            h == fourcc("ILfs") || h == fourcc("IRfs") || h == fourcc("IPRf") ||
            h == fourcc("IPLf")) {
        bool is_LSQ = h == fourcc("ILfs");
        bool is_RQ = h == fourcc("IRfs");
        bool is_PLSQ = h == fourcc("IPLf");

        std::unique_ptr<IndexAdditiveQuantizerFastScan> idxaqfs;
        if (is_LSQ) {
            idxaqfs = std::make_unique<IndexLocalSearchQuantizerFastScan>();
        } else if (is_RQ) {
            idxaqfs = std::make_unique<IndexResidualQuantizerFastScan>();
        } else if (is_PLSQ) {
            idxaqfs = std::make_unique<
                    IndexProductLocalSearchQuantizerFastScan>();
        } else {
            idxaqfs = std::make_unique<IndexProductResidualQuantizerFastScan>();
        }
        read_index_header(*idxaqfs, f);

        if (is_LSQ) {
            read_LocalSearchQuantizer(*(LocalSearchQuantizer*)idxaqfs->aq, f);
        } else if (is_RQ) {
            read_ResidualQuantizer(
                    *(ResidualQuantizer*)idxaqfs->aq, f, io_flags);
        } else if (is_PLSQ) {
            read_ProductLocalSearchQuantizer(
                    *(ProductLocalSearchQuantizer*)idxaqfs->aq, f);
        } else {
            read_ProductResidualQuantizer(
                    *(ProductResidualQuantizer*)idxaqfs->aq, f, io_flags);
        }

        READ1(idxaqfs->implem);
        READ1(idxaqfs->bbs);
        READ1(idxaqfs->qbs);

        READ1(idxaqfs->M);
        READ1(idxaqfs->nbits);
        READ1(idxaqfs->ksub);
        READ1(idxaqfs->code_size);
        READ1(idxaqfs->ntotal2);
        READ1(idxaqfs->M2);

        READ1(idxaqfs->rescale_norm);
        READ1(idxaqfs->norm_scale);
        READ1(idxaqfs->max_train_points);

        READVECTOR(idxaqfs->codes);
        idx = std::move(idxaqfs);
    } else if (
            h == fourcc("IVLf") || h == fourcc("IVRf") || h == fourcc("NPLf") ||
            h == fourcc("NPRf")) {
        bool is_LSQ = h == fourcc("IVLf");
        bool is_RQ = h == fourcc("IVRf");
        bool is_PLSQ = h == fourcc("NPLf");

        std::unique_ptr<IndexIVFAdditiveQuantizerFastScan> ivaqfs;
        if (is_LSQ) {
            ivaqfs = std::make_unique<IndexIVFLocalSearchQuantizerFastScan>();
        } else if (is_RQ) {
            ivaqfs = std::make_unique<IndexIVFResidualQuantizerFastScan>();
        } else if (is_PLSQ) {
            ivaqfs = std::make_unique<
                    IndexIVFProductLocalSearchQuantizerFastScan>();
        } else {
            ivaqfs = std::make_unique<
                    IndexIVFProductResidualQuantizerFastScan>();
        }
        read_ivf_header(ivaqfs.get(), f);

        if (is_LSQ) {
            read_LocalSearchQuantizer(*(LocalSearchQuantizer*)ivaqfs->aq, f);
        } else if (is_RQ) {
            read_ResidualQuantizer(
                    *(ResidualQuantizer*)ivaqfs->aq, f, io_flags);
        } else if (is_PLSQ) {
            read_ProductLocalSearchQuantizer(
                    *(ProductLocalSearchQuantizer*)ivaqfs->aq, f);
        } else {
            read_ProductResidualQuantizer(
                    *(ProductResidualQuantizer*)ivaqfs->aq, f, io_flags);
        }

        READ1(ivaqfs->by_residual);
        READ1(ivaqfs->implem);
        READ1(ivaqfs->bbs);
        READ1(ivaqfs->qbs);

        READ1(ivaqfs->M);
        READ1(ivaqfs->nbits);
        READ1(ivaqfs->ksub);
        READ1(ivaqfs->code_size);
        READ1(ivaqfs->qbs2);
        READ1(ivaqfs->M2);

        READ1(ivaqfs->rescale_norm);
        READ1(ivaqfs->norm_scale);
        READ1(ivaqfs->max_train_points);

        read_InvertedLists(*ivaqfs, f, io_flags);
        ivaqfs->init_code_packer();
        idx = std::move(ivaqfs);
    } else if (h == fourcc("IvFl") || h == fourcc("IvFL")) { // legacy
        auto ivfl = std::make_unique<IndexIVFFlat>();
        std::vector<std::vector<idx_t>> ids;
        read_ivf_header(ivfl.get(), f, &ids);
        ivfl->code_size = ivfl->d * sizeof(float);
        ArrayInvertedLists* ail = set_array_invlist(ivfl.get(), ids);

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
        idx = std::move(ivfl);
    } else if (h == fourcc("IwFd")) {
        auto ivfl = std::make_unique<IndexIVFFlatDedup>();
        read_ivf_header(ivfl.get(), f);
        ivfl->code_size = ivfl->d * sizeof(float);
        {
            std::vector<idx_t> tab;
            READVECTOR(tab);
            FAISS_THROW_IF_NOT_FMT(
                    tab.size() % 2 == 0,
                    "invalid IVFFlatDedup instances table size: %zd "
                    "(must be even)",
                    tab.size());
            for (long i = 0; i < tab.size(); i += 2) {
                std::pair<idx_t, idx_t> pair(tab[i], tab[i + 1]);
                ivfl->instances.insert(pair);
            }
        }
        read_InvertedLists(*ivfl, f, io_flags);
        idx = std::move(ivfl);
    } else if (h == fourcc("IwPn")) {
        auto ivfp = std::make_unique<IndexIVFFlatPanorama>();
        read_ivf_header(ivfp.get(), f);
        ivfp->code_size = ivfp->d * sizeof(float);
        READ1(ivfp->n_levels);
        read_InvertedLists(*ivfp, f, io_flags);
        idx = std::move(ivfp);
    } else if (h == fourcc("IwFl")) {
        auto ivfl = std::make_unique<IndexIVFFlat>();
        read_ivf_header(ivfl.get(), f);
        ivfl->code_size = ivfl->d * sizeof(float);
        read_InvertedLists(*ivfl, f, io_flags);
        idx = std::move(ivfl);
    } else if (h == fourcc("IxSQ")) {
        auto idxs = std::make_unique<IndexScalarQuantizer>();
        read_index_header(*idxs, f);
        read_ScalarQuantizer(&idxs->sq, f);
        read_vector(idxs->codes, f);
        idxs->code_size = idxs->sq.code_size;
        idx = std::move(idxs);
    } else if (h == fourcc("IxLa")) {
        int d, nsq, scale_nbit, r2;
        READ1(d);
        READ1(nsq);
        READ1(scale_nbit);
        READ1(r2);
        FAISS_THROW_IF_NOT_FMT(
                nsq > 0, "invalid IndexLattice nsq %d (must be > 0)", nsq);
        FAISS_THROW_IF_NOT_FMT(
                d > 0 && d % nsq == 0,
                "invalid IndexLattice d=%d, nsq=%d (d must be > 0 and divisible by nsq)",
                d,
                nsq);
        FAISS_THROW_IF_NOT_FMT(
                r2 >= 0, "invalid IndexLattice r2 %d (must be >= 0)", r2);
        auto idxl = std::make_unique<IndexLattice>(d, nsq, scale_nbit, r2);
        read_index_header(*idxl, f);
        READVECTOR(idxl->trained);
        idx = std::move(idxl);
    } else if (h == fourcc("IvSQ")) { // legacy
        auto ivsc = std::make_unique<IndexIVFScalarQuantizer>();
        std::vector<std::vector<idx_t>> ids;
        read_ivf_header(ivsc.get(), f, &ids);
        read_ScalarQuantizer(&ivsc->sq, f);
        READ1(ivsc->code_size);
        ArrayInvertedLists* ail = set_array_invlist(ivsc.get(), ids);
        for (int i = 0; i < ivsc->nlist; i++)
            READVECTOR(ail->codes[i]);
        idx = std::move(ivsc);
    } else if (h == fourcc("IwSQ") || h == fourcc("IwSq")) {
        auto ivsc = std::make_unique<IndexIVFScalarQuantizer>();
        read_ivf_header(ivsc.get(), f);
        read_ScalarQuantizer(&ivsc->sq, f);
        READ1(ivsc->code_size);
        if (h == fourcc("IwSQ")) {
            ivsc->by_residual = true;
        } else {
            READ1(ivsc->by_residual);
        }
        read_InvertedLists(*ivsc, f, io_flags);
        idx = std::move(ivsc);
    } else if (
            h == fourcc("IwLS") || h == fourcc("IwRQ") || h == fourcc("IwPL") ||
            h == fourcc("IwPR")) {
        bool is_LSQ = h == fourcc("IwLS");
        bool is_RQ = h == fourcc("IwRQ");
        bool is_PLSQ = h == fourcc("IwPL");
        std::unique_ptr<IndexIVFAdditiveQuantizer> iva;
        if (is_LSQ) {
            iva = std::make_unique<IndexIVFLocalSearchQuantizer>();
        } else if (is_RQ) {
            iva = std::make_unique<IndexIVFResidualQuantizer>();
        } else if (is_PLSQ) {
            iva = std::make_unique<IndexIVFProductLocalSearchQuantizer>();
        } else {
            iva = std::make_unique<IndexIVFProductResidualQuantizer>();
        }
        read_ivf_header(iva.get(), f);
        READ1(iva->code_size);
        if (is_LSQ) {
            read_LocalSearchQuantizer(*(LocalSearchQuantizer*)iva->aq, f);
        } else if (is_RQ) {
            read_ResidualQuantizer(*(ResidualQuantizer*)iva->aq, f, io_flags);
        } else if (is_PLSQ) {
            read_ProductLocalSearchQuantizer(
                    *(ProductLocalSearchQuantizer*)iva->aq, f);
        } else {
            read_ProductResidualQuantizer(
                    *(ProductResidualQuantizer*)iva->aq, f, io_flags);
        }
        READ1(iva->by_residual);
        READ1(iva->use_precomputed_table);
        read_InvertedLists(*iva, f, io_flags);
        idx = std::move(iva);
    } else if (h == fourcc("IwSh")) {
        auto ivsp = std::make_unique<IndexIVFSpectralHash>();
        read_ivf_header(ivsp.get(), f);
        ivsp->vt = read_VectorTransform(f);
        ivsp->own_fields = true;
        READ1(ivsp->nbit);
        // not stored by write_ivf_header
        ivsp->code_size = (ivsp->nbit + 7) / 8;
        READ1(ivsp->period);
        READ1(ivsp->threshold_type);
        READVECTOR(ivsp->trained);
        read_InvertedLists(*ivsp, f, io_flags);
        idx = std::move(ivsp);
    } else if (
            h == fourcc("IvPQ") || h == fourcc("IvQR") || h == fourcc("IwPQ") ||
            h == fourcc("IwQR")) {
        idx = read_ivfpq(f, h, io_flags);
    } else if (h == fourcc("IwIQ")) {
        auto indep = std::make_unique<IndexIVFIndependentQuantizer>();
        indep->own_fields = true;
        read_index_header(*indep, f);
        indep->quantizer = read_index(f, io_flags);
        bool has_vt;
        READ1(has_vt);
        if (has_vt) {
            indep->vt = read_VectorTransform(f);
        }
        indep->index_ivf = dynamic_cast<IndexIVF*>(read_index(f, io_flags));
        FAISS_THROW_IF_NOT(indep->index_ivf);
        if (auto index_ivfpq = dynamic_cast<IndexIVFPQ*>(indep->index_ivf)) {
            READ1(index_ivfpq->use_precomputed_table);
        }
        idx = std::move(indep);
    } else if (h == fourcc("IxPT")) {
        auto ixpt = std::make_unique<IndexPreTransform>();
        ixpt->own_fields = true;
        read_index_header(*ixpt, f);
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
        idx = std::move(ixpt);
    } else if (h == fourcc("Imiq")) {
        auto imiq = std::make_unique<MultiIndexQuantizer>();
        read_index_header(*imiq, f);
        read_ProductQuantizer(&imiq->pq, f);
        idx = std::move(imiq);
    } else if (h == fourcc("IxRF") || h == fourcc("IxRP")) {
        auto idxrf = std::make_unique<IndexRefine>();
        read_index_header(*idxrf, f);
        idxrf->base_index = read_index(f, io_flags);
        idxrf->refine_index = read_index(f, io_flags);
        READ1(idxrf->k_factor);
        if (h == fourcc("IxRP")) {
            // then make a RefineFlatPanorama with it
            auto idxrf_new = std::make_unique<IndexRefinePanorama>();
            static_cast<IndexRefine&>(*idxrf_new) = *idxrf;
            idxrf = std::move(idxrf_new);
        } else if (dynamic_cast<IndexFlat*>(idxrf->refine_index)) {
            // then make a RefineFlat with it
            auto idxrf_new = std::make_unique<IndexRefineFlat>();
            static_cast<IndexRefine&>(*idxrf_new) = *idxrf;
            idxrf = std::move(idxrf_new);
        }
        idxrf->own_fields = true;
        idxrf->own_refine_index = true;
        idx = std::move(idxrf);
    } else if (h == fourcc("IxMp") || h == fourcc("IxM2")) {
        bool is_map2 = h == fourcc("IxM2");
        std::unique_ptr<IndexIDMap> idxmap = is_map2
                ? std::make_unique<IndexIDMap2>()
                : std::make_unique<IndexIDMap>();
        read_index_header(*idxmap, f);
        idxmap->index = read_index(f, io_flags);
        idxmap->own_fields = true;
        READVECTOR(idxmap->id_map);
        if (is_map2) {
            static_cast<IndexIDMap2*>(idxmap.get())->construct_rev_map();
        }
        idx = std::move(idxmap);
    } else if (h == fourcc("Ix2L")) {
        auto idxp = std::make_unique<Index2Layer>();
        read_index_header(*idxp, f);
        idxp->q1.quantizer = read_index(f, io_flags);
        READ1(idxp->q1.nlist);
        READ1(idxp->q1.quantizer_trains_alone);
        read_ProductQuantizer(&idxp->pq, f);
        READ1(idxp->code_size_1);
        READ1(idxp->code_size_2);
        READ1(idxp->code_size);
        read_vector(idxp->codes, f);
        idx = std::move(idxp);
    } else if (
            h == fourcc("IHNf") || h == fourcc("IHNp") || h == fourcc("IHNs") ||
            h == fourcc("IHN2") || h == fourcc("IHNc") || h == fourcc("IHc2") ||
            h == fourcc("IHfP")) {
        std::unique_ptr<IndexHNSW> idxhnsw;
        if (h == fourcc("IHNf")) {
            idxhnsw = std::make_unique<IndexHNSWFlat>();
        }
        if (h == fourcc("IHfP")) {
            idxhnsw = std::make_unique<IndexHNSWFlatPanorama>();
        }
        if (h == fourcc("IHNp")) {
            idxhnsw = std::make_unique<IndexHNSWPQ>();
        }
        if (h == fourcc("IHNs")) {
            idxhnsw = std::make_unique<IndexHNSWSQ>();
        }
        if (h == fourcc("IHN2")) {
            idxhnsw = std::make_unique<IndexHNSW2Level>();
        }
        if (h == fourcc("IHNc")) {
            idxhnsw = std::make_unique<IndexHNSWCagra>();
        }
        if (h == fourcc("IHc2")) {
            idxhnsw = std::make_unique<IndexHNSWCagra>();
        }
        read_index_header(*idxhnsw, f);
        if (h == fourcc("IHfP")) {
            auto idx_panorama =
                    dynamic_cast<IndexHNSWFlatPanorama*>(idxhnsw.get());
            size_t nlevels;
            READ1(nlevels);
            const_cast<size_t&>(idx_panorama->num_panorama_levels) = nlevels;
            const_cast<Panorama&>(idx_panorama->pano) =
                    Panorama(idx_panorama->d * sizeof(float), nlevels, 1);
            READVECTOR(idx_panorama->cum_sums);
        }
        if (h == fourcc("IHNc") || h == fourcc("IHc2")) {
            READ1(idxhnsw->keep_max_size_level0);
            auto idx_hnsw_cagra = dynamic_cast<IndexHNSWCagra*>(idxhnsw.get());
            READ1(idx_hnsw_cagra->base_level_only);
            READ1(idx_hnsw_cagra->num_base_level_search_entrypoints);
            if (h == fourcc("IHc2")) {
                READ1(idx_hnsw_cagra->numeric_type_);
            } else { // cagra before numeric_type_ was introduced
                idx_hnsw_cagra->set_numeric_type(faiss::Float32);
            }
        }
        read_HNSW(idxhnsw->hnsw, f);
        idxhnsw->hnsw.is_panorama = (h == fourcc("IHfP"));
        idxhnsw->storage = read_index(f, io_flags);
        idxhnsw->own_fields = idxhnsw->storage != nullptr;
        if (h == fourcc("IHNp") && !(io_flags & IO_FLAG_PQ_SKIP_SDC_TABLE)) {
            dynamic_cast<IndexPQ*>(idxhnsw->storage)->pq.compute_sdc_table();
        }
        idx = std::move(idxhnsw);
    } else if (
            h == fourcc("INSf") || h == fourcc("INSp") || h == fourcc("INSs")) {
        std::unique_ptr<IndexNSG> idxnsg;
        if (h == fourcc("INSf")) {
            idxnsg = std::make_unique<IndexNSGFlat>();
        }
        if (h == fourcc("INSp")) {
            idxnsg = std::make_unique<IndexNSGPQ>();
        }
        if (h == fourcc("INSs")) {
            idxnsg = std::make_unique<IndexNSGSQ>();
        }
        read_index_header(*idxnsg, f);
        READ1(idxnsg->GK);
        READ1(idxnsg->build_type);
        READ1(idxnsg->nndescent_S);
        READ1(idxnsg->nndescent_R);
        READ1(idxnsg->nndescent_L);
        READ1(idxnsg->nndescent_iter);
        read_NSG(idxnsg->nsg, f);
        idxnsg->storage = read_index(f, io_flags);
        idxnsg->own_fields = true;
        idx = std::move(idxnsg);
    } else if (h == fourcc("INNf")) {
        auto idxnnd = std::make_unique<IndexNNDescentFlat>();
        read_index_header(*idxnnd, f);
        read_NNDescent(idxnnd->nndescent, f);
        idxnnd->storage = read_index(f, io_flags);
        idxnnd->own_fields = true;
        idx = std::move(idxnnd);
    } else if (h == fourcc("IPfs")) {
        auto idxpqfs = std::make_unique<IndexPQFastScan>();
        read_index_header(*idxpqfs, f);
        read_ProductQuantizer(&idxpqfs->pq, f);
        READ1(idxpqfs->implem);
        READ1(idxpqfs->bbs);
        READ1(idxpqfs->qbs);
        READ1(idxpqfs->ntotal2);
        READ1(idxpqfs->M2);
        READVECTOR(idxpqfs->codes);

        const auto& pq = idxpqfs->pq;
        idxpqfs->M = pq.M;
        idxpqfs->nbits = pq.nbits;
        idxpqfs->ksub = (1 << pq.nbits);
        idxpqfs->code_size = pq.code_size;

        idx = std::move(idxpqfs);

    } else if (h == fourcc("IwPf")) {
        auto ivpq = std::make_unique<IndexIVFPQFastScan>();
        read_ivf_header(ivpq.get(), f);
        READ1(ivpq->by_residual);
        READ1(ivpq->code_size);
        READ1(ivpq->bbs);
        READ1(ivpq->M2);
        READ1(ivpq->implem);
        READ1(ivpq->qbs2);
        read_ProductQuantizer(&ivpq->pq, f);
        read_InvertedLists(*ivpq, f, io_flags);
        ivpq->precompute_table();

        const auto& pq = ivpq->pq;
        ivpq->M = pq.M;
        ivpq->nbits = pq.nbits;
        ivpq->ksub = (1 << pq.nbits);
        ivpq->code_size = pq.code_size;
        ivpq->init_code_packer();

        idx = std::move(ivpq);
    } else if (h == fourcc("IRMf")) {
        auto imm = std::make_unique<IndexRowwiseMinMax>();
        read_index_header(*imm, f);

        imm->index = read_index(f, io_flags);
        imm->own_fields = true;

        idx = std::move(imm);
    } else if (h == fourcc("IRMh")) {
        auto imm = std::make_unique<IndexRowwiseMinMaxFP16>();
        read_index_header(*imm, f);

        imm->index = read_index(f, io_flags);
        imm->own_fields = true;

        idx = std::move(imm);
    } else if (h == fourcc("Irfs")) {
        auto idxqfs = std::make_unique<IndexRaBitQFastScan>();
        read_index_header(*idxqfs, f);
        read_RaBitQuantizer(idxqfs->rabitq, f, true);
        READVECTOR(idxqfs->center);
        READ1(idxqfs->qb);
        READVECTOR(idxqfs->flat_storage);

        READ1(idxqfs->bbs);
        READ1(idxqfs->ntotal2);
        READ1(idxqfs->M2);
        READ1(idxqfs->code_size);

        const size_t M_fastscan = (idxqfs->d + 3) / 4;
        constexpr size_t nbits_fastscan = 4;
        idxqfs->M = M_fastscan;
        idxqfs->nbits = nbits_fastscan;
        idxqfs->ksub = (1 << nbits_fastscan);

        READVECTOR(idxqfs->codes);
        idx = std::move(idxqfs);
    } else if (h == fourcc("Ixrq")) {
        auto idxq = std::make_unique<IndexRaBitQ>();
        read_index_header(*idxq, f);
        read_RaBitQuantizer(idxq->rabitq, f, false);
        READVECTOR(idxq->codes);
        READVECTOR(idxq->center);
        READ1(idxq->qb);

        // rabitq.nb_bits is already set to 1 by read_RaBitQuantizer
        idxq->code_size = idxq->rabitq.code_size;
        idx = std::move(idxq);
    } else if (h == fourcc("Ixrr")) {
        // Ixrr = multi-bit format (new)
        auto idxq = std::make_unique<IndexRaBitQ>();
        read_index_header(*idxq, f);
        read_RaBitQuantizer(idxq->rabitq, f, true); // Reads nb_bits from file
        READVECTOR(idxq->codes);
        READVECTOR(idxq->center);
        READ1(idxq->qb);

        idxq->code_size = idxq->rabitq.code_size;
        idx = std::move(idxq);
    } else if (h == fourcc("Iwrq")) {
        auto ivrq = std::make_unique<IndexIVFRaBitQ>();
        read_ivf_header(ivrq.get(), f);
        read_RaBitQuantizer(ivrq->rabitq, f, false);
        READ1(ivrq->code_size);
        READ1(ivrq->by_residual);
        READ1(ivrq->qb);

        // rabitq.nb_bits is already set to 1 by read_RaBitQuantizer
        // Update rabitq to match nb_bits
        ivrq->rabitq.code_size =
                ivrq->rabitq.compute_code_size(ivrq->d, ivrq->rabitq.nb_bits);
        ivrq->code_size = ivrq->rabitq.code_size;
        read_InvertedLists(*ivrq, f, io_flags);
        idx = std::move(ivrq);
    } else if (h == fourcc("Iwrr")) {
        // Iwrr = multi-bit format (new)
        auto ivrq = std::make_unique<IndexIVFRaBitQ>();
        read_ivf_header(ivrq.get(), f);
        read_RaBitQuantizer(ivrq->rabitq, f, true); // Reads nb_bits from file
        READ1(ivrq->code_size);
        READ1(ivrq->by_residual);
        READ1(ivrq->qb);

        // Update rabitq to match nb_bits
        ivrq->rabitq.code_size =
                ivrq->rabitq.compute_code_size(ivrq->d, ivrq->rabitq.nb_bits);
        ivrq->code_size = ivrq->rabitq.code_size;
        read_InvertedLists(*ivrq, f, io_flags);
        idx = std::move(ivrq);
    }
#ifdef FAISS_ENABLE_SVS
    else if (
            h == fourcc("ILVQ") || h == fourcc("ISVL") || h == fourcc("ISVD")) {
        std::unique_ptr<IndexSVSVamana> svs;
        if (h == fourcc("ILVQ")) {
            svs = std::make_unique<IndexSVSVamanaLVQ>();
        } else if (h == fourcc("ISVL")) {
            svs = std::make_unique<IndexSVSVamanaLeanVec>();
        } else if (h == fourcc("ISVD")) {
            svs = std::make_unique<IndexSVSVamana>();
        }

        read_index_header(*svs, f);
        READ1(svs->graph_max_degree);
        READ1(svs->alpha);
        READ1(svs->search_window_size);
        READ1(svs->search_buffer_capacity);
        READ1(svs->construction_window_size);
        READ1(svs->max_candidate_pool_size);
        READ1(svs->prune_to);
        READ1(svs->use_full_search_history);
        READ1(svs->storage_kind);
        if (h == fourcc("ISVL")) {
            READ1(dynamic_cast<IndexSVSVamanaLeanVec*>(svs.get())->leanvec_d);
        }

        bool initialized;
        READ1(initialized);
        if (initialized) {
            faiss::svs_io::ReaderStreambuf rbuf(f);
            std::istream is(&rbuf);
            svs->deserialize_impl(is);
        }
        if (h == fourcc("ISVL")) {
            bool trained;
            READ1(trained);
            if (trained) {
                faiss::svs_io::ReaderStreambuf rbuf(f);
                std::istream is(&rbuf);
                dynamic_cast<IndexSVSVamanaLeanVec*>(svs.get())
                        ->deserialize_training_data(is);
            }
        }
        idx = std::move(svs);
    } else if (h == fourcc("ISVF")) {
        auto svs = std::make_unique<IndexSVSFlat>();
        read_index_header(*svs, f);

        bool initialized;
        READ1(initialized);
        if (initialized) {
            faiss::svs_io::ReaderStreambuf rbuf(f);
            std::istream is(&rbuf);
            svs->deserialize_impl(is);
        }
        idx = std::move(svs);
    }
#endif // FAISS_ENABLE_SVS
    else if (h == fourcc("Iwrf")) {
        auto ivrqfs = std::make_unique<IndexIVFRaBitQFastScan>();
        read_ivf_header(ivrqfs.get(), f);
        read_RaBitQuantizer(ivrqfs->rabitq, f);
        READ1(ivrqfs->by_residual);
        READ1(ivrqfs->code_size);
        READ1(ivrqfs->bbs);
        READ1(ivrqfs->qbs2);
        READ1(ivrqfs->M2);
        READ1(ivrqfs->implem);
        READ1(ivrqfs->qb);
        READ1(ivrqfs->centered);
        READVECTOR(ivrqfs->flat_storage);

        // Initialize FastScan base class fields
        const size_t M_fastscan = (ivrqfs->d + 3) / 4;
        constexpr size_t nbits_fastscan = 4;
        ivrqfs->M = M_fastscan;
        ivrqfs->nbits = nbits_fastscan;
        ivrqfs->ksub = (1 << nbits_fastscan);

        read_InvertedLists(*ivrqfs, f, io_flags);
        ivrqfs->init_code_packer();
        idx = std::move(ivrqfs);
    } else {
        FAISS_THROW_FMT(
                "Index type 0x%08x (\"%s\") not recognized",
                h,
                fourcc_inv_printable(h).c_str());
        idx.reset();
    }
    return idx;
}

Index* read_index(IOReader* f, int io_flags) {
    return read_index_up(f, io_flags).release();
}

std::unique_ptr<Index> read_index_up(FILE* f, int io_flags) {
    if ((io_flags & IO_FLAG_MMAP_IFC) == IO_FLAG_MMAP_IFC) {
        // enable mmap-supporting IOReader
        auto owner = std::make_shared<MmappedFileMappingOwner>(f);
        MappedFileIOReader reader(owner);
        return read_index_up(&reader, io_flags);
    } else {
        FileIOReader reader(f);
        return read_index_up(&reader, io_flags);
    }
}

Index* read_index(FILE* f, int io_flags) {
    return read_index_up(f, io_flags).release();
}

std::unique_ptr<Index> read_index_up(const char* fname, int io_flags) {
    if ((io_flags & IO_FLAG_MMAP_IFC) == IO_FLAG_MMAP_IFC) {
        // enable mmap-supporting IOReader
        auto owner = std::make_shared<MmappedFileMappingOwner>(fname);
        MappedFileIOReader reader(owner);
        return read_index_up(&reader, io_flags);
    } else {
        FileIOReader reader(fname);
        return read_index_up(&reader, io_flags);
    }
}

Index* read_index(const char* fname, int io_flags) {
    return read_index_up(fname, io_flags).release();
}

std::unique_ptr<VectorTransform> read_VectorTransform_up(const char* fname) {
    FileIOReader reader(fname);
    return read_VectorTransform_up(&reader);
}

VectorTransform* read_VectorTransform(const char* fname) {
    return read_VectorTransform_up(fname).release();
}

/*************************************************************
 * Read binary indexes
 **************************************************************/

static void read_InvertedLists(IndexBinaryIVF& ivf, IOReader* f, int io_flags) {
    InvertedLists* ils = read_InvertedLists(f, io_flags);
    FAISS_THROW_IF_NOT(
            !ils ||
            (ils->nlist == ivf.nlist && ils->code_size == ivf.code_size));
    ivf.invlists = ils;
    ivf.own_invlists = true;
}

static void read_index_binary_header(IndexBinary& idx, IOReader* f) {
    READ1(idx.d);
    READ1(idx.code_size);
    READ1(idx.ntotal);
    READ1(idx.is_trained);
    READ1(idx.metric_type);
    FAISS_THROW_IF_NOT_FMT(
            idx.d >= 0, "invalid binary index dimension %d", idx.d);
    FAISS_THROW_IF_NOT_FMT(
            idx.ntotal >= 0,
            "invalid binary index ntotal %" PRId64,
            (int64_t)idx.ntotal);
    idx.verbose = false;
}

static void read_binary_ivf_header(
        IndexBinaryIVF& ivf,
        IOReader* f,
        std::vector<std::vector<idx_t>>* ids = nullptr) {
    read_index_binary_header(ivf, f);
    READ1(ivf.nlist);
    READ1(ivf.nprobe);
    ivf.quantizer = read_index_binary(f);
    ivf.own_fields = true;
    if (ids) { // used in legacy "Iv" formats
        ids->resize(ivf.nlist);
        for (size_t i = 0; i < ivf.nlist; i++)
            READVECTOR((*ids)[i]);
    }
    read_direct_map(&ivf.direct_map, f);
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
    size_t bits_per_entry = (size_t)b + (size_t)il_nbit;
    std::vector<uint8_t> buf(
            mul_no_overflow(bits_per_entry, sz, "binary hash invlists"));
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
    size_t nbit = add_no_overflow(
            mul_no_overflow((size_t)(b + id_bits), sz, "multi hash map"),
            mul_no_overflow(ntotal, (size_t)id_bits, "multi hash map"),
            "multi hash map total bits");
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

std::unique_ptr<IndexBinary> read_index_binary_up(IOReader* f, int io_flags) {
    std::unique_ptr<IndexBinary> idx;
    uint32_t h;
    READ1(h);
    if (h == fourcc("IBxF")) {
        auto idxf = std::make_unique<IndexBinaryFlat>();
        read_index_binary_header(*idxf, f);
        read_vector(idxf->xb, f);
        FAISS_THROW_IF_NOT(idxf->xb.size() == idxf->ntotal * idxf->code_size);
        idx = std::move(idxf);
    } else if (h == fourcc("IBwF")) {
        auto ivf = std::make_unique<IndexBinaryIVF>();
        read_binary_ivf_header(*ivf, f);
        read_InvertedLists(*ivf, f, io_flags);
        idx = std::move(ivf);
    } else if (h == fourcc("IBFf")) {
        auto idxff = std::make_unique<IndexBinaryFromFloat>();
        read_index_binary_header(*idxff, f);
        idxff->own_fields = true;
        idxff->index = read_index(f, io_flags);
        idx = std::move(idxff);
    } else if (h == fourcc("IBHf")) {
        auto idxhnsw = std::make_unique<IndexBinaryHNSW>();
        read_index_binary_header(*idxhnsw, f);
        read_HNSW(idxhnsw->hnsw, f);
        idxhnsw->hnsw.is_panorama = false;
        idxhnsw->storage = read_index_binary(f, io_flags);
        idxhnsw->own_fields = true;
        idx = std::move(idxhnsw);
    } else if (h == fourcc("IBHc")) {
        auto idxhnsw = std::make_unique<IndexBinaryHNSWCagra>();
        read_index_binary_header(*idxhnsw, f);
        READ1(idxhnsw->keep_max_size_level0);
        READ1(idxhnsw->base_level_only);
        READ1(idxhnsw->num_base_level_search_entrypoints);
        read_HNSW(idxhnsw->hnsw, f);
        idxhnsw->hnsw.is_panorama = false;
        idxhnsw->storage = read_index_binary(f, io_flags);
        idxhnsw->own_fields = true;
        idx = std::move(idxhnsw);
    } else if (h == fourcc("IBMp") || h == fourcc("IBM2")) {
        bool is_map2 = h == fourcc("IBM2");
        std::unique_ptr<IndexBinaryIDMap> idxmap = is_map2
                ? std::make_unique<IndexBinaryIDMap2>()
                : std::make_unique<IndexBinaryIDMap>();
        read_index_binary_header(*idxmap, f);
        idxmap->index = read_index_binary(f, io_flags);
        idxmap->own_fields = true;
        READVECTOR(idxmap->id_map);
        if (is_map2) {
            static_cast<IndexBinaryIDMap2*>(idxmap.get())->construct_rev_map();
        }
        idx = std::move(idxmap);
    } else if (h == fourcc("IBHh")) {
        auto idxh = std::make_unique<IndexBinaryHash>();
        read_index_binary_header(*idxh, f);
        READ1(idxh->b);
        READ1(idxh->nflip);
        read_binary_hash_invlists(idxh->invlists, idxh->b, f);
        idx = std::move(idxh);
    } else if (h == fourcc("IBHm")) {
        auto idxmh = std::make_unique<IndexBinaryMultiHash>();
        read_index_binary_header(*idxmh, f);
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
        idx = std::move(idxmh);
    } else {
        FAISS_THROW_FMT(
                "Index type %08x (\"%s\") not recognized",
                h,
                fourcc_inv_printable(h).c_str());
    }
    return idx;
}

IndexBinary* read_index_binary(IOReader* f, int io_flags) {
    return read_index_binary_up(f, io_flags).release();
}

std::unique_ptr<IndexBinary> read_index_binary_up(FILE* f, int io_flags) {
    if ((io_flags & IO_FLAG_MMAP_IFC) == IO_FLAG_MMAP_IFC) {
        // enable mmap-supporting IOReader
        auto owner = std::make_shared<MmappedFileMappingOwner>(f);
        MappedFileIOReader reader(owner);
        return read_index_binary_up(&reader, io_flags);
    } else {
        FileIOReader reader(f);
        return read_index_binary_up(&reader, io_flags);
    }
}

IndexBinary* read_index_binary(FILE* f, int io_flags) {
    return read_index_binary_up(f, io_flags).release();
}

std::unique_ptr<IndexBinary> read_index_binary_up(
        const char* fname,
        int io_flags) {
    if ((io_flags & IO_FLAG_MMAP_IFC) == IO_FLAG_MMAP_IFC) {
        // enable mmap-supporting IOReader
        auto owner = std::make_shared<MmappedFileMappingOwner>(fname);
        MappedFileIOReader reader(owner);
        return read_index_binary_up(&reader, io_flags);
    } else {
        FileIOReader reader(fname);
        return read_index_binary_up(&reader, io_flags);
    }
}

IndexBinary* read_index_binary(const char* fname, int io_flags) {
    return read_index_binary_up(fname, io_flags).release();
}

} // namespace faiss
