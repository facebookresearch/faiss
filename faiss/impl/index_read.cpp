/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/index_read_utils.h>
#include <faiss/index_io.h>

#include <faiss/impl/io_macros.h>

#include <cstdio> // For ftell, fseek
#include <cstdlib>
#include <optional>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
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

// mmap-ing and viewing facilities
#include <faiss/impl/maybe_owned_vector.h>

#include <faiss/impl/mapped_io.h>
#include <faiss/impl/zerocopy_io.h>
#include <cinttypes>
namespace faiss {

/*************************************************************
 * Mmap-ing and viewing facilities
 **************************************************************/

// This is a baseline functionality for reading mmapped and zerocopied vector.
// * if `beforeknown_size` is defined, then a size of the vector won't be read.
// * if `size_multiplier` is defined, then a size will be multiplied by it.
// * returns true is the case was handled; ownerwise, false
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
    // size is not known beforehand, nultiply the size 4x
    if (read_vector_base<VectorT>(target, f, std::nullopt, 4)) {
        return;
    }

    // the default case
    READXBVECTOR(target);
}

/*************************************************************
 * Read
 **************************************************************/

void read_index_header(Index* idx, IOReader* f) {
    READ1(idx->d);
    READ1(idx->ntotal);
    idx_t dummy;
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
                read_vector_with_known_size(
                        ails->codes[i], f, n * ails->code_size);
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
        return InvertedListsIOHook::lookup(h2)->read_ArrayInvertedLists(
                f, io_flags, nlist, code_size, sizes);
    } else {
        return InvertedListsIOHook::lookup(h)->read(f, io_flags);
    }
}

void read_InvertedLists(IndexIVF* ivf, IOReader* f, int io_flags) {
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

void read_ProductQuantizer(ProductQuantizer* pq, IOReader* f) {
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
        aq->search_type == AdditiveQuantizer::ST_norm_cqint4 ||
        aq->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
        aq->search_type == AdditiveQuantizer::ST_norm_rq2x4) {
        read_xb_vector(aq->qnorm.codes, f);
        aq->qnorm.ntotal = aq->qnorm.codes.size() / 4;
        aq->qnorm.update_permutation();
    }

    if (aq->search_type == AdditiveQuantizer::ST_norm_lsq2x4 ||
        aq->search_type == AdditiveQuantizer::ST_norm_rq2x4) {
        READVECTOR(aq->norm_tabs);
    }

    aq->set_derived_values();
}

static void read_ResidualQuantizer(
        ResidualQuantizer* rq,
        IOReader* f,
        int io_flags) {
    read_AdditiveQuantizer(rq, f);
    READ1(rq->train_type);
    READ1(rq->max_beam_size);
    if ((rq->train_type & ResidualQuantizer::Skip_codebook_tables) ||
        (io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE)) {
        // don't precompute the tables
    } else {
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

static void read_ProductAdditiveQuantizer(
        ProductAdditiveQuantizer* paq,
        IOReader* f) {
    read_AdditiveQuantizer(paq, f);
    READ1(paq->nsplits);
}

static void read_ProductResidualQuantizer(
        ProductResidualQuantizer* prq,
        IOReader* f,
        int io_flags) {
    read_ProductAdditiveQuantizer(prq, f);

    for (size_t i = 0; i < prq->nsplits; i++) {
        auto rq = new ResidualQuantizer();
        read_ResidualQuantizer(rq, f, io_flags);
        prq->quantizers.push_back(rq);
    }
}

static void read_ProductLocalSearchQuantizer(
        ProductLocalSearchQuantizer* plsq,
        IOReader* f) {
    read_ProductAdditiveQuantizer(plsq, f);

    for (size_t i = 0; i < plsq->nsplits; i++) {
        auto lsq = new LocalSearchQuantizer();
        read_LocalSearchQuantizer(lsq, f);
        plsq->quantizers.push_back(lsq);
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

// Modified version of READ1
#define READ1_AND_COUNT(var, count_var, reader_f) \
    do {                                          \
        READANDCHECK(&(var), 1);                  \
        (count_var) += sizeof(var);               \
    } while (0)

// Modified version of READVECTOR
#define READVECTOR_AND_COUNT(vec, count_var, reader_f)               \
    do {                                                             \
        size_t size;                                                 \
        READANDCHECK(&size, 1);                                      \
        (count_var) += sizeof(size_t);                               \
        FAISS_THROW_IF_NOT(size >= 0 && size < (uint64_t{1} << 40)); \
        (vec).resize(size);                                          \
        size_t bytes_to_read =                                       \
                size * sizeof(typename decltype(vec)::value_type);   \
        READANDCHECK((vec).data(), size);                            \
        (count_var) += bytes_to_read;                                \
    } while (0)

static void read_HNSW(
        HNSW* hnsw,
        IOReader* f,
        const HNSWIndexConfig& config = HNSWIndexConfig()) {
    if (!config.is_compact && config.is_skip_neighbors) {
        throw std::invalid_argument(
                "Skipping neighbors data is not allowed for non-compact HNSW indices.");
    }
    printf("[read_HNSW - CSR NL v4] Reading metadata & CSR indices (manual offset)...\n");
    uint64_t calculated_offset = 0;

    READVECTOR_AND_COUNT(hnsw->assign_probas, calculated_offset, f);
    READVECTOR_AND_COUNT(hnsw->cum_nneighbor_per_level, calculated_offset, f);
    READVECTOR_AND_COUNT(hnsw->levels, calculated_offset, f);

    printf("[read_HNSW NL v4] Read levels vector, size: %zd\n",
           hnsw->levels.size());

    if (config.is_compact) {
        bool compact_flag_read;
        READ1_AND_COUNT(compact_flag_read, calculated_offset, f);
        FAISS_THROW_IF_NOT_MSG(
                compact_flag_read == true, "Expected CSR format flag in file.");
        hnsw->storage_is_compact = compact_flag_read;

        printf("[read_HNSW NL v4] Reading Compact Storage format indices...\n");

        READVECTOR_AND_COUNT(hnsw->compact_level_ptr, calculated_offset, f);
        printf("[read_HNSW NL v4] Read compact_level_ptr, size: %zd\n",
               hnsw->compact_level_ptr.size());
        READVECTOR_AND_COUNT(hnsw->compact_node_offsets, calculated_offset, f);
        printf("[read_HNSW NL v4] Read compact_node_offsets, size: %zd\n",
               hnsw->compact_node_offsets.size());
        FAISS_THROW_IF_NOT(
                hnsw->compact_node_offsets.size() == hnsw->levels.size() + 1);
    } else {
        printf("[READ_HNSW] Reading Original Storage format...\n");

        // --- BEGIN INSERTED CODE for handling potential extra byte ---
        printf("[READ_HNSW] Probing for potential extra byte before non-compact offsets...\n");

        // Get reader type for potential rewind
        FileIOReader* file_reader_nc = dynamic_cast<FileIOReader*>(f);
        MappedFileIOReader* mmap_reader_nc =
                dynamic_cast<MappedFileIOReader*>(f);

        off_t pos_before_probe = -1;
        if (file_reader_nc) {
            pos_before_probe = ftell(file_reader_nc->f);
            if (pos_before_probe == -1) {
                int RTERRNO = errno;
                FAISS_THROW_FMT(
                        "ftell failed before extra byte probe: %s",
                        strerror(RTERRNO));
            }
        } else if (mmap_reader_nc) {
            pos_before_probe = (off_t)mmap_reader_nc->pos;
        }
        // else: maybe print warning or throw for unknown reader if rewind is
        // needed

        uint8_t suspected_flag;
        size_t read_count =
                (*f)(&suspected_flag, 1, 1); // Tentatively read 1 byte

        bool is_back_to_original_position = true;
        if (read_count == 1) {
            // Successfully read one byte
            if (suspected_flag == 0x00) {
                // It's exactly 0x00. Assume it's the unexpected 'false' flag.
                // Consume it.
                printf("[READ_HNSW] Found and consumed an unexpected 0x00 byte.\n");
                is_back_to_original_position = false;
            } else if (suspected_flag == 0x01) {
                printf("is_compact is false, but we read 0x01\n");
                assert(false);
            }
        } else {
            // Failed to read (e.g., EOF). Cannot consume or rewind.
            printf("[READ_HNSW] Warning: Failed to probe for extra byte (read_count=%zu).\n",
                   read_count);
            // Proceed assuming no extra byte, rewind is not needed/possible.
        }

        // If the byte read was not the specific 0x00 we decided to consume,
        // rewind.
        if (read_count == 1 && is_back_to_original_position) {
            FAISS_ASSERT_MSG(
                    pos_before_probe != -1,
                    "Cannot rewind reader, initial position unknown for rewind.");
            if (file_reader_nc) {
                if (fseek(file_reader_nc->f, pos_before_probe, SEEK_SET) != 0) {
                    int RTERRNO = errno;
                    FAISS_THROW_FMT(
                            "fseek failed to rewind to original position. errno=%d (%s)",
                            RTERRNO,
                            strerror(RTERRNO));
                }
                printf("[READ_HNSW] Rewound to original position for FileIOReader.\n");
            } else if (mmap_reader_nc) {
                mmap_reader_nc->pos = pos_before_probe;
                printf("[READ_HNSW] Reset MappedFileIOReader pos to original position.\n");
            } else {
                FAISS_THROW_MSG("Cannot rewind unknown reader type.");
            }
        }
        // --- END INSERTED CODE ---

        // --- Read Original Storage data ---
        READVECTOR(hnsw->offsets);
        // Use the specific read_vector function for MaybeOwnedVector
        read_vector(hnsw->neighbors, f);
        printf("[READ_HNSW] Original Storage sizes: offsets=%zd, neighbors=%zd\n",
               hnsw->offsets.size(),
               hnsw->neighbors.size());
    }

    READ1_AND_COUNT(hnsw->entry_point, calculated_offset, f);
    READ1_AND_COUNT(hnsw->max_level, calculated_offset, f);
    READ1_AND_COUNT(hnsw->efConstruction, calculated_offset, f);
    READ1_AND_COUNT(hnsw->efSearch, calculated_offset, f);

    hnsw->max_level = 0;

    READ1_DUMMY(int)
    calculated_offset += sizeof(int);

    printf("[read_HNSW NL v4] Read entry_point: %ld, max_level: %d\n",
           (long)hnsw->entry_point,
           hnsw->max_level);

    if (config.is_compact) {
        uint32_t storage_fourcc;
        READ1_AND_COUNT(storage_fourcc, calculated_offset, f);
        printf("[read_HNSW NL v4] Read storage fourcc: 0x%x\n", storage_fourcc);

        // Attempt to determine the reader type
        FileIOReader* file_reader = dynamic_cast<FileIOReader*>(f);
        MappedFileIOReader* mmap_reader = dynamic_cast<MappedFileIOReader*>(f);

        // Initialize offset before obtaining it
        hnsw->neighbors_start_offset = -1; // Default to invalid

        // Get the current position *after* reading storage_fourcc.
        // This position points to the start of the 8-byte size field for the
        // neighbors vector.
        if (file_reader != nullptr) {
            long current_pos = ftell(file_reader->f);
            if (current_pos == -1) {
                int RTERRNO = errno;
                FAISS_THROW_FMT(
                        "ftell failed after reading storage_fourcc: %s",
                        strerror(RTERRNO));
            }
            hnsw->neighbors_start_offset = (off_t)current_pos;
            printf("[read_HNSW NL v4 FIX] Detected FileIOReader. Neighbors size field offset: %ld\n",
                   (long)hnsw->neighbors_start_offset);
        } else if (mmap_reader != nullptr) {
            // For MappedFileIOReader, 'pos' is the relevant offset within the
            // map
            hnsw->neighbors_start_offset =
                    (off_t)mmap_reader->pos; // pos is size_t, cast to off_t
            printf("[read_HNSW NL v4 FIX] Detected MappedFileIOReader. Neighbors size field relative offset: %zu\n",
                   mmap_reader->pos);
        } else {
            // Handle unexpected reader types if necessary, or assume only these
            // two
            printf("[read_HNSW NL v4 FIX] Warning: Unknown IOReader type. Cannot reliably determine neighbor offset.\n");
            // Optionally: FAISS_THROW_MSG("Unsupported IOReader for HNSW
            // neighbor offset detection");
        }

        // Ensure the offset was successfully obtained if using file reader
        if (file_reader != nullptr) {
            FAISS_THROW_IF_NOT_FMT(
                    hnsw->neighbors_start_offset >= 0,
                    "Failed to obtain valid file offset (ftell result: %ld)",
                    (long)hnsw->neighbors_start_offset);
        }

        if (config.is_skip_neighbors) {
            printf("[read_HNSW NL v4] Skipping neighbors data.\n");

            // Determine the type of reader and handle accordingly
            FileIOReader* file_reader = dynamic_cast<FileIOReader*>(f);
            MappedFileIOReader* mmap_reader =
                    dynamic_cast<MappedFileIOReader*>(f);

            // Set flag that neighbors should be read on-demand
            hnsw->neighbors_on_disk = true;

            if (file_reader) {
                // --- FileIOReader case: use file offset + pread ---
                printf("[read_HNSW NL v4] Using FileIOReader, will read on demand with pread.\n");

                // Set flag that we're using pread (not mmap)
                hnsw->neighbors_use_mmap = false;

                // Read the size field (8 bytes) to skip past it
                size_t neighbors_size_field;
                if ((*f)(&neighbors_size_field, sizeof(size_t), 1) != 1) {
                    FAISS_THROW_MSG(
                            "Failed to read neighbors size field while skipping");
                }

                // Calculate bytes to skip
                size_t neighbors_bytes =
                        neighbors_size_field * sizeof(HNSW::storage_idx_t);

                // Skip past the neighbor data
                if (fseek(file_reader->f, neighbors_bytes, SEEK_CUR) != 0) {
                    int RTERRNO = errno;
                    FAISS_THROW_FMT(
                            "fseek failed to skip %zu bytes of neighbor data: %s",
                            neighbors_bytes,
                            strerror(RTERRNO));
                }

                printf("[read_HNSW NL v4] Skipped %zu bytes of neighbor data.\n",
                       neighbors_bytes);

            } else if (mmap_reader) {
                // --- MappedFileIOReader case: use memory pointer ---
                printf("[read_HNSW NL v4] Using MappedFileIOReader, will access via memory pointer.\n");

                // Set flag that we're using mmap
                hnsw->neighbors_use_mmap = true;

                // Record current position (pointing to size field) for
                // debugging
                hnsw->neighbors_start_offset = mmap_reader->pos;

                // Read the size field (8 bytes) to skip past it
                size_t neighbors_size_field;
                if ((*f)(&neighbors_size_field, sizeof(size_t), 1) != 1) {
                    FAISS_THROW_MSG(
                            "Failed to read neighbors size field while skipping (mmap)");
                }

                // Store the memory pointer where neighbors data starts (right
                // after size field) At this point, pos is positioned at the
                // start of actual data
                hnsw->neighbors_mmap_ptr =
                        (HNSW::storage_idx_t*)((char*)mmap_reader->mmap_owner
                                                       ->data() +
                                               mmap_reader->pos);

                printf("[read_HNSW NL v4] Neighbor data starts at mmap offset (relative): %zu, pointer: %p\n",
                       mmap_reader->pos,
                       (void*)hnsw->neighbors_mmap_ptr);

                // Calculate bytes to skip and the end position
                size_t neighbors_bytes =
                        neighbors_size_field * sizeof(HNSW::storage_idx_t);
                size_t end_pos = mmap_reader->pos + neighbors_bytes;

                // Check boundary
                FAISS_THROW_IF_NOT_FMT(
                        end_pos <= mmap_reader->mmap_owner->size(),
                        "Attempt to skip past mmap region (current pos %zu, skip %zu, total size %zu)",
                        mmap_reader->pos,
                        neighbors_bytes,
                        mmap_reader->mmap_owner->size());

                // Advance mmap reader's position to skip data
                mmap_reader->pos = end_pos;

                printf("[read_HNSW NL v4] Advanced mmap reader pos by %zu bytes to %zu.\n",
                       neighbors_bytes,
                       mmap_reader->pos);

            } else {
                // Unsupported reader type
                FAISS_THROW_MSG(
                        "Skipping neighbors requires FileIOReader or MappedFileIOReader");
            }

        } else {
            printf("[read_HNSW NL v4] Reading neighbors data into memory.\n");

            // Read neighbors into memory as before
            hnsw->neighbors_on_disk = false;
            hnsw->neighbors_use_mmap = false;
            READVECTOR_AND_COUNT(
                    hnsw->compact_neighbors_data, calculated_offset, f);
            printf("[read_HNSW NL v4] Read neighbors data, size: %zd\n",
                   hnsw->compact_neighbors_data.size());
        }
    }
    printf("[read_HNSW NL v4] Finished reading metadata and CSR indices.\n");
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

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < R + 1; j++) {
            int id;
            READ1(id);
            if (id != EMPTY_ID) {
                graph->at(i, j) = id;
            } else {
                break;
            }
        }
    }
}

static void read_NNDescent(NNDescent* nnd, IOReader* f) {
    READ1(nnd->ntotal);
    READ1(nnd->d);
    READ1(nnd->K);
    READ1(nnd->S);
    READ1(nnd->R);
    READ1(nnd->L);
    READ1(nnd->iter);
    READ1(nnd->search_L);
    READ1(nnd->random_seed);
    READ1(nnd->has_built);

    READVECTOR(nnd->final_graph);
}

ProductQuantizer* read_ProductQuantizer(const char* fname) {
    FileIOReader reader(fname);
    return read_ProductQuantizer(&reader);
}

ProductQuantizer* read_ProductQuantizer(IOReader* reader) {
    ProductQuantizer* pq = new ProductQuantizer();
    std::unique_ptr<ProductQuantizer> del(pq);

    read_ProductQuantizer(pq, reader);
    del.release();
    return pq;
}

static void read_RaBitQuantizer(RaBitQuantizer* rabitq, IOReader* f) {
    // don't care about rabitq->centroid
    READ1(rabitq->d);
    READ1(rabitq->code_size);
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
ArrayInvertedLists* set_array_invlist(
        IndexIVF* ivf,
        std::vector<std::vector<idx_t>>& ids) {
    ArrayInvertedLists* ail =
            new ArrayInvertedLists(ivf->nlist, ivf->code_size);

    ail->ids.resize(ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        ail->ids[i] = MaybeOwnedVector<idx_t>(std::move(ids[i]));
    }

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

    std::vector<std::vector<idx_t>> ids;
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

Index* read_index(
        IOReader* f,
        int io_flags,
        const HNSWIndexConfig& hnsw_config) {
    Index* idx = nullptr;
    uint32_t h;
    READ1(h);
    if (h == fourcc("null")) {
        // denotes a missing index, useful for some cases
        return nullptr;
    } else if (
            h == fourcc("IxFI") || h == fourcc("IxF2") || h == fourcc("IxFl")) {
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
        read_xb_vector(idxf->codes, f);
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
        read_vector(idxl->codes, f);
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
        idx = idxp;
    } else if (h == fourcc("IxRQ") || h == fourcc("IxRq")) {
        IndexResidualQuantizer* idxr = new IndexResidualQuantizer();
        read_index_header(idxr, f);
        if (h == fourcc("IxRQ")) {
            read_ResidualQuantizer_old(&idxr->rq, f);
        } else {
            read_ResidualQuantizer(&idxr->rq, f, io_flags);
        }
        READ1(idxr->code_size);
        read_vector(idxr->codes, f);
        idx = idxr;
    } else if (h == fourcc("IxLS")) {
        auto idxr = new IndexLocalSearchQuantizer();
        read_index_header(idxr, f);
        read_LocalSearchQuantizer(&idxr->lsq, f);
        READ1(idxr->code_size);
        read_vector(idxr->codes, f);
        idx = idxr;
    } else if (h == fourcc("IxPR")) {
        auto idxpr = new IndexProductResidualQuantizer();
        read_index_header(idxpr, f);
        read_ProductResidualQuantizer(&idxpr->prq, f, io_flags);
        READ1(idxpr->code_size);
        read_vector(idxpr->codes, f);
        idx = idxpr;
    } else if (h == fourcc("IxPL")) {
        auto idxpl = new IndexProductLocalSearchQuantizer();
        read_index_header(idxpl, f);
        read_ProductLocalSearchQuantizer(&idxpl->plsq, f);
        READ1(idxpl->code_size);
        read_vector(idxpl->codes, f);
        idx = idxpl;
    } else if (h == fourcc("ImRQ")) {
        ResidualCoarseQuantizer* idxr = new ResidualCoarseQuantizer();
        read_index_header(idxr, f);
        read_ResidualQuantizer(&idxr->rq, f, io_flags);
        READ1(idxr->beam_factor);
        if (io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE) {
            // then we force the beam factor to -1
            // which skips the table precomputation.
            idxr->beam_factor = -1;
        }
        idxr->set_beam_factor(idxr->beam_factor);
        idx = idxr;
    } else if (
            h == fourcc("ILfs") || h == fourcc("IRfs") || h == fourcc("IPRf") ||
            h == fourcc("IPLf")) {
        bool is_LSQ = h == fourcc("ILfs");
        bool is_RQ = h == fourcc("IRfs");
        bool is_PLSQ = h == fourcc("IPLf");

        IndexAdditiveQuantizerFastScan* idxaqfs;
        if (is_LSQ) {
            idxaqfs = new IndexLocalSearchQuantizerFastScan();
        } else if (is_RQ) {
            idxaqfs = new IndexResidualQuantizerFastScan();
        } else if (is_PLSQ) {
            idxaqfs = new IndexProductLocalSearchQuantizerFastScan();
        } else {
            idxaqfs = new IndexProductResidualQuantizerFastScan();
        }
        read_index_header(idxaqfs, f);

        if (is_LSQ) {
            read_LocalSearchQuantizer((LocalSearchQuantizer*)idxaqfs->aq, f);
        } else if (is_RQ) {
            read_ResidualQuantizer(
                    (ResidualQuantizer*)idxaqfs->aq, f, io_flags);
        } else if (is_PLSQ) {
            read_ProductLocalSearchQuantizer(
                    (ProductLocalSearchQuantizer*)idxaqfs->aq, f);
        } else {
            read_ProductResidualQuantizer(
                    (ProductResidualQuantizer*)idxaqfs->aq, f, io_flags);
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
        idx = idxaqfs;
    } else if (
            h == fourcc("IVLf") || h == fourcc("IVRf") || h == fourcc("NPLf") ||
            h == fourcc("NPRf")) {
        bool is_LSQ = h == fourcc("IVLf");
        bool is_RQ = h == fourcc("IVRf");
        bool is_PLSQ = h == fourcc("NPLf");

        IndexIVFAdditiveQuantizerFastScan* ivaqfs;
        if (is_LSQ) {
            ivaqfs = new IndexIVFLocalSearchQuantizerFastScan();
        } else if (is_RQ) {
            ivaqfs = new IndexIVFResidualQuantizerFastScan();
        } else if (is_PLSQ) {
            ivaqfs = new IndexIVFProductLocalSearchQuantizerFastScan();
        } else {
            ivaqfs = new IndexIVFProductResidualQuantizerFastScan();
        }
        read_ivf_header(ivaqfs, f);

        if (is_LSQ) {
            read_LocalSearchQuantizer((LocalSearchQuantizer*)ivaqfs->aq, f);
        } else if (is_RQ) {
            read_ResidualQuantizer((ResidualQuantizer*)ivaqfs->aq, f, io_flags);
        } else if (is_PLSQ) {
            read_ProductLocalSearchQuantizer(
                    (ProductLocalSearchQuantizer*)ivaqfs->aq, f);
        } else {
            read_ProductResidualQuantizer(
                    (ProductResidualQuantizer*)ivaqfs->aq, f, io_flags);
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

        read_InvertedLists(ivaqfs, f, io_flags);
        ivaqfs->init_code_packer();
        idx = ivaqfs;
    } else if (h == fourcc("IvFl") || h == fourcc("IvFL")) { // legacy
        IndexIVFFlat* ivfl = new IndexIVFFlat();
        std::vector<std::vector<idx_t>> ids;
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
            std::vector<idx_t> tab;
            READVECTOR(tab);
            for (long i = 0; i < tab.size(); i += 2) {
                std::pair<idx_t, idx_t> pair(tab[i], tab[i + 1]);
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
        read_vector(idxs->codes, f);
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
        std::vector<std::vector<idx_t>> ids;
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
    } else if (
            h == fourcc("IwLS") || h == fourcc("IwRQ") || h == fourcc("IwPL") ||
            h == fourcc("IwPR")) {
        bool is_LSQ = h == fourcc("IwLS");
        bool is_RQ = h == fourcc("IwRQ");
        bool is_PLSQ = h == fourcc("IwPL");
        IndexIVFAdditiveQuantizer* iva;
        if (is_LSQ) {
            iva = new IndexIVFLocalSearchQuantizer();
        } else if (is_RQ) {
            iva = new IndexIVFResidualQuantizer();
        } else if (is_PLSQ) {
            iva = new IndexIVFProductLocalSearchQuantizer();
        } else {
            iva = new IndexIVFProductResidualQuantizer();
        }
        read_ivf_header(iva, f);
        READ1(iva->code_size);
        if (is_LSQ) {
            read_LocalSearchQuantizer((LocalSearchQuantizer*)iva->aq, f);
        } else if (is_RQ) {
            read_ResidualQuantizer((ResidualQuantizer*)iva->aq, f, io_flags);
        } else if (is_PLSQ) {
            read_ProductLocalSearchQuantizer(
                    (ProductLocalSearchQuantizer*)iva->aq, f);
        } else {
            read_ProductResidualQuantizer(
                    (ProductResidualQuantizer*)iva->aq, f, io_flags);
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
    } else if (h == fourcc("IwIQ")) {
        auto* indep = new IndexIVFIndependentQuantizer();
        indep->own_fields = true;
        read_index_header(indep, f);
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
        idx = indep;
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
        read_vector(idxp->codes, f);
        idx = idxp;
    } else if (
            h == fourcc("IHNf") || h == fourcc("IHNp") || h == fourcc("IHNs") ||
            h == fourcc("IHN2") || h == fourcc("IHNc")) {
        IndexHNSW* idxhnsw = nullptr;
        if (h == fourcc("IHNf"))
            idxhnsw = new IndexHNSWFlat();
        if (h == fourcc("IHNp"))
            idxhnsw = new IndexHNSWPQ();
        if (h == fourcc("IHNs"))
            idxhnsw = new IndexHNSWSQ();
        if (h == fourcc("IHN2"))
            idxhnsw = new IndexHNSW2Level();
        if (h == fourcc("IHNc"))
            idxhnsw = new IndexHNSWCagra();
        read_index_header(idxhnsw, f);
        if (h == fourcc("IHNc")) {
            READ1(idxhnsw->keep_max_size_level0);
            auto idx_hnsw_cagra = dynamic_cast<IndexHNSWCagra*>(idxhnsw);
            READ1(idx_hnsw_cagra->base_level_only);
            READ1(idx_hnsw_cagra->num_base_level_search_entrypoints);
        }
        read_HNSW(&idxhnsw->hnsw, f, hnsw_config);

        if (hnsw_config.is_recompute) {
            printf("INFO: Skipping external storage loading, since is_recompute is true.\n");
            idxhnsw->is_recompute = true;
        } else if (hnsw_config.external_storage_path != nullptr) {
            // Load storage from the external file
            printf("INFO: Loading external storage from: %s\n",
                   hnsw_config.external_storage_path);
            // Decide reader type based on io_flags if desired (e.g.,
            // mmap)
            std::unique_ptr<IOReader> storage_reader;
            if ((io_flags & IO_FLAG_MMAP_IFC) == IO_FLAG_MMAP_IFC) {
                auto owner = std::make_shared<MmappedFileMappingOwner>(
                        hnsw_config.external_storage_path);
                storage_reader = std::make_unique<MappedFileIOReader>(owner);
                printf("INFO: Using MappedFileIOReader for external storage.\n");
            } else {
                storage_reader = std::make_unique<FileIOReader>(
                        hnsw_config.external_storage_path);
                printf("INFO: Using FileIOReader for external storage.\n");
            }

            // Recursively call read_index for the storage part, passing
            // flags Pass nullptr for external_storage_path in recursive
            // call
            idxhnsw->storage = read_index(storage_reader.get(), io_flags);

            if (!idxhnsw->storage) {
                FAISS_THROW_FMT(
                        "Failed to read external storage index from %s",
                        hnsw_config.external_storage_path);
            }
            printf("INFO: Successfully loaded external storage.\n");
            // IMPORTANT: We DO NOT read storage from the primary reader 'f'
            // anymore. 'f' remains positioned after the HNSW graph data.

        } else if (io_flags & IO_FLAG_SKIP_STORAGE) {
            // Original logic for skipping storage from primary reader
            uint32_t null_check;
            READ1(null_check);
            FAISS_THROW_IF_NOT_MSG(
                    null_check == fourcc("null"),
                    "IO_FLAG_SKIP_STORAGE set, but storage field was not null");
            idxhnsw->storage = nullptr; // Explicitly set to null
        } else {
            // Original logic: Read storage sequentially from primary reader 'f'
            // Pass nullptr for external_storage_path in recursive call
            idxhnsw->storage = read_index(f, io_flags);
        }

        idxhnsw->own_fields = idxhnsw->storage != nullptr;
        if (h == fourcc("IHNp") && !(io_flags & IO_FLAG_PQ_SKIP_SDC_TABLE)) {
            dynamic_cast<IndexPQ*>(idxhnsw->storage)->pq.compute_sdc_table();
        }
        idx = idxhnsw;
    } else if (
            h == fourcc("INSf") || h == fourcc("INSp") || h == fourcc("INSs")) {
        IndexNSG* idxnsg;
        if (h == fourcc("INSf"))
            idxnsg = new IndexNSGFlat();
        if (h == fourcc("INSp"))
            idxnsg = new IndexNSGPQ();
        if (h == fourcc("INSs"))
            idxnsg = new IndexNSGSQ();
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
    } else if (h == fourcc("INNf")) {
        IndexNNDescent* idxnnd = new IndexNNDescentFlat();
        read_index_header(idxnnd, f);
        read_NNDescent(&idxnnd->nndescent, f);
        idxnnd->storage = read_index(f, io_flags);
        idxnnd->own_fields = true;
        idx = idxnnd;
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

        const auto& pq = idxpqfs->pq;
        idxpqfs->M = pq.M;
        idxpqfs->nbits = pq.nbits;
        idxpqfs->ksub = (1 << pq.nbits);
        idxpqfs->code_size = pq.code_size;

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

        const auto& pq = ivpq->pq;
        ivpq->M = pq.M;
        ivpq->nbits = pq.nbits;
        ivpq->ksub = (1 << pq.nbits);
        ivpq->code_size = pq.code_size;
        ivpq->init_code_packer();

        idx = ivpq;
    } else if (h == fourcc("IRMf")) {
        IndexRowwiseMinMax* imm = new IndexRowwiseMinMax();
        read_index_header(imm, f);

        imm->index = read_index(f, io_flags);
        imm->own_fields = true;

        idx = imm;
    } else if (h == fourcc("IRMh")) {
        IndexRowwiseMinMaxFP16* imm = new IndexRowwiseMinMaxFP16();
        read_index_header(imm, f);

        imm->index = read_index(f, io_flags);
        imm->own_fields = true;

        idx = imm;
    } else if (h == fourcc("Ixrq")) {
        IndexRaBitQ* idxq = new IndexRaBitQ();
        read_index_header(idxq, f);
        read_RaBitQuantizer(&idxq->rabitq, f);
        READVECTOR(idxq->codes);
        READVECTOR(idxq->center);
        READ1(idxq->qb);
        idxq->code_size = idxq->rabitq.code_size;
        idx = idxq;
    } else if (h == fourcc("Iwrq")) {
        IndexIVFRaBitQ* ivrq = new IndexIVFRaBitQ();
        read_ivf_header(ivrq, f);
        read_RaBitQuantizer(&ivrq->rabitq, f);
        READ1(ivrq->code_size);
        READ1(ivrq->by_residual);
        READ1(ivrq->qb);
        read_InvertedLists(ivrq, f, io_flags);
        idx = ivrq;
    } else {
        FAISS_THROW_FMT(
                "Index type 0x%08x (\"%s\") not recognized",
                h,
                fourcc_inv_printable(h).c_str());
        idx = nullptr;
    }
    return idx;
}

Index* read_index(FILE* f, int io_flags, const HNSWIndexConfig& hnsw_config) {
    if ((io_flags & IO_FLAG_MMAP_IFC) == IO_FLAG_MMAP_IFC) {
        // enable mmap-supporting IOReader
        auto owner = std::make_shared<MmappedFileMappingOwner>(f);
        MappedFileIOReader reader(owner);
        return read_index(&reader, io_flags, hnsw_config);
    } else {
        FileIOReader reader(f);
        return read_index(&reader, io_flags, hnsw_config);
    }
}

Index* read_index(
        const char* fname,
        int io_flags,
        const HNSWIndexConfig& hnsw_config) {
    if ((io_flags & IO_FLAG_MMAP_IFC) == IO_FLAG_MMAP_IFC) {
        // enable mmap-supporting IOReader
        auto owner = std::make_shared<MmappedFileMappingOwner>(fname);
        MappedFileIOReader reader(owner);
        return read_index(&reader, io_flags, hnsw_config);
    } else {
        FileIOReader reader(fname);
        Index* idx = read_index(&reader, io_flags, hnsw_config);
        return idx;
    }
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
        std::vector<std::vector<idx_t>>* ids = nullptr) {
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
        read_vector(idxf->xb, f);
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
    if ((io_flags & IO_FLAG_MMAP_IFC) == IO_FLAG_MMAP_IFC) {
        // enable mmap-supporting IOReader
        auto owner = std::make_shared<MmappedFileMappingOwner>(f);
        MappedFileIOReader reader(owner);
        return read_index_binary(&reader, io_flags);
    } else {
        FileIOReader reader(f);
        return read_index_binary(&reader, io_flags);
    }
}

IndexBinary* read_index_binary(const char* fname, int io_flags) {
    if ((io_flags & IO_FLAG_MMAP_IFC) == IO_FLAG_MMAP_IFC) {
        // enable mmap-supporting IOReader
        auto owner = std::make_shared<MmappedFileMappingOwner>(fname);
        MappedFileIOReader reader(owner);
        return read_index_binary(&reader, io_flags);
    } else {
        FileIOReader reader(fname);
        IndexBinary* idx = read_index_binary(&reader, io_flags);
        return idx;
    }
}

} // namespace faiss
