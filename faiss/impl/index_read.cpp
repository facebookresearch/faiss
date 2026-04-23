/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/index_read_utils.h>
#include <faiss/index_io.h>

#include <faiss/impl/io_macros.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/io.h>
#include <faiss/utils/hamming.h>

#include <faiss/invlists/InvertedListsIOHook.h>

#include <faiss/invlists/BlockInvertedLists.h>

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

namespace {
size_t deserialization_loop_limit_ = 0;
size_t deserialization_vector_byte_limit_ = uint64_t{1} << 40; // 1 TB
} // namespace

size_t get_deserialization_loop_limit() {
    return deserialization_loop_limit_;
}

void set_deserialization_loop_limit(size_t value) {
    deserialization_loop_limit_ = value;
}

size_t get_deserialization_vector_byte_limit() {
    return deserialization_vector_byte_limit_;
}

void set_deserialization_vector_byte_limit(size_t value) {
    deserialization_vector_byte_limit_ = value;
}

#define FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(val, field_name) \
    do {                                                        \
        auto limit_ = get_deserialization_loop_limit();         \
        if (limit_ > 0) {                                       \
            FAISS_THROW_IF_NOT_FMT(                             \
                    static_cast<size_t>(val) <= limit_,         \
                    "%s=%zd exceeds deserialization_loop_limit" \
                    " of %zd",                                  \
                    field_name,                                 \
                    static_cast<size_t>(val),                   \
                    limit_);                                    \
        }                                                       \
    } while (0)

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
    int metric_type_int;
    READ1(metric_type_int);
    idx.metric_type = metric_type_from_int(metric_type_int);
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
        FAISS_THROW_IF_NOT(
                lt->A.size() >= size_t(lt->d_in) * size_t(lt->d_out));
        FAISS_THROW_IF_NOT(!lt->have_bias || lt->b.size() >= size_t(lt->d_out));
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
    } else if (h == fourcc("HRot")) {
        auto hr = std::make_unique<HadamardRotation>();
        READ1(hr->seed);
        vt = std::move(hr);
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
    FAISS_THROW_IF_NOT_FMT(
            vt->d_in >= 0,
            "invalid VectorTransform d_in=%d (must be >= 0)",
            vt->d_in);
    FAISS_THROW_IF_NOT_FMT(
            vt->d_out >= 0,
            "invalid VectorTransform d_out=%d (must be >= 0)",
            vt->d_out);
    {
        size_t dim_product = mul_no_overflow(
                vt->d_in, vt->d_out, "VectorTransform d_in * d_out");
        FAISS_THROW_IF_NOT_MSG(
                dim_product <=
                        get_deserialization_vector_byte_limit() / sizeof(float),
                "VectorTransform d_in * d_out would exceed "
                "deserialization vector byte limit");
    }
    if (h == fourcc("HRot")) {
        FAISS_THROW_IF_NOT_FMT(
                vt->d_out > 0 && (vt->d_out & (vt->d_out - 1)) == 0,
                "invalid HadamardRotation d_out=%d (must be a power of 2 > 0)",
                vt->d_out);
        FAISS_THROW_IF_NOT_FMT(
                vt->d_out >= vt->d_in,
                "invalid HadamardRotation d_out=%d < d_in=%d",
                vt->d_out,
                vt->d_in);
        FAISS_THROW_IF_NOT_FMT(
                static_cast<size_t>(vt->d_out) <=
                        get_deserialization_vector_byte_limit() /
                                (3 * sizeof(float)),
                "HadamardRotation d_out=%d would exceed deserialization byte limit",
                vt->d_out);
        auto* hr = dynamic_cast<HadamardRotation*>(vt.get());
        FAISS_THROW_IF_NOT_MSG(hr, "dynamic_cast to HadamardRotation failed");
        FAISS_THROW_IF_NOT_FMT(
                vt->d_in > 0,
                "invalid HadamardRotation d_in=%d (must be > 0)",
                vt->d_in);
        size_t p = 1;
        while (p < static_cast<size_t>(vt->d_in)) {
            p <<= 1;
        }
        FAISS_THROW_IF_NOT_FMT(
                static_cast<size_t>(vt->d_out) == p,
                "invalid HadamardRotation d_out %d for d_in %d"
                " (d_out must be the smallest power of 2 >= d_in)",
                vt->d_out,
                vt->d_in);
        size_t byte_limit = get_deserialization_vector_byte_limit();
        FAISS_THROW_IF_NOT_MSG(
                p <= byte_limit / (3 * sizeof(float)),
                "HadamardRotation d_out exceeds deserialization byte limit");
        hr->init(hr->seed);
    }
    if (h == fourcc("RmDT")) {
        auto* rdt = dynamic_cast<RemapDimensionsTransform*>(vt.get());
        FAISS_THROW_IF_NOT_MSG(
                rdt, "dynamic_cast to RemapDimensionsTransform failed");
        FAISS_THROW_IF_NOT_FMT(
                static_cast<int>(rdt->map.size()) >= rdt->d_out,
                "RemapDimensionsTransform map size %d < d_out %d",
                (int)rdt->map.size(),
                rdt->d_out);
    }
    if (h == fourcc("VNrm")) {
        FAISS_THROW_IF_NOT_FMT(
                vt->d_in == vt->d_out,
                "NormalizationTransform requires d_in == d_out, "
                "got d_in=%d d_out=%d",
                vt->d_in,
                vt->d_out);
    }
    if (h == fourcc("VCnt")) {
        auto* ct = dynamic_cast<CenteringTransform*>(vt.get());
        FAISS_THROW_IF_NOT_MSG(ct, "dynamic_cast to CenteringTransform failed");
        FAISS_THROW_IF_NOT_FMT(
                static_cast<int>(ct->mean.size()) >= ct->d_in,
                "CenteringTransform mean size %d < d_in %d",
                (int)ct->mean.size(),
                ct->d_in);
        FAISS_THROW_IF_NOT_FMT(
                vt->d_in == vt->d_out,
                "CenteringTransform requires d_in == d_out, "
                "got d_in=%d d_out=%d",
                vt->d_in,
                vt->d_out);
    }
    if (h == fourcc("Viqt")) {
        auto* itqt = dynamic_cast<ITQTransform*>(vt.get());
        FAISS_THROW_IF_NOT_MSG(itqt, "dynamic_cast to ITQTransform failed");
        FAISS_THROW_IF_NOT_FMT(
                static_cast<int>(itqt->mean.size()) >= itqt->d_in,
                "ITQTransform mean size %d < d_in %d",
                (int)itqt->mean.size(),
                itqt->d_in);
    }
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

bool index_read_warn_on_null_invlists = true;

std::unique_ptr<InvertedLists> read_InvertedLists_up(
        IOReader* f,
        int io_flags) {
    uint32_t h;
    READ1(h);
    if (h == fourcc("il00")) {
        if (index_read_warn_on_null_invlists) {
            fprintf(stderr,
                    "read_InvertedLists:"
                    " WARN! inverted lists not stored with IVF object\n");
        }
        return nullptr;
    } else if (h == fourcc("ilpn") && !(io_flags & IO_FLAG_SKIP_IVF_DATA)) {
        size_t nlist, code_size, n_levels;
        READ1(nlist);
        FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(nlist, "ilpn nlist");
        READ1(code_size);
        READ1(n_levels);
        FAISS_THROW_IF_NOT_FMT(
                n_levels > 0, "invalid ilpn n_levels %zd", n_levels);
        constexpr size_t bs = Panorama::kDefaultBatchSize;
        auto ailp = std::make_unique<ArrayInvertedListsPanorama>(
                nlist, code_size, n_levels, bs);
        std::vector<size_t> sizes(nlist);
        read_ArrayInvertedLists_sizes(f, sizes);
        size_t byte_limit = get_deserialization_vector_byte_limit();
        for (size_t i = 0; i < nlist; i++) {
            FAISS_THROW_IF_NOT_FMT(
                    sizes[i] <= byte_limit / sizeof(idx_t),
                    "inverted list %zu ids size %zu exceeds "
                    "deserialization byte limit",
                    i,
                    sizes[i]);
            ailp->ids[i].resize(sizes[i]);
            size_t num_elems = ((sizes[i] + bs - 1) / bs) * bs;
            size_t codes_bytes = mul_no_overflow(
                    num_elems, code_size, "inverted list codes");
            FAISS_THROW_IF_NOT_FMT(
                    codes_bytes <= byte_limit,
                    "inverted list %zu codes size %zu exceeds "
                    "deserialization byte limit",
                    i,
                    codes_bytes);
            ailp->codes[i].resize(codes_bytes);
            size_t cum_sums_count = mul_no_overflow(
                    num_elems,
                    add_no_overflow(
                            n_levels, 1, "inverted list cum_sums n_levels"),
                    "inverted list cum_sums");
            FAISS_THROW_IF_NOT_FMT(
                    cum_sums_count <= byte_limit / sizeof(ailp->cum_sums[0][0]),
                    "inverted list %zu cum_sums size %zu exceeds "
                    "deserialization byte limit",
                    i,
                    cum_sums_count);
            ailp->cum_sums[i].resize(cum_sums_count);
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
    } else if (h == fourcc("ilp2") && !(io_flags & IO_FLAG_SKIP_IVF_DATA)) {
        size_t nlist, code_size, n_levels, bs;
        READ1(nlist);
        FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(nlist, "ilp2 nlist");
        READ1(code_size);
        READ1(n_levels);
        READ1(bs);
        FAISS_THROW_IF_NOT_FMT(
                n_levels > 0, "invalid ilp2 n_levels %zd", n_levels);
        FAISS_THROW_IF_NOT_FMT(bs > 0, "invalid ilp2 batch_size %zd", bs);
        auto ailp = std::make_unique<ArrayInvertedListsPanorama>(
                nlist, code_size, n_levels, bs);
        std::vector<size_t> sizes(nlist);
        read_ArrayInvertedLists_sizes(f, sizes);
        size_t byte_limit = get_deserialization_vector_byte_limit();
        for (size_t i = 0; i < nlist; i++) {
            FAISS_THROW_IF_NOT_FMT(
                    sizes[i] <= byte_limit / sizeof(idx_t),
                    "inverted list %zu ids size %zu exceeds "
                    "deserialization byte limit",
                    i,
                    sizes[i]);
            ailp->ids[i].resize(sizes[i]);
            size_t num_elems = ((sizes[i] + bs - 1) / bs) * bs;
            size_t codes_bytes = mul_no_overflow(
                    num_elems, code_size, "inverted list codes");
            FAISS_THROW_IF_NOT_FMT(
                    codes_bytes <= byte_limit,
                    "inverted list %zu codes size %zu exceeds "
                    "deserialization byte limit",
                    i,
                    codes_bytes);
            ailp->codes[i].resize(codes_bytes);
            size_t cum_sums_count = mul_no_overflow(
                    num_elems,
                    add_no_overflow(
                            n_levels, 1, "inverted list cum_sums n_levels"),
                    "inverted list cum_sums");
            FAISS_THROW_IF_NOT_FMT(
                    cum_sums_count <= byte_limit / sizeof(ailp->cum_sums[0][0]),
                    "inverted list %zu cum_sums size %zu exceeds "
                    "deserialization byte limit",
                    i,
                    cum_sums_count);
            ailp->cum_sums[i].resize(cum_sums_count);
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
        FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(ails->nlist, "ilar nlist");
        READ1(ails->code_size);
        ails->ids.resize(ails->nlist);
        ails->codes.resize(ails->nlist);
        std::vector<size_t> sizes(ails->nlist);
        read_ArrayInvertedLists_sizes(f, sizes);
        size_t ilar_byte_limit = get_deserialization_vector_byte_limit();
        for (size_t i = 0; i < ails->nlist; i++) {
            FAISS_THROW_IF_NOT_FMT(
                    sizes[i] <= ilar_byte_limit / sizeof(idx_t),
                    "inverted list %zu ids size %zu exceeds "
                    "deserialization byte limit",
                    i,
                    sizes[i]);
            ails->ids[i].resize(sizes[i]);
            size_t codes_bytes = mul_no_overflow(
                    sizes[i], ails->code_size, "inverted list codes");
            FAISS_THROW_IF_NOT_FMT(
                    codes_bytes <= ilar_byte_limit,
                    "inverted list %zu codes size %zu exceeds "
                    "deserialization byte limit",
                    i,
                    codes_bytes);
            ails->codes[i].resize(codes_bytes);
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
        FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(nlist, "ilar skip nlist");
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
    auto ils = read_InvertedLists_up(f, io_flags);
    if (ils) {
        FAISS_THROW_IF_NOT(ils->nlist == ivf.nlist);
        FAISS_THROW_IF_NOT(
                ils->code_size == InvertedLists::INVALID_CODE_SIZE ||
                ils->code_size == ivf.code_size);
    }
    ivf.invlists = ils.release();
    ivf.own_invlists = true;
}

void read_ProductQuantizer(ProductQuantizer* pq, IOReader* f) {
    READ1(pq->d);
    READ1(pq->M);
    READ1(pq->nbits);
    FAISS_THROW_IF_NOT_FMT(
            pq->M > 0, "invalid ProductQuantizer M=%zd (must be > 0)", pq->M);
    FAISS_THROW_IF_NOT_FMT(
            pq->nbits <= 24, "invalid ProductQuantizer nbits=%zd", pq->nbits);
    {
        size_t ksub = size_t{1} << pq->nbits;
        size_t n = mul_no_overflow(pq->d, ksub, "PQ centroids");
        FAISS_THROW_IF_NOT_MSG(
                n < get_deserialization_vector_byte_limit() / sizeof(float),
                "PQ centroids allocation would exceed deserialization byte limit");
    }
    pq->set_derived_values();
    READVECTOR(pq->centroids);
    FAISS_THROW_IF_NOT_FMT(
            pq->centroids.size() == pq->d * pq->ksub,
            "ProductQuantizer centroids size %zu != d * ksub (%zu * %zu = %zu)",
            pq->centroids.size(),
            pq->d,
            pq->ksub,
            pq->d * pq->ksub);
}

static void read_ResidualQuantizer_old(ResidualQuantizer& rq, IOReader* f) {
    READ1(rq.d);
    FAISS_THROW_IF_NOT_FMT(
            rq.d > 0, "invalid AdditiveQuantizer d %zd, must be > 0", rq.d);
    READ1(rq.M);
    FAISS_THROW_IF_NOT_FMT(
            rq.M > 0, "invalid AdditiveQuantizer M %zd, must be > 0", rq.M);
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
    FAISS_THROW_IF_NOT_FMT(
            aq.d > 0, "invalid AdditiveQuantizer d %zd, must be > 0", aq.d);
    READ1(aq.M);
    FAISS_THROW_IF_NOT_FMT(
            aq.M > 0, "invalid AdditiveQuantizer M %zd, must be > 0", aq.M);
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

    // Sanity-check codebooks size without knowing the effective dimension.
    // codebooks stores effective_d * total_codebook_size floats, so its
    // size must be a positive multiple of total_codebook_size.
    if (aq.total_codebook_size > 0) {
        FAISS_THROW_IF_NOT_FMT(
                aq.codebooks.size() >= aq.total_codebook_size &&
                        aq.codebooks.size() % aq.total_codebook_size == 0,
                "AdditiveQuantizer codebooks size %zd is not a positive "
                "multiple of total_codebook_size %zd",
                aq.codebooks.size(),
                aq.total_codebook_size);
    }
}

// Validate that the codebooks vector is large enough for the given
// effective dimension.  For a standalone AdditiveQuantizer the effective
// dimension equals aq.d.  For a ProductAdditiveQuantizer the codebooks
// are sized for d_sub = d / nsplits, so callers pass that instead.
static void validate_codebooks_size(
        const AdditiveQuantizer& aq,
        size_t effective_d) {
    size_t required = mul_no_overflow(
            effective_d, aq.total_codebook_size, "codebooks validation");
    FAISS_THROW_IF_NOT_FMT(
            aq.codebooks.size() >= required,
            "AdditiveQuantizer codebooks size %zd too small for "
            "d=%zd * total_codebook_size=%zd",
            aq.codebooks.size(),
            effective_d,
            aq.total_codebook_size);
}

// Validate FastScan fields shared by all FastScan index types.
// M, ksub, bbs must be positive; bbs must be 32-aligned; M2 must be
// roundup(M, 2); and ksub * M / ksub * M2 must not overflow.
static void validate_fastscan_fields(
        size_t M,
        size_t M2,
        size_t ksub,
        int bbs,
        const char* index_type) {
    FAISS_THROW_IF_NOT_FMT(
            M > 0 && ksub > 0,
            "%s: invalid quantizer state (M=%zd, ksub=%zd, must be > 0)",
            index_type,
            M,
            ksub);
    FAISS_THROW_IF_NOT_FMT(
            bbs > 0 && bbs % 32 == 0,
            "%s: invalid bbs=%d (must be > 0 and a multiple of 32)",
            index_type,
            bbs);
    size_t expected_M2 = (M + 1) & ~static_cast<size_t>(1); // roundup(M, 2)
    FAISS_THROW_IF_NOT_FMT(
            M2 == expected_M2,
            "%s: invalid M2=%zd (expected roundup(M=%zd, 2) = %zd)",
            index_type,
            M2,
            M,
            expected_M2);
    mul_no_overflow(ksub, M, index_type);
    mul_no_overflow(ksub, M2, index_type);
}

// Validate that the AdditiveQuantizer dimension matches the index header
// dimension.  compute_LUT() treats codebooks as a (d, total_codebook_size)
// matrix and query vectors are sized for idx_d, so a mismatch leads to
// out-of-bounds reads.
static void validate_aq_dimension_match(
        const AdditiveQuantizer& aq,
        int idx_d,
        const char* index_type) {
    FAISS_THROW_IF_NOT_FMT(
            aq.d == static_cast<size_t>(idx_d),
            "%s: AdditiveQuantizer d=%zd does not match index d=%d",
            index_type,
            aq.d,
            idx_d);
}

static void read_ResidualQuantizer(
        ResidualQuantizer& rq,
        IOReader* f,
        int io_flags) {
    read_AdditiveQuantizer(rq, f);
    validate_codebooks_size(rq, rq.d);
    READ1(rq.train_type);
    READ1(rq.max_beam_size);
    FAISS_THROW_IF_NOT_FMT(
            rq.max_beam_size > 0,
            "invalid max_beam_size %d, must be > 0",
            rq.max_beam_size);
    {
        // Validate that the key allocation driven by max_beam_size
        // (beam_size * M * sizeof(int32_t)) fits within the byte limit.
        size_t beam_alloc = mul_no_overflow(
                static_cast<size_t>(rq.max_beam_size),
                rq.M,
                "max_beam_size * M");
        beam_alloc = mul_no_overflow(
                beam_alloc, sizeof(int32_t), "max_beam_size * M * elem");
        FAISS_THROW_IF_NOT_FMT(
                beam_alloc < get_deserialization_vector_byte_limit(),
                "max_beam_size %d * M %zd would exceed "
                "deserialization vector byte limit",
                rq.max_beam_size,
                rq.M);
    }
    if ((rq.train_type & ResidualQuantizer::Skip_codebook_tables) ||
        (io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE)) {
        // don't precompute the tables
    } else {
        rq.compute_codebook_tables();
    }
}

static void read_LocalSearchQuantizer(LocalSearchQuantizer& lsq, IOReader* f) {
    read_AdditiveQuantizer(lsq, f);
    validate_codebooks_size(lsq, lsq.d);
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
    FAISS_THROW_IF_NOT_FMT(
            paq.nsplits > 0,
            "invalid ProductAdditiveQuantizer nsplits %zd (must be > 0)",
            paq.nsplits);
    FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(paq.nsplits, "nsplits");
    FAISS_THROW_IF_NOT_FMT(
            paq.d % paq.nsplits == 0,
            "ProductAdditiveQuantizer d=%zd not divisible by nsplits=%zd",
            paq.d,
            paq.nsplits);
    validate_codebooks_size(paq, paq.d / paq.nsplits);
}

static void read_ProductResidualQuantizer(
        ProductResidualQuantizer& prq,
        IOReader* f,
        int io_flags) {
    read_ProductAdditiveQuantizer(prq, f);

    size_t d_sub = prq.d / prq.nsplits;
    for (size_t i = 0; i < prq.nsplits; i++) {
        auto rq = std::make_unique<ResidualQuantizer>();
        read_ResidualQuantizer(*rq, f, io_flags);
        FAISS_THROW_IF_NOT_FMT(
                rq->d == d_sub,
                "ProductResidualQuantizer sub-quantizer %zd has d=%zd, "
                "expected d_sub=%zd (d=%zd / nsplits=%zd)",
                i,
                rq->d,
                d_sub,
                prq.d,
                prq.nsplits);
        prq.quantizers.push_back(rq.release());
    }
}

static void read_ProductLocalSearchQuantizer(
        ProductLocalSearchQuantizer& plsq,
        IOReader* f) {
    read_ProductAdditiveQuantizer(plsq, f);

    size_t d_sub = plsq.d / plsq.nsplits;
    for (size_t i = 0; i < plsq.nsplits; i++) {
        auto lsq = std::make_unique<LocalSearchQuantizer>();
        read_LocalSearchQuantizer(*lsq, f);
        FAISS_THROW_IF_NOT_FMT(
                lsq->d == d_sub,
                "ProductLocalSearchQuantizer sub-quantizer %zd has d=%zd, "
                "expected d_sub=%zd (d=%zd / nsplits=%zd)",
                i,
                lsq->d,
                d_sub,
                plsq.d,
                plsq.nsplits);
        plsq.quantizers.push_back(lsq.release());
    }
}

void read_ScalarQuantizer(
        ScalarQuantizer* ivsc,
        IOReader* f,
        const Index& idx) {
    int qtype_int;
    READ1(qtype_int);
    FAISS_THROW_IF_NOT_FMT(
            qtype_int >= ScalarQuantizer::QT_8bit &&
                    qtype_int < ScalarQuantizer::QT_count,
            "invalid ScalarQuantizer qtype %d",
            qtype_int);
    ivsc->qtype = static_cast<ScalarQuantizer::QuantizerType>(qtype_int);
    READ1(ivsc->rangestat);
    READ1(ivsc->rangestat_arg);
    READ1(ivsc->d);
    READ1(ivsc->code_size);
    FAISS_THROW_IF_NOT_FMT(
            static_cast<size_t>(idx.d) == ivsc->d,
            "ScalarQuantizer d %zu != index header d %d",
            ivsc->d,
            idx.d);
    READVECTOR(ivsc->trained);
    // Validate trained vector size matches the quantizer type and dimension.
    // UNIFORM/NON_UNIFORM qtypes require training data; other qtypes
    // (fp16, bf16, 8bit_direct*) need none.
    // An untrained index (is_trained == false) legitimately has
    // trained.size() == 0, so we allow that case.
    {
        size_t expected = 0;
        switch (ivsc->qtype) {
            case ScalarQuantizer::QT_4bit_uniform:
            case ScalarQuantizer::QT_8bit_uniform:
                expected = 2;
                break;
            case ScalarQuantizer::QT_4bit:
            case ScalarQuantizer::QT_8bit:
            case ScalarQuantizer::QT_6bit:
                expected = 2 * ivsc->d;
                break;
            case ScalarQuantizer::QT_fp16:
            case ScalarQuantizer::QT_bf16:
            case ScalarQuantizer::QT_8bit_direct:
            case ScalarQuantizer::QT_8bit_direct_signed:
            case ScalarQuantizer::QT_0bit:
            case ScalarQuantizer::QT_count:
                expected = 0;
                break;
            case ScalarQuantizer::QT_1bit_tqmse:
                expected = 2 + 1; // 2^bits centroids + (2^bits - 1) boundaries
                break;
            case ScalarQuantizer::QT_2bit_tqmse:
                expected = 4 + 3;
                break;
            case ScalarQuantizer::QT_3bit_tqmse:
                expected = 8 + 7;
                break;
            case ScalarQuantizer::QT_4bit_tqmse:
                expected = 16 + 15;
                break;
            case ScalarQuantizer::QT_8bit_tqmse:
                expected = 256 + 255;
                break;
        }
        if (ivsc->trained.empty() && expected > 0) {
            // Empty trained is only valid for untrained indices.
            FAISS_THROW_IF_NOT_FMT(
                    !idx.is_trained,
                    "ScalarQuantizer trained size 0 != expected %zu "
                    "for qtype %d, d %zu (index is marked as trained)",
                    expected,
                    (int)ivsc->qtype,
                    ivsc->d);
        } else {
            FAISS_THROW_IF_NOT_FMT(
                    ivsc->trained.size() == expected,
                    "ScalarQuantizer trained size %zu != expected %zu "
                    "for qtype %d, d %zu",
                    ivsc->trained.size(),
                    expected,
                    (int)ivsc->qtype,
                    ivsc->d);
            if (expected > 0) {
                FAISS_THROW_IF_NOT_MSG(
                        idx.is_trained,
                        "ScalarQuantizer has training data but "
                        "index header is_trained is false");
            }
        }
    }
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

    // every levels[i] must be a valid index into cum_nneighbor_per_level
    size_t cum_size = hnsw.cum_nneighbor_per_level.size();
    for (size_t i = 0; i < ntotal; i++) {
        FAISS_THROW_IF_NOT_FMT(
                hnsw.levels[i] >= 0 &&
                        static_cast<size_t>(hnsw.levels[i]) < cum_size,
                "HNSW levels[%zd] = %d out of range [0, %zd)",
                i,
                hnsw.levels[i],
                cum_size);
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
    FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(nsg.ntotal, "nsg.ntotal");
    FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(nsg.R, "nsg.R");
    FAISS_THROW_IF_NOT_FMT(nsg.R > 0, "invalid NSG R %d (must be > 0)", nsg.R);
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
    // Validate neighbor IDs in the graph
    if (nnd.has_built && nnd.K > 0 && nnd.ntotal > 0) {
        FAISS_THROW_IF_NOT_FMT(
                nnd.final_graph.size() == (size_t)nnd.ntotal * (size_t)nnd.K,
                "NNDescent final_graph size %zu != ntotal * K (%d * %d = %zu)",
                nnd.final_graph.size(),
                nnd.ntotal,
                nnd.K,
                (size_t)nnd.ntotal * (size_t)nnd.K);
        for (size_t i = 0; i < nnd.final_graph.size(); i++) {
            int id = nnd.final_graph[i];
            FAISS_THROW_IF_NOT_FMT(
                    id >= -1 && id < nnd.ntotal,
                    "NNDescent final_graph[%zu] = %d out of range "
                    "[-1, %d)",
                    i,
                    id,
                    nnd.ntotal);
        }
    }
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
        int expected_d,
        bool multi_bit = true) {
    READ1(rabitq.d);
    READ1(rabitq.code_size);
    int metric_type_int;
    READ1(metric_type_int);
    rabitq.metric_type = metric_type_from_int(metric_type_int);

    if (multi_bit) {
        READ1(rabitq.nb_bits);
    } else {
        rabitq.nb_bits = 1;
    }

    FAISS_THROW_IF_NOT_FMT(
            rabitq.d == static_cast<size_t>(expected_d),
            "RaBitQuantizer dimension mismatch: rabitq.d=%zu vs index d=%d",
            rabitq.d,
            expected_d);
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
    FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(ivf->nlist, "nlist");
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
            // k_factor multiplies k to size search-time allocations
            // (n * k * k_factor labels + distances).  Defaults are 1
            // (IndexRefine) and 4 (IndexIVFPQR); AutoTune explores
            // powers-of-two up to 64.  Cap at 1000 to leave ample
            // headroom beyond any known usage while still blocking
            // OOM from crafted files (same cap as beam_factor in
            // ResidualCoarseQuantizer).
            FAISS_THROW_IF_NOT_FMT(
                    std::isfinite(ivfpqr->k_factor) &&
                            ivfpqr->k_factor >= 1.0f &&
                            ivfpqr->k_factor <= 1000.0f,
                    "k_factor %.6g out of valid range [1, 1000]"
                    " for IndexIVFPQR",
                    ivfpqr->k_factor);
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
        FAISS_THROW_IF_NOT_FMT(n_levels > 0, "invalid n_levels %zd", n_levels);
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
        FAISS_THROW_IF_NOT(
                idxp->codes.size() == idxp->ntotal * idxp->code_size);
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
        validate_aq_dimension_match(
                idxr->rq, idxr->d, "IndexResidualQuantizer");
        READ1(idxr->code_size);
        read_vector(idxr->codes, f);
        FAISS_THROW_IF_NOT(
                idxr->codes.size() == idxr->ntotal * idxr->code_size);
        idx = std::move(idxr);
    } else if (h == fourcc("IxLS")) {
        auto idxr = std::make_unique<IndexLocalSearchQuantizer>();
        read_index_header(*idxr, f);
        read_LocalSearchQuantizer(idxr->lsq, f);
        validate_aq_dimension_match(
                idxr->lsq, idxr->d, "IndexLocalSearchQuantizer");
        READ1(idxr->code_size);
        read_vector(idxr->codes, f);
        FAISS_THROW_IF_NOT(
                idxr->codes.size() == idxr->ntotal * idxr->code_size);
        idx = std::move(idxr);
    } else if (h == fourcc("IxPR")) {
        auto idxpr = std::make_unique<IndexProductResidualQuantizer>();
        read_index_header(*idxpr, f);
        read_ProductResidualQuantizer(idxpr->prq, f, io_flags);
        validate_aq_dimension_match(
                idxpr->prq, idxpr->d, "IndexProductResidualQuantizer");
        READ1(idxpr->code_size);
        read_vector(idxpr->codes, f);
        FAISS_THROW_IF_NOT(
                idxpr->codes.size() == idxpr->ntotal * idxpr->code_size);
        idx = std::move(idxpr);
    } else if (h == fourcc("IxPL")) {
        auto idxpl = std::make_unique<IndexProductLocalSearchQuantizer>();
        read_index_header(*idxpl, f);
        read_ProductLocalSearchQuantizer(idxpl->plsq, f);
        validate_aq_dimension_match(
                idxpl->plsq, idxpl->d, "IndexProductLocalSearchQuantizer");
        READ1(idxpl->code_size);
        read_vector(idxpl->codes, f);
        FAISS_THROW_IF_NOT(
                idxpl->codes.size() == idxpl->ntotal * idxpl->code_size);
        idx = std::move(idxpl);
    } else if (h == fourcc("ImRQ")) {
        auto idxr = std::make_unique<ResidualCoarseQuantizer>();
        read_index_header(*idxr, f);
        read_ResidualQuantizer(idxr->rq, f, io_flags);
        validate_aq_dimension_match(
                idxr->rq, idxr->d, "ResidualCoarseQuantizer");
        READ1(idxr->beam_factor);
        if (io_flags & IO_FLAG_SKIP_PRECOMPUTE_TABLE) {
            // then we force the beam factor to -1
            // which skips the table precomputation.
            idxr->beam_factor = -1;
        }
        FAISS_THROW_IF_NOT_MSG(
                static_cast<size_t>(idxr->ntotal) <
                        get_deserialization_vector_byte_limit() / sizeof(float),
                "ResidualCoarseQuantizer centroid norms allocation would "
                "exceed deserialization byte limit");
        // Validate beam_factor to prevent overflow in search() where
        // beam_size = int(k * beam_factor) and allocations scale with it.
        if (idxr->beam_factor > 0) {
            FAISS_THROW_IF_NOT_FMT(
                    idxr->beam_factor <= 1000.0f,
                    "beam_factor %.6g is too large (max 1000)",
                    idxr->beam_factor);
        }
        // Validate ntotal against byte limit: search() allocates
        // O(ntotal * M) when beam_size is capped to ntotal.
        {
            size_t ntotal_alloc = mul_no_overflow(
                    static_cast<size_t>(idxr->ntotal),
                    idxr->rq.M,
                    "ntotal * M");
            ntotal_alloc = mul_no_overflow(
                    ntotal_alloc, sizeof(int32_t), "ntotal * M * elem");
            FAISS_THROW_IF_NOT_FMT(
                    ntotal_alloc < get_deserialization_vector_byte_limit(),
                    "ResidualCoarseQuantizer ntotal %" PRId64
                    " * M %zd would exceed "
                    "deserialization vector byte limit",
                    idxr->ntotal,
                    idxr->rq.M);
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
        validate_aq_dimension_match(
                *idxaqfs->aq, idxaqfs->d, "IndexAdditiveQuantizerFastScan");

        READ1(idxaqfs->implem);
        READ1(idxaqfs->bbs);
        READ1(idxaqfs->qbs);
        FAISS_THROW_IF_NOT_MSG(idxaqfs->qbs >= 0, "qbs must be non-negative");

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

        validate_fastscan_fields(
                idxaqfs->M,
                idxaqfs->M2,
                idxaqfs->ksub,
                idxaqfs->bbs,
                "IndexAdditiveQuantizerFastScan");

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
        validate_aq_dimension_match(
                *ivaqfs->aq, ivaqfs->d, "IndexIVFAdditiveQuantizerFastScan");

        READ1(ivaqfs->by_residual);
        READ1(ivaqfs->implem);
        READ1(ivaqfs->bbs);
        READ1(ivaqfs->qbs);
        FAISS_THROW_IF_NOT_MSG(ivaqfs->qbs >= 0, "qbs must be non-negative");

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

        validate_fastscan_fields(
                ivaqfs->M,
                ivaqfs->M2,
                ivaqfs->ksub,
                ivaqfs->bbs,
                "IndexIVFAdditiveQuantizerFastScan");

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
            for (size_t i = 0; i < tab.size(); i += 2) {
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
        ivfp->batch_size = Panorama::kDefaultBatchSize;
        read_InvertedLists(*ivfp, f, io_flags);
        idx = std::move(ivfp);
    } else if (h == fourcc("IwP2")) {
        auto ivfp = std::make_unique<IndexIVFFlatPanorama>();
        read_ivf_header(ivfp.get(), f);
        ivfp->code_size = ivfp->d * sizeof(float);
        READ1(ivfp->n_levels);
        READ1(ivfp->batch_size);
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
        read_ScalarQuantizer(&idxs->sq, f, *idxs);
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
                r2 > 0, "invalid IndexLattice r2 %d (must be > 0)", r2);
        int dsq = d / nsq;
        FAISS_THROW_IF_NOT_FMT(
                dsq >= 2 && (dsq & (dsq - 1)) == 0,
                "invalid IndexLattice d=%d, nsq=%d: d/nsq=%d must be a power of 2 >= 2",
                d,
                nsq,
                dsq);
        auto idxl = std::make_unique<IndexLattice>(d, nsq, scale_nbit, r2);
        read_index_header(*idxl, f);
        READVECTOR(idxl->trained);
        idx = std::move(idxl);
    } else if (h == fourcc("IvSQ")) { // legacy
        auto ivsc = std::make_unique<IndexIVFScalarQuantizer>();
        std::vector<std::vector<idx_t>> ids;
        read_ivf_header(ivsc.get(), f, &ids);
        read_ScalarQuantizer(&ivsc->sq, f, *ivsc);
        READ1(ivsc->code_size);
        ArrayInvertedLists* ail = set_array_invlist(ivsc.get(), ids);
        for (size_t i = 0; i < ivsc->nlist; i++)
            READVECTOR(ail->codes[i]);
        idx = std::move(ivsc);
    } else if (h == fourcc("IwSQ") || h == fourcc("IwSq")) {
        auto ivsc = std::make_unique<IndexIVFScalarQuantizer>();
        read_ivf_header(ivsc.get(), f);
        read_ScalarQuantizer(&ivsc->sq, f, *ivsc);
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
        validate_aq_dimension_match(
                *iva->aq, iva->d, "IndexIVFAdditiveQuantizer");
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
        auto ivf_idx = read_index_up(f, io_flags);
        indep->index_ivf = dynamic_cast<IndexIVF*>(ivf_idx.get());
        FAISS_THROW_IF_NOT(indep->index_ivf);
        ivf_idx.release();
        if (indep->vt) {
            FAISS_THROW_IF_NOT_FMT(
                    indep->vt->d_in == indep->d,
                    "IndexIVFIndependentQuantizer: vt->d_in (%d) != index d (%d)",
                    indep->vt->d_in,
                    indep->d);
            FAISS_THROW_IF_NOT_FMT(
                    indep->vt->d_out == indep->index_ivf->d,
                    "IndexIVFIndependentQuantizer: vt->d_out (%d) != index_ivf->d (%d)",
                    indep->vt->d_out,
                    indep->index_ivf->d);
        }
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
        FAISS_THROW_IF_NOT_FMT(
                nt >= 0,
                "invalid VectorTransform chain length %d (must be >= 0)",
                nt);
        FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(
                nt, "VectorTransform chain length");
        for (int i = 0; i < nt; i++) {
            ixpt->chain.push_back(read_VectorTransform(f));
        }
        ixpt->index = read_index(f, io_flags);
        // Validate transform chain dimension consistency:
        // chain[0].d_in must equal the outer index d, consecutive
        // transforms must have matching d_out/d_in, and the last
        // transform's d_out must equal the sub-index d.
        if (nt > 0) {
            FAISS_THROW_IF_NOT_FMT(
                    ixpt->chain[0]->d_in == ixpt->d,
                    "IndexPreTransform chain[0] d_in=%d != index d=%d",
                    ixpt->chain[0]->d_in,
                    ixpt->d);
            for (int i = 1; i < nt; i++) {
                FAISS_THROW_IF_NOT_FMT(
                        ixpt->chain[i]->d_in == ixpt->chain[i - 1]->d_out,
                        "IndexPreTransform chain[%d] d_in=%d != "
                        "chain[%d] d_out=%d",
                        i,
                        ixpt->chain[i]->d_in,
                        i - 1,
                        ixpt->chain[i - 1]->d_out);
            }
            if (ixpt->index) {
                FAISS_THROW_IF_NOT_FMT(
                        ixpt->chain[nt - 1]->d_out == ixpt->index->d,
                        "IndexPreTransform chain[%d] d_out=%d "
                        "!= sub-index d=%d",
                        nt - 1,
                        ixpt->chain[nt - 1]->d_out,
                        ixpt->index->d);
            }
        }
        idx = std::move(ixpt);
    } else if (h == fourcc("Imiq")) {
        auto imiq = std::make_unique<MultiIndexQuantizer>();
        read_index_header(*imiq, f);
        read_ProductQuantizer(&imiq->pq, f);
        idx = std::move(imiq);
    } else if (h == fourcc("IxRF") || h == fourcc("IxRP")) {
        auto idxrf = std::make_unique<IndexRefine>();
        read_index_header(*idxrf, f);
        auto base = read_index_up(f, io_flags);
        auto refine = read_index_up(f, io_flags);
        READ1(idxrf->k_factor);
        // Same rationale as IndexIVFPQR k_factor above.
        FAISS_THROW_IF_NOT_FMT(
                std::isfinite(idxrf->k_factor) && idxrf->k_factor >= 1.0f &&
                        idxrf->k_factor <= 1000.0f,
                "k_factor %.6g out of valid range [1, 1000] for IndexRefine",
                idxrf->k_factor);
        if (h == fourcc("IxRP")) {
            // then make a RefineFlatPanorama with it
            auto idxrf_new = std::make_unique<IndexRefinePanorama>();
            static_cast<IndexRefine&>(*idxrf_new) = *idxrf;
            idxrf = std::move(idxrf_new);
        } else if (dynamic_cast<IndexFlat*>(refine.get())) {
            // then make a RefineFlat with it
            auto idxrf_new = std::make_unique<IndexRefineFlat>();
            static_cast<IndexRefine&>(*idxrf_new) = *idxrf;
            idxrf = std::move(idxrf_new);
        }
        idxrf->base_index = base.release();
        idxrf->refine_index = refine.release();
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
        FAISS_THROW_IF_NOT_FMT(
                idxmap->id_map.size() == idxmap->ntotal,
                "IndexIDMap id_map size (%" PRId64
                ") does not match ntotal (%" PRId64 ")",
                int64_t(idxmap->id_map.size()),
                idxmap->ntotal);
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
            FAISS_THROW_IF_NOT_MSG(
                    idx_panorama,
                    "dynamic_cast to IndexHNSWFlatPanorama failed");
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
            FAISS_THROW_IF_NOT_MSG(
                    idx_hnsw_cagra, "dynamic_cast to IndexHNSWCagra failed");
            READ1(idx_hnsw_cagra->base_level_only);
            READ1(idx_hnsw_cagra->num_base_level_search_entrypoints);
            if (h == fourcc("IHc2")) {
                READ1(idx_hnsw_cagra->numeric_type_);
            } else { // cagra before numeric_type_ was introduced
                idx_hnsw_cagra->set_numeric_type(faiss::Float32);
            }
        }
        read_HNSW(idxhnsw->hnsw, f);
        // Cross-check HNSW graph size against index header ntotal
        FAISS_THROW_IF_NOT_FMT(
                idxhnsw->hnsw.levels.size() == (size_t)idxhnsw->ntotal,
                "HNSW levels size %zu != index ntotal %" PRId64,
                idxhnsw->hnsw.levels.size(),
                idxhnsw->ntotal);
        idxhnsw->hnsw.is_panorama = (h == fourcc("IHfP"));
        idxhnsw->storage = read_index(f, io_flags);
        idxhnsw->own_fields = idxhnsw->storage != nullptr;
        // Cross-check storage ntotal and d against index
        if (idxhnsw->storage) {
            FAISS_THROW_IF_NOT_FMT(
                    idxhnsw->storage->ntotal == idxhnsw->ntotal,
                    "HNSW storage ntotal %" PRId64 " != index ntotal %" PRId64,
                    idxhnsw->storage->ntotal,
                    idxhnsw->ntotal);
            FAISS_THROW_IF_NOT_FMT(
                    idxhnsw->storage->d == idxhnsw->d,
                    "HNSW storage d %d != index d %d",
                    idxhnsw->storage->d,
                    idxhnsw->d);
        }
        if (h == fourcc("IHN2")) {
            FAISS_THROW_IF_NOT_MSG(
                    idxhnsw->storage,
                    "IndexHNSW2Level requires non-null storage");
            FAISS_THROW_IF_NOT_MSG(
                    dynamic_cast<Index2Layer*>(idxhnsw->storage) ||
                            dynamic_cast<IndexIVFPQ*>(idxhnsw->storage),
                    "IndexHNSW2Level storage must be Index2Layer or IndexIVFPQ");
        }
        if (h == fourcc("IHNp") && !(io_flags & IO_FLAG_PQ_SKIP_SDC_TABLE)) {
            auto* storage_pq = dynamic_cast<IndexPQ*>(idxhnsw->storage);
            FAISS_THROW_IF_NOT_MSG(
                    storage_pq,
                    "dynamic_cast to IndexPQ failed for HNSW storage");
            storage_pq->pq.compute_sdc_table();
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
        // Cross-check NSG graph ntotal against index header ntotal
        if (idxnsg->nsg.is_built) {
            FAISS_THROW_IF_NOT_FMT(
                    idxnsg->nsg.ntotal == idxnsg->ntotal,
                    "NSG ntotal %d != index ntotal %" PRId64,
                    idxnsg->nsg.ntotal,
                    idxnsg->ntotal);
        }
        idxnsg->storage = read_index(f, io_flags);
        idxnsg->own_fields = true;
        // Cross-check storage ntotal and d against index
        if (idxnsg->storage) {
            FAISS_THROW_IF_NOT_FMT(
                    idxnsg->storage->ntotal == idxnsg->ntotal,
                    "NSG storage ntotal %" PRId64 " != index ntotal %" PRId64,
                    idxnsg->storage->ntotal,
                    idxnsg->ntotal);
            FAISS_THROW_IF_NOT_FMT(
                    idxnsg->storage->d == idxnsg->d,
                    "NSG storage d %d != index d %d",
                    idxnsg->storage->d,
                    idxnsg->d);
        }
        idx = std::move(idxnsg);
    } else if (h == fourcc("INNf")) {
        auto idxnnd = std::make_unique<IndexNNDescentFlat>();
        read_index_header(*idxnnd, f);
        read_NNDescent(idxnnd->nndescent, f);
        // Cross-check NNDescent ntotal against index header ntotal
        if (idxnnd->nndescent.has_built) {
            FAISS_THROW_IF_NOT_FMT(
                    idxnnd->nndescent.ntotal == idxnnd->ntotal,
                    "NNDescent ntotal %d != index ntotal %" PRId64,
                    idxnnd->nndescent.ntotal,
                    idxnnd->ntotal);
        }
        idxnnd->storage = read_index(f, io_flags);
        idxnnd->own_fields = true;
        // Cross-check storage ntotal and d against index
        if (idxnnd->storage) {
            FAISS_THROW_IF_NOT_FMT(
                    idxnnd->storage->ntotal == idxnnd->ntotal,
                    "NNDescent storage ntotal %" PRId64
                    " != index ntotal %" PRId64,
                    idxnnd->storage->ntotal,
                    idxnnd->ntotal);
            FAISS_THROW_IF_NOT_FMT(
                    idxnnd->storage->d == idxnnd->d,
                    "NNDescent storage d %d != index d %d",
                    idxnnd->storage->d,
                    idxnnd->d);
        }
        idx = std::move(idxnnd);
    } else if (h == fourcc("IPfs")) {
        auto idxpqfs = std::make_unique<IndexPQFastScan>();
        read_index_header(*idxpqfs, f);
        read_ProductQuantizer(&idxpqfs->pq, f);
        READ1(idxpqfs->implem);
        READ1(idxpqfs->bbs);
        READ1(idxpqfs->qbs);
        FAISS_THROW_IF_NOT_MSG(idxpqfs->qbs >= 0, "qbs must be non-negative");
        READ1(idxpqfs->ntotal2);
        READ1(idxpqfs->M2);
        READVECTOR(idxpqfs->codes);

        const auto& pq = idxpqfs->pq;
        idxpqfs->M = pq.M;
        idxpqfs->nbits = pq.nbits;
        idxpqfs->ksub = (1 << pq.nbits);
        idxpqfs->code_size = pq.code_size;

        validate_fastscan_fields(
                idxpqfs->M,
                idxpqfs->M2,
                idxpqfs->ksub,
                idxpqfs->bbs,
                "IndexPQFastScan");

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

        validate_fastscan_fields(
                ivpq->M, ivpq->M2, ivpq->ksub, ivpq->bbs, "IndexIVFPQFastScan");

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
    } else if (h == fourcc("Irfn") || h == fourcc("Irfs")) {
        // Irfn = new format (aux data embedded in SIMD blocks)
        // Irfs = legacy format (flat_storage separate, needs migration)
        const bool is_legacy = (h == fourcc("Irfs"));

        auto idxqfs = std::make_unique<IndexRaBitQFastScan>();
        read_index_header(*idxqfs, f);
        read_RaBitQuantizer(idxqfs->rabitq, f, idxqfs->d, true);
        READVECTOR(idxqfs->center);
        READ1(idxqfs->qb);
        FAISS_THROW_IF_NOT_FMT(
                idxqfs->qb > 0 && idxqfs->qb <= 8,
                "invalid RaBitQ qb=%d (must be in [1, 8])",
                idxqfs->qb);

        std::vector<uint8_t> legacy_flat_storage;
        if (is_legacy) {
            READVECTOR(legacy_flat_storage);
        }

        READ1(idxqfs->bbs);
        READ1(idxqfs->ntotal2);
        READ1(idxqfs->M2);
        READ1(idxqfs->code_size);

        const size_t M_fastscan = (idxqfs->d + 3) / 4;
        constexpr size_t nbits_fastscan = 4;
        idxqfs->M = M_fastscan;
        idxqfs->nbits = nbits_fastscan;
        idxqfs->ksub = (1 << nbits_fastscan);

        validate_fastscan_fields(
                idxqfs->M,
                idxqfs->M2,
                idxqfs->ksub,
                idxqfs->bbs,
                "IndexRaBitQFastScan");

        READVECTOR(idxqfs->codes);

        if (is_legacy) {
            const size_t storage_size =
                    rabitq_utils::compute_per_vector_storage_size(
                            idxqfs->rabitq.nb_bits, idxqfs->d);

            FAISS_THROW_IF_NOT_MSG(
                    legacy_flat_storage.size() ==
                            static_cast<size_t>(idxqfs->ntotal) * storage_size,
                    "legacy flat_storage size mismatch during migration");

            rabitq_utils::populate_block_aux_from_flat_storage(
                    legacy_flat_storage,
                    idxqfs->codes,
                    static_cast<size_t>(idxqfs->ntotal),
                    idxqfs->bbs,
                    idxqfs->M2,
                    ((idxqfs->M2 + 1) / 2) * idxqfs->bbs,
                    idxqfs->get_block_stride(),
                    storage_size);
        }

        idx = std::move(idxqfs);
    } else if (h == fourcc("Ixrq")) {
        // Ixrq = original single-bit format
        auto idxq = std::make_unique<IndexRaBitQ>();
        read_index_header(*idxq, f);
        read_RaBitQuantizer(idxq->rabitq, f, idxq->d, false);
        READVECTOR(idxq->codes);
        READVECTOR(idxq->center);
        READ1(idxq->qb);
        // qb=0: Not quantized - direct distance computation on given float32s.
        // qb>0 && qb<=8: Scalar-quantized with qb bits of precision.
        FAISS_THROW_IF_NOT_FMT(
                idxq->qb <= 8,
                "invalid RaBitQ qb=%d (must be in [0, 8])",
                idxq->qb);

        // rabitq.nb_bits is already set to 1 by read_RaBitQuantizer
        idxq->code_size = idxq->rabitq.code_size;
        idx = std::move(idxq);
    } else if (h == fourcc("Ixrr")) {
        // Ixrr = multi-bit format (new)
        auto idxq = std::make_unique<IndexRaBitQ>();
        read_index_header(*idxq, f);
        read_RaBitQuantizer(
                idxq->rabitq, f, idxq->d, true); // Reads nb_bits from file
        READVECTOR(idxq->codes);
        READVECTOR(idxq->center);
        READ1(idxq->qb);
        // qb=0: Not quantized - direct distance computation on given float32s.
        // qb>0 && qb<=8: Scalar-quantized with qb bits of precision.
        FAISS_THROW_IF_NOT_FMT(
                idxq->qb <= 8,
                "invalid RaBitQ qb=%d (must be in [0, 8])",
                idxq->qb);

        idxq->code_size = idxq->rabitq.code_size;
        idx = std::move(idxq);
    } else if (h == fourcc("Iwrq")) {
        auto ivrq = std::make_unique<IndexIVFRaBitQ>();
        read_ivf_header(ivrq.get(), f);
        read_RaBitQuantizer(ivrq->rabitq, f, ivrq->d, false);
        READ1(ivrq->code_size);
        READ1(ivrq->by_residual);
        READ1(ivrq->qb);
        // qb=0: Not quantized - direct distance computation on given float32s.
        // qb>0 && qb<=8: Scalar-quantized with qb bits of precision.
        FAISS_THROW_IF_NOT_FMT(
                ivrq->qb <= 8,
                "invalid RaBitQ qb=%d (must be in [0, 8])",
                ivrq->qb);

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
        read_RaBitQuantizer(
                ivrq->rabitq, f, ivrq->d, true); // Reads nb_bits from file
        READ1(ivrq->code_size);
        READ1(ivrq->by_residual);
        READ1(ivrq->qb);
        // qb=0: Not quantized - direct distance computation on given float32s.
        // qb>0 && qb<=8: Scalar-quantized with qb bits of precision.
        FAISS_THROW_IF_NOT_FMT(
                ivrq->qb <= 8,
                "invalid RaBitQ qb=%d (must be in [0, 8])",
                ivrq->qb);

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

        int sk;
        READ1(sk);
        FAISS_THROW_IF_NOT_FMT(
                sk >= 0 && sk < static_cast<int>(SVS_count),
                "invalid SVS storage_kind=%d (must be in [0, %d))",
                sk,
                static_cast<int>(SVS_count));
        svs->storage_kind = static_cast<SVSStorageKind>(sk);

        if (h == fourcc("ISVL")) {
            auto* leanvec = dynamic_cast<IndexSVSVamanaLeanVec*>(svs.get());
            FAISS_THROW_IF_NOT_MSG(
                    leanvec, "dynamic_cast to IndexSVSVamanaLeanVec failed");
            READ1(leanvec->leanvec_d);
        }

        bool initialized;
        READ1(initialized);
        if (initialized) {
            faiss::svs_io::ReaderStreambuf rbuf(
                    f, get_deserialization_vector_byte_limit());
            std::istream is(&rbuf);
            svs->deserialize_impl(is);
        }
        if (h == fourcc("ISVL")) {
            bool trained;
            READ1(trained);
            if (trained) {
                faiss::svs_io::ReaderStreambuf rbuf(
                        f, get_deserialization_vector_byte_limit());
                std::istream is(&rbuf);
                auto* leanvec = dynamic_cast<IndexSVSVamanaLeanVec*>(svs.get());
                FAISS_THROW_IF_NOT_MSG(
                        leanvec,
                        "dynamic_cast to IndexSVSVamanaLeanVec failed");
                leanvec->deserialize_training_data(is);
            }
        }
        idx = std::move(svs);
    } else if (h == fourcc("ISVF")) {
        auto svs = std::make_unique<IndexSVSFlat>();
        read_index_header(*svs, f);

        bool initialized;
        READ1(initialized);
        if (initialized) {
            faiss::svs_io::ReaderStreambuf rbuf(
                    f, get_deserialization_vector_byte_limit());
            std::istream is(&rbuf);
            svs->deserialize_impl(is);
        }
        idx = std::move(svs);
    }
#endif // FAISS_ENABLE_SVS
    else if (h == fourcc("Iwrn") || h == fourcc("Iwrf")) {
        // Iwrn = new format (aux data embedded in SIMD blocks)
        // Iwrf = legacy format (flat_storage separate, needs migration)
        const bool is_legacy = (h == fourcc("Iwrf"));

        auto ivrqfs = std::make_unique<IndexIVFRaBitQFastScan>();
        read_ivf_header(ivrqfs.get(), f);
        read_RaBitQuantizer(ivrqfs->rabitq, f, ivrqfs->d);
        READ1(ivrqfs->by_residual);
        READ1(ivrqfs->code_size);
        READ1(ivrqfs->bbs);
        READ1(ivrqfs->qbs2);
        READ1(ivrqfs->M2);
        READ1(ivrqfs->implem);
        READ1(ivrqfs->qb);
        FAISS_THROW_IF_NOT_FMT(
                ivrqfs->qb > 0 && ivrqfs->qb <= 8,
                "invalid RaBitQ qb=%d (must be in [1, 8])",
                ivrqfs->qb);
        READ1(ivrqfs->centered);

        std::vector<uint8_t> legacy_flat_storage;
        if (is_legacy) {
            READVECTOR(legacy_flat_storage);
        }

        // Initialize FastScan base class fields
        const size_t M_fastscan = (ivrqfs->d + 3) / 4;
        constexpr size_t nbits_fastscan = 4;
        ivrqfs->M = M_fastscan;
        ivrqfs->nbits = nbits_fastscan;
        ivrqfs->ksub = (1 << nbits_fastscan);

        validate_fastscan_fields(
                ivrqfs->M,
                ivrqfs->M2,
                ivrqfs->ksub,
                ivrqfs->bbs,
                "IndexIVFRaBitQFastScan");

        read_InvertedLists(*ivrqfs, f, io_flags);
        ivrqfs->init_code_packer();

        if (is_legacy) {
            auto* bil = dynamic_cast<BlockInvertedLists*>(ivrqfs->invlists);
            FAISS_THROW_IF_NOT(bil);

            const size_t storage_size =
                    rabitq_utils::compute_per_vector_storage_size(
                            ivrqfs->rabitq.nb_bits, ivrqfs->d);
            const size_t new_block_stride = ivrqfs->get_block_stride();

            for (size_t list_no = 0; list_no < ivrqfs->nlist; list_no++) {
                if (bil->list_size(list_no) == 0) {
                    continue;
                }
                rabitq_utils::populate_block_aux_from_flat_storage(
                        legacy_flat_storage,
                        bil->codes[list_no],
                        bil->list_size(list_no),
                        ivrqfs->bbs,
                        ivrqfs->M2,
                        bil->block_size,
                        new_block_stride,
                        storage_size,
                        bil->ids[list_no].data());
            }

            if (bil->block_size < new_block_stride) {
                bil->block_size = new_block_stride;
            }
        }

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
    auto ils = read_InvertedLists_up(f, io_flags);
    FAISS_THROW_IF_NOT(
            !ils ||
            (ils->nlist == ivf.nlist && ils->code_size == ivf.code_size));
    ivf.invlists = ils.release();
    ivf.own_invlists = true;
}

static void read_index_binary_header(IndexBinary& idx, IOReader* f) {
    READ1(idx.d);
    READ1(idx.code_size);
    READ1(idx.ntotal);
    READ1(idx.is_trained);
    int metric_type_int;
    READ1(metric_type_int);
    idx.metric_type = metric_type_from_int(metric_type_int);
    FAISS_THROW_IF_NOT_FMT(
            idx.d > 0 && idx.d % 8 == 0,
            "invalid binary index dimension %d (must be > 0 and a multiple of 8)",
            idx.d);
    FAISS_THROW_IF_NOT_FMT(
            idx.code_size == idx.d / 8,
            "binary index code_size=%d does not match d/8=%d",
            (int)idx.code_size,
            idx.d / 8);
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
    FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(ivf.nlist, "nlist");
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
        size_t code_size,
        IOReader* f) {
    size_t sz;
    READ1(sz);
    FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(sz, "binary hash invlists sz");
    int il_nbit = 0;
    READ1(il_nbit);
    FAISS_THROW_IF_NOT_FMT(
            il_nbit >= 0,
            "invalid binary hash invlists il_nbit=%d (must be >= 0)",
            il_nbit);
    if (sz > 0) {
        FAISS_THROW_IF_NOT_FMT(
                il_nbit > 0,
                "invalid binary hash invlists il_nbit=%d for sz=%zd "
                "(must be > 0 when entries exist)",
                il_nbit,
                sz);
    }
    // buffer for bitstrings
    size_t bits_per_entry = (size_t)b + (size_t)il_nbit;
    size_t total_bits =
            mul_no_overflow(bits_per_entry, sz, "binary hash invlists");
    size_t needed_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> buf;
    READVECTOR(buf);
    FAISS_THROW_IF_NOT_FMT(
            buf.size() >= needed_bytes,
            "binary hash invlists: buffer size %zd < needed %zd bytes "
            "for %zd entries of %zd bits each",
            buf.size(),
            needed_bytes,
            sz,
            bits_per_entry);
    BitstringReader rd(buf.data(), buf.size());
    invlists.reserve(sz);
    for (size_t i = 0; i < sz; i++) {
        uint64_t hash = rd.read(b);
        uint64_t ilsz = rd.read(il_nbit);
        auto& il = invlists[hash];
        READVECTOR(il.ids);
        FAISS_THROW_IF_NOT(il.ids.size() == ilsz);
        READVECTOR(il.vecs);
        FAISS_THROW_IF_NOT_FMT(
                il.vecs.size() == il.ids.size() * code_size,
                "binary hash invlists: vecs size %zu != ids size %zu * "
                "code_size %zu",
                il.vecs.size(),
                il.ids.size(),
                code_size);
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
    FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(sz, "multi hash map sz");
    std::vector<uint8_t> buf;
    READVECTOR(buf);
    size_t nbit = add_no_overflow(
            mul_no_overflow((size_t)(b + id_bits), sz, "multi hash map"),
            mul_no_overflow(ntotal, (size_t)id_bits, "multi hash map"),
            "multi hash map total bits");
    FAISS_THROW_IF_NOT(buf.size() == (nbit + 7) / 8);
    BitstringReader rd(buf.data(), buf.size());
    map.reserve(sz);
    size_t total_ids = 0;
    for (size_t i = 0; i < sz; i++) {
        uint64_t hash = rd.read(b);
        uint64_t ilsz = rd.read(id_bits);
        FAISS_THROW_IF_NOT_FMT(
                ilsz <= ntotal - total_ids,
                "multi hash map: ilsz=%zu at entry %zu would exceed "
                "ntotal=%zu (already read %zu ids)",
                (size_t)ilsz,
                i,
                ntotal,
                total_ids);
        total_ids += ilsz;
        auto& il = map[hash];
        for (size_t j = 0; j < ilsz; j++) {
            uint64_t id = rd.read(id_bits);
            FAISS_THROW_IF_NOT_FMT(
                    id < ntotal,
                    "multi hash map: id=%zu >= ntotal=%zu",
                    (size_t)id,
                    ntotal);
            il.push_back(id);
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
        FAISS_THROW_IF_NOT_FMT(
                idxhnsw->hnsw.levels.size() == (size_t)idxhnsw->ntotal,
                "IndexBinaryHNSW HNSW levels size %zu != ntotal %" PRId64,
                idxhnsw->hnsw.levels.size(),
                idxhnsw->ntotal);
        idxhnsw->storage = read_index_binary(f, io_flags);
        idxhnsw->own_fields = true;
        FAISS_THROW_IF_NOT_MSG(
                idxhnsw->storage &&
                        dynamic_cast<IndexBinaryFlat*>(idxhnsw->storage) !=
                                nullptr,
                "IndexBinaryHNSW requires IndexBinaryFlat storage");
        FAISS_THROW_IF_NOT_MSG(
                idxhnsw->storage->ntotal == idxhnsw->ntotal,
                "IndexBinaryHNSW storage ntotal mismatch");
        idx = std::move(idxhnsw);
    } else if (h == fourcc("IBHc")) {
        auto idxhnsw = std::make_unique<IndexBinaryHNSWCagra>();
        read_index_binary_header(*idxhnsw, f);
        READ1(idxhnsw->keep_max_size_level0);
        READ1(idxhnsw->base_level_only);
        READ1(idxhnsw->num_base_level_search_entrypoints);
        read_HNSW(idxhnsw->hnsw, f);
        idxhnsw->hnsw.is_panorama = false;
        FAISS_THROW_IF_NOT_FMT(
                idxhnsw->hnsw.levels.size() == (size_t)idxhnsw->ntotal,
                "IndexBinaryHNSWCagra HNSW levels size %zu != ntotal %" PRId64,
                idxhnsw->hnsw.levels.size(),
                idxhnsw->ntotal);
        idxhnsw->storage = read_index_binary(f, io_flags);
        idxhnsw->own_fields = true;
        FAISS_THROW_IF_NOT_MSG(
                idxhnsw->storage &&
                        dynamic_cast<IndexBinaryFlat*>(idxhnsw->storage) !=
                                nullptr,
                "IndexBinaryHNSWCagra requires IndexBinaryFlat storage");
        FAISS_THROW_IF_NOT_MSG(
                idxhnsw->storage->ntotal == idxhnsw->ntotal,
                "IndexBinaryHNSWCagra storage ntotal mismatch");
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
        FAISS_THROW_IF_NOT(idxmap->id_map.size() == idxmap->ntotal);
        if (is_map2) {
            static_cast<IndexBinaryIDMap2*>(idxmap.get())->construct_rev_map();
        }
        idx = std::move(idxmap);
    } else if (h == fourcc("IBHh")) {
        auto idxh = std::make_unique<IndexBinaryHash>();
        read_index_binary_header(*idxh, f);
        READ1(idxh->b);
        FAISS_THROW_IF_NOT_FMT(
                idxh->b > 0,
                "invalid IndexBinaryHash b=%d (must be > 0)",
                idxh->b);
        FAISS_THROW_IF_NOT_FMT(
                static_cast<size_t>(idxh->b) <= idxh->code_size * 8,
                "IndexBinaryHash b=%d exceeds code_size=%d bits",
                idxh->b,
                idxh->code_size);
        READ1(idxh->nflip);
        read_binary_hash_invlists(idxh->invlists, idxh->b, idxh->code_size, f);
        idx = std::move(idxh);
    } else if (h == fourcc("IBHm")) {
        auto idxmh = std::make_unique<IndexBinaryMultiHash>();
        read_index_binary_header(*idxmh, f);
        auto storage_idx = read_index_binary_up(f);
        auto* flat_ptr = dynamic_cast<IndexBinaryFlat*>(storage_idx.get());
        FAISS_THROW_IF_NOT(flat_ptr && flat_ptr->ntotal == idxmh->ntotal);
        idxmh->storage = flat_ptr;
        storage_idx.release();
        idxmh->own_fields = true;
        READ1(idxmh->b);
        FAISS_THROW_IF_NOT_FMT(
                idxmh->b > 0,
                "invalid IndexBinaryMultiHash b=%d (must be > 0)",
                idxmh->b);
        READ1(idxmh->nhash);
        FAISS_THROW_IF_NOT_FMT(
                idxmh->nhash > 0,
                "invalid IndexBinaryMultiHash nhash %d (must be > 0)",
                idxmh->nhash);
        FAISS_CHECK_DESERIALIZATION_LOOP_LIMIT(idxmh->nhash, "nhash");
        FAISS_THROW_IF_NOT_FMT(
                mul_no_overflow(idxmh->nhash, idxmh->b, "nhash * b") <=
                        mul_no_overflow(idxmh->code_size, 8, "code_size * 8"),
                "IndexBinaryMultiHash nhash=%d * b=%d exceeds code_size=%d bits",
                idxmh->nhash,
                idxmh->b,
                idxmh->code_size);
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
