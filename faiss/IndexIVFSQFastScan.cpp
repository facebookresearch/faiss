/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFSQFastScan.h>

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

namespace faiss {

namespace {

size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

bool is_native_4bit(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_4bit ||
            qtype == ScalarQuantizer::QT_4bit_uniform;
}

bool needs_rerank(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_6bit ||
            qtype == ScalarQuantizer::QT_8bit ||
            qtype == ScalarQuantizer::QT_8bit_uniform ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed;
}

bool is_fallback(ScalarQuantizer::QuantizerType qtype) {
    return !is_native_4bit(qtype) && !needs_rerank(qtype);
}

bool is_uniform_range(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_8bit_uniform ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed;
}

void get_uniform_range(const ScalarQuantizer& sq, float& vmin, float& vdiff) {
    if (sq.qtype == ScalarQuantizer::QT_8bit_direct) {
        vmin = 0;
        vdiff = 255;
    } else if (sq.qtype == ScalarQuantizer::QT_8bit_direct_signed) {
        vmin = -128;
        vdiff = 255;
    } else {
        vmin = sq.trained[0];
        vdiff = sq.trained[1];
    }
}

void float_to_4bit_nibbles(
        const float* x,
        uint8_t* nibbles,
        idx_t n,
        int d,
        int M2,
        const ScalarQuantizer& sq) {
    const bool is_uniform = is_uniform_range(sq.qtype) ||
            sq.qtype == ScalarQuantizer::QT_4bit_uniform;
    const float* vmin_arr = nullptr;
    const float* vdiff_arr = nullptr;
    float vmin_s = 0, inv_vdiff_s = 0;

    if (is_uniform) {
        float vdiff;
        get_uniform_range(sq, vmin_s, vdiff);
        inv_vdiff_s = (vdiff > 0) ? (1.0f / vdiff) : 0;
    } else {
        vmin_arr = sq.trained.data();
        vdiff_arr = sq.trained.data() + d;
    }

    const int half = M2 / 2;
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        uint8_t* dst = nibbles + i * half;
        memset(dst, 0, half);
        for (int m = 0; m + 1 < d; m += 2) {
            float f0, f1;
            if (is_uniform) {
                f0 = (xi[m] - vmin_s) * inv_vdiff_s;
                f1 = (xi[m + 1] - vmin_s) * inv_vdiff_s;
            } else {
                f0 = (vdiff_arr[m] > 0) ? (xi[m] - vmin_arr[m]) / vdiff_arr[m]
                                        : 0;
                f1 = (vdiff_arr[m + 1] > 0)
                        ? (xi[m + 1] - vmin_arr[m + 1]) / vdiff_arr[m + 1]
                        : 0;
            }
            uint8_t lo = (uint8_t)std::min(15, std::max(0, (int)(f0 * 15.0f)));
            uint8_t hi = (uint8_t)std::min(15, std::max(0, (int)(f1 * 15.0f)));
            dst[m / 2] = lo | (hi << 4);
        }
        if (d & 1) {
            float f;
            if (is_uniform) {
                f = (xi[d - 1] - vmin_s) * inv_vdiff_s;
            } else {
                f = (vdiff_arr[d - 1] > 0)
                        ? (xi[d - 1] - vmin_arr[d - 1]) / vdiff_arr[d - 1]
                        : 0;
            }
            dst[(d - 1) / 2] =
                    (uint8_t)std::min(15, std::max(0, (int)(f * 15.0f)));
        }
    }
}

} // anonymous namespace

// -----------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------

IndexIVFSQFastScan::IndexIVFSQFastScan(
        Index* quantizer_in,
        size_t d_in,
        size_t nlist_in,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        int bbs_in,
        bool by_residual_in)
        : IndexIVFFastScan(quantizer_in, d_in, nlist_in, 0, metric, false),
          sq(d_in, qtype) {
    if (is_fallback(qtype)) {
        by_residual = by_residual_in;
        if (ScalarQuantizer::TurboQuantRefine::is_turboq_full(qtype)) {
            by_residual = false;
        }
    } else {
        // FastScan path: by_residual=false because our 2D LUT cannot
        // handle per-probe residuals (same rationale as IndexIVFPQFastScan).
        by_residual = false;
    }

    if (is_fallback(qtype)) {
        // Fallback: use standard ArrayInvertedLists with full SQ codes.
        // Search uses IndexIVF's InvertedListScanner path.
        code_size = sq.code_size;
        replace_invlists(new ArrayInvertedLists(nlist_in, code_size), true);
    } else {
        size_t M_in = d_in;
        M = M_in;
        nbits = 4;
        bbs = bbs_in;
        ksub = 16;
        M2 = roundup(M_in, 2);
        code_size = M2 / 2;

        replace_invlists(
                new BlockInvertedLists(nlist_in, get_CodePacker()), true);

        if (needs_rerank(qtype)) {
            orig_codes_invlists =
                    new ArrayInvertedLists(nlist_in, sq.code_size);
        }
    }
}

IndexIVFSQFastScan::IndexIVFSQFastScan() = default;

IndexIVFSQFastScan::IndexIVFSQFastScan(
        const IndexIVFScalarQuantizer& orig,
        int bbs_in)
        : IndexIVFFastScan(
                  orig.quantizer,
                  orig.d,
                  orig.nlist,
                  0,
                  orig.metric_type,
                  false),
          sq(orig.sq) {
    ntotal = orig.ntotal;
    is_trained = orig.is_trained;
    nprobe = orig.nprobe;

    if (is_fallback(sq.qtype)) {
        by_residual = orig.by_residual;
        code_size = sq.code_size;
        replace_invlists(new ArrayInvertedLists(nlist, code_size), true);

        for (size_t list_no = 0; list_no < nlist; list_no++) {
            size_t list_size = orig.invlists->list_size(list_no);
            if (list_size == 0)
                continue;
            InvertedLists::ScopedCodes codes(orig.invlists, list_no);
            InvertedLists::ScopedIds ids(orig.invlists, list_no);
            invlists->add_entries(list_no, list_size, ids.get(), codes.get());
        }
        return;
    }

    FAISS_THROW_IF_NOT_MSG(
            !orig.by_residual,
            "IndexIVFSQFastScan: conversion from IndexIVFScalarQuantizer "
            "with by_residual=true is not supported for native/rerank "
            "types. Set orig.by_residual=false and retrain, or construct "
            "IndexIVFSQFastScan directly.");

    by_residual = false;

    size_t M_in = d;
    M = M_in;
    nbits = 4;
    bbs = bbs_in;
    ksub = 16;
    M2 = roundup(M_in, 2);
    code_size = M2 / 2;

    replace_invlists(new BlockInvertedLists(nlist, get_CodePacker()), true);

    if (needs_rerank(sq.qtype)) {
        orig_codes_invlists = new ArrayInvertedLists(nlist, sq.code_size);
    }

    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = orig.invlists->list_size(list_no);
        if (list_size == 0) {
            continue;
        }

        InvertedLists::ScopedCodes orig_codes(orig.invlists, list_no);
        InvertedLists::ScopedIds orig_ids(orig.invlists, list_no);

        if (needs_rerank(sq.qtype)) {
            orig_codes_invlists->add_entries(
                    list_no, list_size, orig_ids.get(), orig_codes.get());
        }

        // Decode to float and re-quantize to 4-bit.
        std::vector<float> recon(list_size * d);
        sq.decode(orig_codes.get(), recon.data(), list_size);

        std::vector<uint8_t> flat4(list_size * M2 / 2, 0);
        if (is_native_4bit(sq.qtype)) {
            sq.compute_codes(recon.data(), flat4.data(), list_size);
        } else {
            float_to_4bit_nibbles(
                    recon.data(), flat4.data(), list_size, d, M2, sq);
        }

        BlockInvertedLists* bil = dynamic_cast<BlockInvertedLists*>(invlists);
        FAISS_THROW_IF_NOT(bil);

        size_t nb2 = roundup(list_size, bbs);
        AlignedTable<uint8_t> packed(nb2 * M2 / 2);
        pq4_pack_codes(flat4.data(), list_size, M, nb2, bbs, M2, packed.get());
        bil->add_entries(list_no, list_size, orig_ids.get(), packed.get());
    }
}

IndexIVFSQFastScan::~IndexIVFSQFastScan() {
    delete orig_codes_invlists;
}

// -----------------------------------------------------------------------
// Virtual method implementations
// -----------------------------------------------------------------------

size_t IndexIVFSQFastScan::fast_scan_code_size() const {
    return M2 / 2;
}

void IndexIVFSQFastScan::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* /*assign*/) {
    sq.train(n, x);
}

idx_t IndexIVFSQFastScan::train_encoder_num_vectors() const {
    return 100000;
}

void IndexIVFSQFastScan::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes_out,
        bool include_listnos) const {
    std::vector<float> residuals;
    const float* to_encode = x;

    if (by_residual) {
        residuals.resize(n * d);
        for (idx_t i = 0; i < n; i++) {
            if (list_nos[i] >= 0) {
                quantizer->compute_residual(
                        x + i * d, residuals.data() + i * d, list_nos[i]);
            } else {
                memset(residuals.data() + i * d, 0, sizeof(float) * d);
            }
        }
        to_encode = residuals.data();
    }

    if (is_fallback(sq.qtype)) {
        sq.compute_codes(to_encode, codes_out, n);
    } else if (is_native_4bit(sq.qtype)) {
        sq.compute_codes(to_encode, codes_out, n);
    } else {
        float_to_4bit_nibbles(to_encode, codes_out, n, d, M2, sq);
    }

    if (include_listnos) {
        size_t coarse_size = coarse_code_size();
        for (idx_t i = n - 1; i >= 0; i--) {
            uint8_t* code = codes_out + i * (coarse_size + code_size);
            memmove(code + coarse_size, codes_out + i * code_size, code_size);
            encode_listno(list_nos[i], code);
        }
    }
}

bool IndexIVFSQFastScan::lookup_table_is_3d() const {
    // For SQ, the LUT doesn't depend on which list we're scanning
    // (unlike PQ with residuals and precomputed tables).
    return false;
}

void IndexIVFSQFastScan::compute_LUT(
        size_t n,
        const float* x,
        const CoarseQuantized& cq,
        AlignedTable<float>& dis_tables,
        AlignedTable<float>& biases,
        const FastScanDistancePostProcessing&) const {
    size_t dim = d;
    size_t ksub_val = 16;
    size_t dim12 = dim * ksub_val;

    // Build reconstruction table: for each dimension, the 16 centroid values.
    bool is_uniform = is_uniform_range(sq.qtype) ||
            sq.qtype == ScalarQuantizer::QT_4bit_uniform;

    std::vector<float> recon_table(dim * ksub_val);

    if (is_uniform) {
        float vmin, vdiff;
        get_uniform_range(sq, vmin, vdiff);
        for (size_t c = 0; c < ksub_val; c++) {
            float recon = vmin + ((c + 0.5f) / 15.0f) * vdiff;
            for (size_t m = 0; m < dim; m++) {
                recon_table[m * ksub_val + c] = recon;
            }
        }
    } else {
        const float* vmin = sq.trained.data();
        const float* vdiff = sq.trained.data() + dim;
        for (size_t m = 0; m < dim; m++) {
            for (size_t c = 0; c < ksub_val; c++) {
                recon_table[m * ksub_val + c] =
                        vmin[m] + ((c + 0.5f) / 15.0f) * vdiff[m];
            }
        }
    }

    // by_residual is always false for the fast-scan path (non-fallback
    // types), so codes store raw quantized values and the LUT computes
    // distances directly from the raw query.  No biases needed.

    dis_tables.resize(n * dim12);

    if (metric_type == METRIC_L2) {
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * dim;
            float* lut_i = dis_tables.get() + i * dim12;
            for (size_t m = 0; m < dim; m++) {
                float qi = xi[m];
                const float* recon_m = recon_table.data() + m * ksub_val;
                float* lut_m = lut_i + m * ksub_val;
                for (size_t c = 0; c < ksub_val; c++) {
                    float diff = qi - recon_m[c];
                    lut_m[c] = diff * diff;
                }
            }
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            const float* xi = x + i * dim;
            float* lut_i = dis_tables.get() + i * dim12;
            for (size_t m = 0; m < dim; m++) {
                float qi = xi[m];
                const float* recon_m = recon_table.data() + m * ksub_val;
                float* lut_m = lut_i + m * ksub_val;
                for (size_t c = 0; c < ksub_val; c++) {
                    lut_m[c] = qi * recon_m[c];
                }
            }
        }
    }
}

void IndexIVFSQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    size_t coarse_size = coarse_code_size();

#pragma omp parallel if (n > 1)
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = bytes + i * (code_size + coarse_size);
            int64_t list_no = decode_listno(code);
            float* xi = x + i * d;
            sq.decode(code + coarse_size, xi, 1);
            if (by_residual) {
                quantizer->reconstruct(list_no, residual.data());
                for (int j = 0; j < d; j++) {
                    xi[j] += residual[j];
                }
            }
        }
    }
}

void IndexIVFSQFastScan::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    if (is_fallback(sq.qtype)) {
        // Fallback: codes stored directly in invlists.
        InvertedLists::ScopedCodes codes(invlists, list_no);
        sq.decode(codes.get() + offset * sq.code_size, recons, 1);
    } else if (orig_codes_invlists) {
        // Rerank: use original full-precision codes for reconstruction.
        InvertedLists::ScopedCodes codes(orig_codes_invlists, list_no);
        sq.decode(codes.get() + offset * sq.code_size, recons, 1);
    } else {
        // Native 4-bit: unpack from block inverted lists.
        std::vector<uint8_t> code(M2 / 2, 0);
        InvertedLists::ScopedCodes list_codes(invlists, list_no);
        BitstringWriter bsw(code.data(), M2 / 2);
        for (size_t m = 0; m < M; m++) {
            uint8_t c = pq4_get_packed_element(
                    list_codes.get(), bbs, M2, offset, m);
            bsw.write(c, 4);
        }
        sq.decode(code.data(), recons, 1);
    }

    if (by_residual) {
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());
        for (int j = 0; j < d; j++) {
            recons[j] += centroid[j];
        }
    }
}

InvertedListScanner* IndexIVFSQFastScan::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    return sq.select_InvertedListScanner(
            metric_type, quantizer, store_pairs, sel, by_residual);
}

// -----------------------------------------------------------------------
// search — fallback types use IndexIVF's scanner-based path
// -----------------------------------------------------------------------

void IndexIVFSQFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (is_fallback(sq.qtype)) {
        IndexIVF::search(n, x, k, distances, labels, params);
    } else {
        IndexIVFFastScan::search(n, x, k, distances, labels, params);
    }
}

void IndexIVFSQFastScan::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    if (is_fallback(sq.qtype)) {
        IndexIVF::search_preassigned(
                n,
                x,
                k,
                assign,
                centroid_dis,
                distances,
                labels,
                store_pairs,
                params,
                stats);
    } else {
        IndexIVFFastScan::search_preassigned(
                n,
                x,
                k,
                assign,
                centroid_dis,
                distances,
                labels,
                store_pairs,
                params,
                stats);
    }
}

void IndexIVFSQFastScan::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    if (is_fallback(sq.qtype)) {
        IndexIVF::range_search(n, x, radius, result, params);
    } else {
        IndexIVFFastScan::range_search(n, x, radius, result, params);
    }
}

// -----------------------------------------------------------------------
// add_with_ids — custom implementation to store both packed and original codes
// -----------------------------------------------------------------------

void IndexIVFSQFastScan::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);

    constexpr idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            add_with_ids(i1 - i0, x + i0 * d, xids ? xids + i0 : nullptr);
        }
        return;
    }

    // Assign to coarse clusters
    std::unique_ptr<idx_t[]> idx(new idx_t[n]);
    quantizer->assign(n, x, idx.get());

    if (is_fallback(sq.qtype)) {
        // Fallback: store SQ codes directly in invlists (same as
        // IndexIVFScalarQuantizer).
        std::vector<uint8_t> flat_codes(n * sq.code_size);
        encode_vectors(n, x, idx.get(), flat_codes.data());

        for (idx_t i = 0; i < n; i++) {
            if (idx[i] < 0) {
                continue;
            }
            idx_t id = xids ? xids[i] : ntotal + i;
            invlists->add_entry(
                    idx[i], id, flat_codes.data() + i * sq.code_size);
        }
        ntotal += n;
        return;
    }

    // Produce 4-bit packed codes for the SIMD scan,
    // and (for rerank types) also store original codes.
    AlignedTable<uint8_t> flat_codes(n * code_size);
    encode_vectors(n, x, idx.get(), flat_codes.get());

    // Store original SQ codes for rerank types
    std::vector<uint8_t> orig_flat_codes;
    if (needs_rerank(sq.qtype)) {
        orig_flat_codes.resize(n * sq.code_size);
        for (idx_t i = 0; i < n; i++) {
            if (idx[i] < 0) {
                continue;
            }
            if (by_residual) {
                std::vector<float> residual(d);
                quantizer->compute_residual(x + i * d, residual.data(), idx[i]);
                sq.compute_codes(
                        residual.data(),
                        orig_flat_codes.data() + i * sq.code_size,
                        1);
            } else {
                sq.compute_codes(
                        x + i * d,
                        orig_flat_codes.data() + i * sq.code_size,
                        1);
            }
        }
    }

    // Sort by list assignment
    std::vector<idx_t> order(n);
    for (idx_t i = 0; i < n; i++) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&idx](idx_t a, idx_t b) {
        return idx[a] < idx[b];
    });

    BlockInvertedLists* bil = dynamic_cast<BlockInvertedLists*>(invlists);
    FAISS_THROW_IF_NOT_MSG(bil, "only block inverted lists supported");

    idx_t i0 = 0;
    while (i0 < n) {
        idx_t list_no = idx[order[i0]];
        idx_t i1 = i0 + 1;
        while (i1 < n && idx[order[i1]] == list_no) {
            i1++;
        }

        if (list_no == -1) {
            i0 = i1;
            continue;
        }

        size_t list_size = bil->list_size(list_no);
        bil->resize(list_no, list_size + i1 - i0);

        AlignedTable<uint8_t> list_codes((i1 - i0) * code_size);
        for (idx_t i = i0; i < i1; i++) {
            size_t ofs = list_size + i - i0;
            idx_t id = xids ? xids[order[i]] : ntotal + order[i];
            bil->ids[list_no][ofs] = id;
            memcpy(list_codes.data() + (i - i0) * code_size,
                   flat_codes.data() + order[i] * code_size,
                   code_size);

            if (needs_rerank(sq.qtype)) {
                orig_codes_invlists->add_entry(
                        list_no,
                        id,
                        orig_flat_codes.data() + order[i] * sq.code_size);
            }
        }

        pq4_pack_codes_range(
                list_codes.data(),
                M,
                list_size,
                list_size + i1 - i0,
                bbs,
                M2,
                bil->codes[list_no].data(),
                0,
                get_block_stride());

        i0 = i1;
    }

    ntotal += n;
}

void IndexIVFSQFastScan::reset() {
    IndexIVFFastScan::reset();
    if (orig_codes_invlists) {
        for (size_t i = 0; i < nlist; i++) {
            orig_codes_invlists->resize(i, 0);
        }
    }
}

} // namespace faiss
