/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSQFastScan.h>

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include <faiss/IndexIVF.h> // InvertedListScanner
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/fast_scan/FastScanDistancePostProcessing.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/utils.h>

namespace faiss {

namespace {

size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

} // anonymous namespace

/// Native 4-bit types: use IndexFastScan SIMD path directly.
static bool is_native_4bit(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_4bit ||
            qtype == ScalarQuantizer::QT_4bit_uniform;
}

/// Types that benefit from re-quantisation to 4-bit + rerank.
static bool needs_rerank(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_6bit ||
            qtype == ScalarQuantizer::QT_8bit ||
            qtype == ScalarQuantizer::QT_8bit_uniform ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed;
}

/// Everything else (QT_fp16, QT_bf16, …) falls back to the
/// ScalarQuantizer's own SIMD-optimised scanner — same speed as
/// IndexScalarQuantizer, but available through one unified class.
static bool is_fallback(ScalarQuantizer::QuantizerType qtype) {
    return !is_native_4bit(qtype) && !needs_rerank(qtype);
}

/// Returns true for types that use a single uniform scalar range
/// (either learned or implicit).
static bool is_uniform_range(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_8bit_uniform ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed;
}

/// Get the uniform vmin/vdiff for a type.
/// QT_8bit_uniform: learned from sq.trained.
/// QT_8bit_direct: implicit [0, 255].
/// QT_8bit_direct_signed: implicit [-128, 255].
/// QT_4bit_uniform: learned from sq.trained.
static void get_uniform_range(
        const ScalarQuantizer& sq,
        float& vmin,
        float& vdiff) {
    if (sq.qtype == ScalarQuantizer::QT_8bit_direct) {
        vmin = 0;
        vdiff = 255;
    } else if (sq.qtype == ScalarQuantizer::QT_8bit_direct_signed) {
        vmin = -128;
        vdiff = 255;
    } else {
        // QT_8bit_uniform, QT_4bit_uniform: learned range
        vmin = sq.trained[0];
        vdiff = sq.trained[1];
    }
}

/// Quantise float vectors directly to 4-bit nibble-packed format.
/// Works for any SQ type: maps each dimension through its vmin / vdiff
/// to produce a 0–15 nibble code.
static void float_to_4bit_nibbles(
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

// -----------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------

IndexSQFastScan::IndexSQFastScan(
        int d_in,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        int bbs_in)
        : sq(d_in, qtype) {
    // init_fastscan sets base-class members (d, metric_type, M, bbs, …).
    // For fallback types we still call it so the struct is well-formed,
    // but the packed codes / LUT path is never used.
    init_fastscan(d_in, d_in, 4, metric, bbs_in);
}

IndexSQFastScan::IndexSQFastScan() = default;

IndexSQFastScan::IndexSQFastScan(const IndexScalarQuantizer& orig, int bbs_in)
        : sq(orig.sq) {
    init_fastscan(orig.d, orig.d, 4, orig.metric_type, bbs_in);
    ntotal = orig.ntotal;
    is_trained = orig.is_trained;

    if (needs_rerank(sq.qtype)) {
        // Copy original codes for reranking
        size_t cs = sq.code_size;
        codes_8bit.resize(ntotal * cs);
        memcpy(codes_8bit.data(), orig.codes.data(), ntotal * cs);

        // Decode to float, re-quantise to 4-bit, and pack for SIMD scan
        std::vector<float> recon(ntotal * d);
        sq.decode(codes_8bit.data(), recon.data(), ntotal);
        ntotal2 = roundup(ntotal, bbs_in);
        std::vector<uint8_t> flat(ntotal2 * M2 / 2, 0);
        float_to_4bit_nibbles(recon.data(), flat.data(), ntotal, d, M2, sq);
        codes.resize(ntotal2 * M2 / 2);
        pq4_pack_codes(
                flat.data(), ntotal, M, ntotal2, bbs_in, M2, codes.get());
    } else if (is_fallback(sq.qtype)) {
        // Fallback: just copy original codes for the SQ scanner
        size_t cs = sq.code_size;
        codes_8bit.resize(ntotal * cs);
        memcpy(codes_8bit.data(), orig.codes.data(), ntotal * cs);
    } else {
        // Native 4-bit: copy SQ codes to codes_8bit, then pack
        size_t cs = sq.code_size;
        codes_8bit.resize(ntotal * cs);
        memcpy(codes_8bit.data(), orig.codes.data(), ntotal * cs);
        ntotal2 = roundup(ntotal, bbs_in);
        codes.resize(ntotal2 * M2 / 2);
        pq4_pack_codes(
                orig.codes.data(), ntotal, M, ntotal2, bbs_in, M2, codes.get());
    }
}

// -----------------------------------------------------------------------
// train / add / reset
// -----------------------------------------------------------------------

void IndexSQFastScan::train(idx_t n, const float* x) {
    if (is_trained) {
        return;
    }
    sq.train(n, x);
    is_trained = true;
}

void IndexSQFastScan::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);

    if (needs_rerank(sq.qtype)) {
        // 1. Store original codes for reranking
        size_t cs = sq.code_size;
        codes_8bit.resize((ntotal + n) * cs);
        sq.compute_codes(x, codes_8bit.data() + ntotal * cs, n);

        // 2. Delegate to IndexFastScan::add which calls our
        //    compute_codes() override — that re-quantises to 4-bit.
        IndexFastScan::add(n, x);
    } else if (is_fallback(sq.qtype)) {
        // Fallback: store original codes only (no fast-scan packing)
        size_t cs = sq.code_size;
        codes_8bit.resize((ntotal + n) * cs);
        sq.compute_codes(x, codes_8bit.data() + ntotal * cs, n);
        ntotal += n;
    } else {
        // Native 4-bit: store SQ codes in codes_8bit (for distance
        // computer / HNSW), then pack into SIMD layout
        size_t cs = sq.code_size;
        codes_8bit.resize((ntotal + n) * cs);
        sq.compute_codes(x, codes_8bit.data() + ntotal * cs, n);
        IndexFastScan::add(n, x);
    }
}

void IndexSQFastScan::reset() {
    IndexFastScan::reset();
    codes_8bit.clear();
    codes_8bit.shrink_to_fit();
}

// -----------------------------------------------------------------------
// search
// -----------------------------------------------------------------------

void IndexSQFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    const IDSelector* sel = params ? params->sel : nullptr;

    if (is_native_4bit(sq.qtype)) {
        if (!sel) {
            IndexFastScan::search(n, x, k, distances, labels, nullptr);
            return;
        }
        // Fall through to SQ scanner path with IDSelector support.
    }

    if (is_native_4bit(sq.qtype) || is_fallback(sq.qtype)) {
        // SQ's own SIMD-optimised scanner (same as
        // IndexScalarQuantizer::search), supports IDSelector natively.
#pragma omp parallel
        {
            std::unique_ptr<InvertedListScanner> scanner(
                    sq.select_InvertedListScanner(
                            metric_type, nullptr, true, sel));
            scanner->list_no = 0;

#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                float* D = distances + k * i;
                idx_t* I = labels + k * i;
                if (metric_type == METRIC_L2) {
                    maxheap_heapify(k, D, I);
                } else {
                    minheap_heapify(k, D, I);
                }
                scanner->set_query(x + i * d);
                scanner->scan_codes(
                        ntotal, codes_8bit.data(), nullptr, D, I, k);
                if (metric_type == METRIC_L2) {
                    maxheap_reorder(k, D, I);
                } else {
                    minheap_reorder(k, D, I);
                }
            }
        }
        return;
    }

    // --- rerank path (QT_6bit / QT_8bit / QT_8bit_uniform) ---
    // Pass 1: coarse 4-bit fast scan with overselection
    // IndexFastScan::search does not support params, pass nullptr.
    idx_t k_rerank = std::min((idx_t)((int64_t)k * rerank_factor), ntotal);
    k_rerank = std::max(k_rerank, k);

    std::vector<float> coarse_dis(n * k_rerank);
    std::vector<idx_t> coarse_ids(n * k_rerank);
    IndexFastScan::search(
            n, x, k_rerank, coarse_dis.data(), coarse_ids.data(), nullptr);

    // Pass 2: rerank candidates with exact 8-bit distances,
    // applying IDSelector filtering.
#pragma omp parallel
    {
        std::unique_ptr<ScalarQuantizer::SQDistanceComputer> dc(
                sq.get_distance_computer(metric_type));
        dc->code_size = sq.code_size;
        dc->codes = codes_8bit.data();

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            dc->set_query(x + i * d);

            float* cD = coarse_dis.data() + i * k_rerank;
            idx_t* cI = coarse_ids.data() + i * k_rerank;

            // Recompute exact distances for every candidate
            for (idx_t j = 0; j < k_rerank; j++) {
                idx_t id = cI[j];
                if (id >= 0) {
                    if (sel && !sel->is_member(id)) {
                        cI[j] = -1;
                        continue;
                    }
                    cD[j] = dc->query_to_code(
                            codes_8bit.data() + id * sq.code_size);
                }
            }

            // Build final top-k via heap
            float* D = distances + i * k;
            idx_t* I = labels + i * k;

            if (metric_type == METRIC_L2) {
                maxheap_heapify(k, D, I);
                for (idx_t j = 0; j < k_rerank; j++) {
                    if (cI[j] >= 0 && cD[j] < D[0]) {
                        maxheap_replace_top(k, D, I, cD[j], cI[j]);
                    }
                }
                maxheap_reorder(k, D, I);
            } else {
                minheap_heapify(k, D, I);
                for (idx_t j = 0; j < k_rerank; j++) {
                    if (cI[j] >= 0 && cD[j] > D[0]) {
                        minheap_replace_top(k, D, I, cD[j], cI[j]);
                    }
                }
                minheap_reorder(k, D, I);
            }
        }
    }
}

// -----------------------------------------------------------------------
// compute_codes  — called by IndexFastScan::add
// -----------------------------------------------------------------------

void IndexSQFastScan::compute_codes(uint8_t* out_codes, idx_t n, const float* x)
        const {
    FAISS_THROW_IF_NOT_MSG(
            !is_fallback(sq.qtype),
            "compute_codes should not be called for fallback SQ types");
    if (needs_rerank(sq.qtype)) {
        // Go directly from float vectors to 4-bit nibble-packed codes.
        // (The original full-precision codes were already saved by add().)
        float_to_4bit_nibbles(x, out_codes, n, d, M2, sq);
    } else {
        // Native 4-bit: SQ produces nibble-packed output directly
        sq.compute_codes(x, out_codes, n);
    }
}

// -----------------------------------------------------------------------
// compute_float_LUT  — builds the d × 16 distance LUT for vpshufb
// -----------------------------------------------------------------------

void IndexSQFastScan::compute_float_LUT(
        float* lut,
        idx_t n,
        const float* x,
        const FastScanDistancePostProcessing&) const {
    const size_t dim = d;
    const size_t ksub = 16;

    // All types use 16 centroids per dim after (re-)quantisation to 4-bit.
    // Centroid c reconstructs to  vmin + (c + 0.5)/15 · vdiff.
    bool is_uniform = is_uniform_range(sq.qtype) ||
            sq.qtype == ScalarQuantizer::QT_4bit_uniform;

    std::vector<float> recon_table(dim * ksub);

    if (is_uniform) {
        float vmin, vdiff;
        get_uniform_range(sq, vmin, vdiff);
        for (size_t c = 0; c < ksub; c++) {
            float recon = vmin + ((c + 0.5f) / 15.0f) * vdiff;
            for (size_t m = 0; m < dim; m++) {
                recon_table[m * ksub + c] = recon;
            }
        }
    } else {
        const float* vmin = sq.trained.data();
        const float* vdiff = sq.trained.data() + dim;
        for (size_t m = 0; m < dim; m++) {
            for (size_t c = 0; c < ksub; c++) {
                recon_table[m * ksub + c] =
                        vmin[m] + ((c + 0.5f) / 15.0f) * vdiff[m];
            }
        }
    }

    if (metric_type == METRIC_L2) {
        for (idx_t i = 0; i < n; i++) {
            const float* xi = x + i * dim;
            float* lut_i = lut + i * dim * ksub;
            for (size_t m = 0; m < dim; m++) {
                float qi = xi[m];
                const float* recon_m = recon_table.data() + m * ksub;
                float* lut_m = lut_i + m * ksub;
                for (size_t c = 0; c < ksub; c++) {
                    float diff = qi - recon_m[c];
                    lut_m[c] = diff * diff;
                }
            }
        }
    } else {
        for (idx_t i = 0; i < n; i++) {
            const float* xi = x + i * dim;
            float* lut_i = lut + i * dim * ksub;
            for (size_t m = 0; m < dim; m++) {
                float qi = xi[m];
                const float* recon_m = recon_table.data() + m * ksub;
                float* lut_m = lut_i + m * ksub;
                for (size_t c = 0; c < ksub; c++) {
                    lut_m[c] = qi * recon_m[c];
                }
            }
        }
    }
}

void IndexSQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    sq.decode(bytes, x, n);
}

void IndexSQFastScan::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    sq.decode(codes_8bit.data() + key * sq.code_size, recons, 1);
}

size_t IndexSQFastScan::sa_code_size() const {
    return sq.code_size;
}

size_t IndexSQFastScan::fast_scan_code_size() const {
    return M2 / 2;
}

// -----------------------------------------------------------------------
// sa_encode — produce original SQ-format codes (not 4-bit packed)
// -----------------------------------------------------------------------

void IndexSQFastScan::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    sq.compute_codes(x, bytes, n);
}

// -----------------------------------------------------------------------
// reconstruct_n — batch reconstruction
// -----------------------------------------------------------------------

void IndexSQFastScan::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(i0 >= 0 && i0 + ni <= ntotal);
    sq.decode(codes_8bit.data() + i0 * sq.code_size, recons, ni);
}

// -----------------------------------------------------------------------
// add_sa_codes — add pre-encoded SQ-format codes directly
// -----------------------------------------------------------------------

void IndexSQFastScan::add_sa_codes(
        idx_t n,
        const uint8_t* code,
        const idx_t* /*xids*/) {
    FAISS_THROW_IF_NOT(is_trained);

    if (needs_rerank(sq.qtype)) {
        // 1. Append to codes_8bit
        size_t cs = sq.code_size;
        codes_8bit.resize((ntotal + n) * cs);
        memcpy(codes_8bit.data() + ntotal * cs, code, n * cs);

        // 2. Decode new codes to float, then add via IndexFastScan
        //    which calls our compute_codes() to re-quantise to 4-bit
        std::vector<float> recon(n * d);
        sq.decode(code, recon.data(), n);
        IndexFastScan::add(n, recon.data());
    } else if (is_fallback(sq.qtype)) {
        // Fallback: just copy codes
        size_t cs = sq.code_size;
        codes_8bit.resize((ntotal + n) * cs);
        memcpy(codes_8bit.data() + ntotal * cs, code, n * cs);
        ntotal += n;
    } else {
        // Native 4-bit: copy codes to codes_8bit, then decode and
        // re-add via IndexFastScan for SIMD packing.
        size_t cs = sq.code_size;
        codes_8bit.resize((ntotal + n) * cs);
        memcpy(codes_8bit.data() + ntotal * cs, code, n * cs);

        std::vector<float> recon(n * d);
        sq.decode(code, recon.data(), n);
        IndexFastScan::add(n, recon.data());
    }
}

// -----------------------------------------------------------------------
// permute_entries — reorder stored vectors
// -----------------------------------------------------------------------

/// Helper: unpack all packed 4-bit codes into flat nibble-packed format
/// using the CodePacker interface.
static void unpack_all_codes(
        const IndexFastScan& idx,
        std::vector<uint8_t>& flat) {
    std::unique_ptr<CodePacker> packer(idx.get_CodePacker());
    size_t code_sz = idx.code_size;
    flat.resize(idx.ntotal * code_sz, 0);
    for (idx_t i = 0; i < idx.ntotal; i++) {
        packer->unpack_1(idx.codes.data(), i, flat.data() + i * code_sz);
    }
}

void IndexSQFastScan::permute_entries(const idx_t* perm) {
    if (is_native_4bit(sq.qtype) || needs_rerank(sq.qtype)) {
        // Permute the packed SIMD codes
        std::vector<uint8_t> flat_old;
        unpack_all_codes(*this, flat_old);

        size_t code_sz = code_size;
        std::vector<uint8_t> flat_new(ntotal2 * code_sz, 0);
        for (idx_t i = 0; i < ntotal; i++) {
            memcpy(flat_new.data() + i * code_sz,
                   flat_old.data() + perm[i] * code_sz,
                   code_sz);
        }
        pq4_pack_codes(
                flat_new.data(), ntotal, M, ntotal2, bbs, M2, codes.get());
    }

    // Permute codes_8bit (populated for all types)
    {
        size_t cs = sq.code_size;
        std::vector<uint8_t> new_codes(ntotal * cs);
        for (idx_t i = 0; i < ntotal; i++) {
            memcpy(new_codes.data() + i * cs,
                   codes_8bit.data() + perm[i] * cs,
                   cs);
        }
        codes_8bit.swap(new_codes);
    }
}

// -----------------------------------------------------------------------
// get_distance_computer / get_FlatCodesDistanceComputer
// -----------------------------------------------------------------------

FlatCodesDistanceComputer* IndexSQFastScan::get_FlatCodesDistanceComputer()
        const {
    ScalarQuantizer::SQDistanceComputer* dc =
            sq.get_distance_computer(metric_type);
    dc->code_size = sq.code_size;
    dc->codes = codes_8bit.data();
    return dc;
}

DistanceComputer* IndexSQFastScan::get_distance_computer() const {
    return get_FlatCodesDistanceComputer();
}

// -----------------------------------------------------------------------
// range_search
// -----------------------------------------------------------------------

void IndexSQFastScan::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(is_trained);
    const IDSelector* sel = params ? params->sel : nullptr;

    // All types have codes_8bit populated, so we use the SQ scanner
    // uniformly. range_search with SIMD accumulators is not implemented
    // in IndexFastScan, so the SQ scanner is used for all paths.

    std::vector<RangeSearchPartialResult*> partial_results(
            omp_get_max_threads());

#pragma omp parallel
    {
        int rank = omp_get_thread_num();
        RangeSearchPartialResult* pres = new RangeSearchPartialResult(result);
        partial_results[rank] = pres;

        std::unique_ptr<InvertedListScanner> scanner(
                sq.select_InvertedListScanner(metric_type, nullptr, true, sel));
        scanner->list_no = 0;

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            scanner->set_query(x + i * d);
            RangeQueryResult& qres = pres->new_result(i);

            for (idx_t j = 0; j < ntotal; j++) {
                if (sel && !sel->is_member(j)) {
                    continue;
                }
                float dis = scanner->distance_to_code(
                        codes_8bit.data() + j * sq.code_size);
                if (metric_type == METRIC_L2) {
                    if (dis < radius) {
                        qres.add(dis, j);
                    }
                } else {
                    if (dis > radius) {
                        qres.add(dis, j);
                    }
                }
            }
        }
    }

    RangeSearchPartialResult::merge(partial_results);
}

// -----------------------------------------------------------------------
// remove_ids — compact codes_8bit, then delegate to IndexFastScan
// -----------------------------------------------------------------------

size_t IndexSQFastScan::remove_ids(const IDSelector& sel) {
    size_t cs = sq.code_size;
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (!sel.is_member(i)) {
            if (i > j) {
                memcpy(codes_8bit.data() + j * cs,
                       codes_8bit.data() + i * cs,
                       cs);
            }
            j++;
        }
    }
    size_t nremove = IndexFastScan::remove_ids(sel);
    codes_8bit.resize(ntotal * cs);
    return nremove;
}

// -----------------------------------------------------------------------
// check_compatible_for_merge — validate qtype in addition to base checks
// -----------------------------------------------------------------------

void IndexSQFastScan::check_compatible_for_merge(
        const Index& otherIndex) const {
    IndexFastScan::check_compatible_for_merge(otherIndex);
    const IndexSQFastScan* other =
            dynamic_cast<const IndexSQFastScan*>(&otherIndex);
    FAISS_THROW_IF_NOT_MSG(other, "merge requires IndexSQFastScan");
    FAISS_THROW_IF_NOT_MSG(
            other->sq.qtype == sq.qtype,
            "merge requires matching ScalarQuantizer types");
}

// -----------------------------------------------------------------------
// merge_from — copy codes_8bit, then delegate to IndexFastScan
// -----------------------------------------------------------------------

void IndexSQFastScan::merge_from(Index& otherIndex, idx_t add_id) {
    check_compatible_for_merge(otherIndex);
    const IndexSQFastScan* other =
            static_cast<const IndexSQFastScan*>(&otherIndex);

    idx_t n0 = ntotal;
    size_t cs = sq.code_size;
    codes_8bit.resize((n0 + other->ntotal) * cs);
    memcpy(codes_8bit.data() + n0 * cs,
           other->codes_8bit.data(),
           other->ntotal * cs);

    IndexFastScan::merge_from(otherIndex, add_id);
}

// -----------------------------------------------------------------------
// search1 — single-query search with custom result handler
// -----------------------------------------------------------------------

void IndexSQFastScan::search1(
        const float* x,
        ResultHandler& handler,
        SearchParameters* params) const {
    const IDSelector* sel = params ? params->sel : nullptr;
    std::unique_ptr<DistanceComputer> dc(get_distance_computer());
    dc->set_query(x);
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel && !sel->is_member(i)) {
            continue;
        }
        float dis = (*dc)(i);
        handler.add_result(dis, i);
    }
}

} // namespace faiss
