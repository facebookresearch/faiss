/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSQFastScan.h>
#include <faiss/impl/sq_fastscan_utils.h>

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/fast_scan/FastScanDistancePostProcessing.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

using namespace faiss::sq_fastscan;

// -----------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------

IndexSQFastScan::IndexSQFastScan(
        int d_in,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        int bbs_in)
        : sq(d_in, qtype) {
    FAISS_THROW_IF_NOT_MSG(
            !is_fallback(qtype),
            "IndexSQFastScan supports the native 4-bit and the reranked "
            "types. QT_fp16 and QT_bf16 have no fast-scan path; use "
            "IndexScalarQuantizer for those.");
    init_fastscan(d_in, d_in, 4, metric, bbs_in);
}

IndexSQFastScan::IndexSQFastScan() = default;

IndexSQFastScan::IndexSQFastScan(const IndexScalarQuantizer& orig, int bbs_in)
        : sq(orig.sq) {
    FAISS_THROW_IF_NOT_MSG(
            !is_fallback(sq.qtype),
            "IndexSQFastScan conversion constructor supports the native "
            "4-bit and the reranked types only.");
    init_fastscan(orig.d, orig.d, 4, orig.metric_type, bbs_in);
    ntotal = orig.ntotal;
    is_trained = orig.is_trained;

    ntotal2 = roundup(ntotal, bbs_in);
    codes.resize(ntotal2 * M2 / 2);
    if (is_native_4bit(sq.qtype)) {
        pq4_pack_codes(
                orig.codes.data(), ntotal, M, ntotal2, bbs_in, M2, codes.get());
    } else {
        // High nibbles feed the scan, low nibbles the rerank.
        std::vector<uint8_t> nibbles(ntotal * (M2 / 2));
        const int bits = sq_bits(sq.qtype), lo_bits = bits - 4;
        const size_t half = sq_rerank_size(d, sq.qtype);
        lo_codes.assign(size_t(ntotal) * half, 0);
        for (idx_t i = 0; i < ntotal; i++) {
            const uint8_t* src = orig.codes.data() + size_t(i) * sq.code_size;
            uint8_t* hi = nibbles.data() + size_t(i) * (M2 / 2);
            BitstringReader bsr(src, sq.code_size);
            BitstringWriter bsw(lo_codes.data() + size_t(i) * half, half);
            for (int m = 0; m < d; m++) {
                const int code = int(bsr.read(bits));
                const int h = code >> lo_bits;
                hi[m / 2] |= (m & 1) ? uint8_t(h << 4) : uint8_t(h);
                bsw.write(code & ((1 << lo_bits) - 1), lo_bits);
            }
        }
        pq4_pack_codes(
                nibbles.data(), ntotal, M, ntotal2, bbs_in, M2, codes.get());
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
    const size_t before = lo_codes.size();
    IndexFastScan::add(n, x);
    if (!needs_rerank(sq.qtype) || lo_codes.size() != before) {
        // IndexFastScan::add blocks a large batch and re-enters this override
        // once per block, so the blocks have already appended the low bits.
        return;
    }
    const int bits = sq_bits(sq.qtype), lo_bits = bits - 4;
    const size_t stride = sq_rerank_size(d, sq.qtype);
    lo_codes.resize(before + size_t(n) * stride, 0);
    const SQRanges ranges(sq, d);
    for (idx_t i = 0; i < n; i++) {
        const float* xi = x + i * d;
        uint8_t* dst = lo_codes.data() + before + size_t(i) * stride;
        BitstringWriter bsw(dst, stride);
        for (int m = 0; m < d; m++) {
            const int code =
                    sq_code_of(xi[m], ranges.vmin(m), ranges.vdiff(m), bits);
            bsw.write(code & ((1 << lo_bits) - 1), lo_bits);
        }
    }
}

void IndexSQFastScan::reset() {
    lo_codes.clear();
    IndexFastScan::reset();
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
    // The kernel applies the selector itself, so no generic-scanner fallback.
    if (!needs_rerank(sq.qtype)) {
        IndexFastScan::search(n, x, k, distances, labels, params);
        return;
    }

    // The 4-bit scan is a shortlist. rerank_factor bounds the recall.
    idx_t k_coarse = idx_t(double(k) * rerank_factor);
    FAISS_THROW_IF_NOT(k_coarse >= k);
    FAISS_THROW_IF_NOT_MSG(
            n <= INT64_MAX / k_coarse, "n * k_coarse would overflow int64");
    std::vector<float> coarse_dis(n * k_coarse);
    std::vector<idx_t> coarse_ids(n * k_coarse);
    IndexFastScan::search(
            n, x, k_coarse, coarse_dis.data(), coarse_ids.data(), params);

    const bool is_max = metric_type == METRIC_L2;
#pragma omp parallel
    {
        std::unique_ptr<CodePacker> packer(get_CodePacker());
        std::vector<uint8_t> hi_buf(M2 / 2);
        std::vector<int> code_buf(d);
        const size_t lo_stride = sq_rerank_size(d, sq.qtype);
        const int maxv = (1 << sq_bits(sq.qtype)) - 1;
        const SQRanges ranges(sq, d);
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            float* D = distances + i * k;
            idx_t* I = labels + i * k;
            if (is_max) {
                maxheap_heapify(k, D, I);
            } else {
                minheap_heapify(k, D, I);
            }
            for (idx_t j = 0; j < k_coarse; j++) {
                idx_t id = coarse_ids[i * k_coarse + j];
                if (id < 0) {
                    continue;
                }
                float dis;
                {
                    // Recombine the two halves into the original code.
                    packer->unpack_1(codes.get(), id, hi_buf.data());
                    split_to_codes(
                            hi_buf.data(),
                            lo_codes.data() + size_t(id) * lo_stride,
                            d,
                            sq.qtype,
                            code_buf.data());
                    const float* xi = x + i * d;
                    float acc = 0;
                    for (int m = 0; m < d; m++) {
                        // Matches Codec*bit::decode_component: (c+0.5)/maxv.
                        const float v = ranges.vmin(m) +
                                ((code_buf[m] + 0.5f) / float(maxv)) *
                                        ranges.vdiff(m);
                        if (is_max) {
                            const float df = xi[m] - v;
                            acc += df * df;
                        } else {
                            acc += xi[m] * v;
                        }
                    }
                    dis = acc;
                }
                if (is_max) {
                    if (dis < D[0]) {
                        maxheap_replace_top(k, D, I, dis, id);
                    }
                } else {
                    if (dis > D[0]) {
                        minheap_replace_top(k, D, I, dis, id);
                    }
                }
            }
            if (is_max) {
                maxheap_reorder(k, D, I);
            } else {
                minheap_reorder(k, D, I);
            }
        }
    }
}

// -----------------------------------------------------------------------
// compute_codes  -- called by IndexFastScan::add
// -----------------------------------------------------------------------

void IndexSQFastScan::compute_codes(uint8_t* out_codes, idx_t n, const float* x)
        const {
    if (is_native_4bit(sq.qtype)) {
        sq.compute_codes(x, out_codes, n);
    } else {
        // The scanned nibble is the top half. lo_codes holds the bottom.
        std::vector<uint8_t> lo_discard(
                size_t(n) * sq_rerank_size(d, sq.qtype));
        float_to_split_codes(x, out_codes, lo_discard.data(), n, d, M2, sq);
    }
}

// -----------------------------------------------------------------------
// compute_float_LUT  -- builds the d x 16 distance LUT for vpshufb
// -----------------------------------------------------------------------

void IndexSQFastScan::compute_float_LUT(
        float* lut,
        idx_t n,
        const float* x,
        const FastScanDistancePostProcessing&) const {
    const size_t dim = d;
    const size_t ksub_val = 16;

    bool is_uniform = (sq.qtype == ScalarQuantizer::QT_4bit_uniform) ||
            is_uniform_range(sq.qtype);

    std::vector<float> recon_table(dim * ksub_val);

    const float lut_step = float(1 << sq_rerank_bits(sq.qtype));
    const float lut_offset = lut_step / 2.0f;
    const float lut_scale = float((1 << sq_bits(sq.qtype)) - 1);

    if (is_uniform) {
        float vmin, vdiff;
        get_uniform_range(sq, vmin, vdiff);
        for (size_t c = 0; c < ksub_val; c++) {
            float recon =
                    vmin + ((c * lut_step + lut_offset) / lut_scale) * vdiff;
            for (size_t m = 0; m < dim; m++) {
                recon_table[m * ksub_val + c] = recon;
            }
        }
    } else {
        // QT_4bit: per-dimension ranges
        const float* vmin = sq.trained.data();
        const float* vdiff = sq.trained.data() + dim;
        for (size_t m = 0; m < dim; m++) {
            for (size_t c = 0; c < ksub_val; c++) {
                recon_table[m * ksub_val + c] = vmin[m] +
                        ((c * lut_step + lut_offset) / lut_scale) * vdiff[m];
            }
        }
    }

    if (metric_type == METRIC_L2) {
        for (idx_t i = 0; i < n; i++) {
            const float* xi = x + i * dim;
            float* lut_i = lut + i * dim * ksub_val;
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
        for (idx_t i = 0; i < n; i++) {
            const float* xi = x + i * dim;
            float* lut_i = lut + i * dim * ksub_val;
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

void IndexSQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    sq.decode(bytes, x, n);
}

void IndexSQFastScan::fill_sa_code(
        const CodePacker& packer,
        idx_t id,
        uint8_t* scratch,
        int* values,
        uint8_t* code_out) const {
    packer.unpack_1(codes.data(), id, scratch);
    if (lo_codes.empty()) {
        memcpy(code_out, scratch, sq.code_size);
        return;
    }
    split_to_codes(
            scratch,
            lo_codes.data() + size_t(id) * sq_rerank_size(d, sq.qtype),
            d,
            sq.qtype,
            values);
    memset(code_out, 0, sq.code_size);
    BitstringWriter out(code_out, sq.code_size);
    const int bits = sq_bits(sq.qtype);
    for (int m = 0; m < d; m++) {
        out.write(values[m], bits);
    }
}

void IndexSQFastScan::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::unique_ptr<CodePacker> packer(get_CodePacker());
    std::vector<uint8_t> scratch(M2 / 2), flat_code(sq.code_size);
    std::vector<int> values(d);
    fill_sa_code(*packer, key, scratch.data(), values.data(), flat_code.data());
    sq.decode(flat_code.data(), recons, 1);
}

size_t IndexSQFastScan::sa_code_size() const {
    return sq.code_size;
}

size_t IndexSQFastScan::fast_scan_code_size() const {
    return M2 / 2;
}

// -----------------------------------------------------------------------
// sa_encode
// -----------------------------------------------------------------------

void IndexSQFastScan::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    sq.compute_codes(x, bytes, n);
}

// -----------------------------------------------------------------------
// reconstruct_n
// -----------------------------------------------------------------------

void IndexSQFastScan::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(i0 >= 0 && i0 + ni <= ntotal);
    std::unique_ptr<CodePacker> packer(get_CodePacker());
    std::vector<uint8_t> scratch(M2 / 2), flat_code(sq.code_size);
    std::vector<int> values(d);
    for (idx_t i = 0; i < ni; i++) {
        fill_sa_code(
                *packer,
                i0 + i,
                scratch.data(),
                values.data(),
                flat_code.data());
        sq.decode(flat_code.data(), recons + i * d, 1);
    }
}

// -----------------------------------------------------------------------
// add_sa_codes
// -----------------------------------------------------------------------

void IndexSQFastScan::add_sa_codes(
        idx_t n,
        const uint8_t* code,
        const idx_t* /*xids*/) {
    FAISS_THROW_IF_NOT(is_trained);
    // Decode to float then re-add via IndexFastScan for SIMD packing
    std::vector<float> recon(n * d);
    sq.decode(code, recon.data(), n);
    add(n, recon.data());
}

// -----------------------------------------------------------------------
// permute_entries
// -----------------------------------------------------------------------

void IndexSQFastScan::permute_entries(const idx_t* perm) {
    std::vector<uint8_t> flat_old;
    {
        std::unique_ptr<CodePacker> packer(get_CodePacker());
        size_t code_sz = code_size;
        flat_old.resize(ntotal * code_sz, 0);
        for (idx_t i = 0; i < ntotal; i++) {
            packer->unpack_1(codes.data(), i, flat_old.data() + i * code_sz);
        }
    }

    size_t code_sz = code_size;
    std::vector<uint8_t> flat_new(ntotal2 * code_sz, 0);
    for (idx_t i = 0; i < ntotal; i++) {
        memcpy(flat_new.data() + i * code_sz,
               flat_old.data() + perm[i] * code_sz,
               code_sz);
    }
    pq4_pack_codes(flat_new.data(), ntotal, M, ntotal2, bbs, M2, codes.get());

    if (!lo_codes.empty()) {
        const size_t lo_sz = sq_rerank_size(d, sq.qtype);
        std::vector<uint8_t> lo_new(size_t(ntotal) * lo_sz);
        for (idx_t i = 0; i < ntotal; i++) {
            memcpy(lo_new.data() + size_t(i) * lo_sz,
                   lo_codes.data() + size_t(perm[i]) * lo_sz,
                   lo_sz);
        }
        lo_codes.swap(lo_new);
    }
}

// -----------------------------------------------------------------------
// get_distance_computer / get_FlatCodesDistanceComputer
// -----------------------------------------------------------------------

namespace {

/// Wrapper that owns an unpacked code buffer and delegates distance
/// computation to an underlying SQDistanceComputer.
struct OwningFlatCodesDistanceComputer : FlatCodesDistanceComputer {
    std::vector<uint8_t> owned_codes;
    std::unique_ptr<ScalarQuantizer::SQDistanceComputer> dc;

    OwningFlatCodesDistanceComputer(
            std::vector<uint8_t>&& codes_buf,
            ScalarQuantizer::SQDistanceComputer* dc_in)
            : owned_codes(std::move(codes_buf)), dc(dc_in) {
        this->codes = owned_codes.data();
        this->code_size = dc->code_size;
        dc->codes = owned_codes.data();
    }

    void set_query(const float* x) override {
        dc->set_query(x);
    }

    float distance_to_code(const uint8_t* code) override {
        return dc->distance_to_code(code);
    }

    void distance_to_code_batch_4(
            const uint8_t* c0,
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            float& d0,
            float& d1,
            float& d2,
            float& d3) override {
        dc->distance_to_code_batch_4(c0, c1, c2, c3, d0, d1, d2, d3);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return dc->symmetric_dis(i, j);
    }
};

} // anonymous namespace

FlatCodesDistanceComputer* IndexSQFastScan::get_FlatCodesDistanceComputer()
        const {
    std::unique_ptr<CodePacker> packer(get_CodePacker());
    size_t cs = sq.code_size;
    std::vector<uint8_t> flat(ntotal * cs);
    std::vector<uint8_t> scratch(M2 / 2);
    std::vector<int> values(d);
    for (idx_t i = 0; i < ntotal; i++) {
        fill_sa_code(
                *packer,
                i,
                scratch.data(),
                values.data(),
                flat.data() + i * cs);
    }

    ScalarQuantizer::SQDistanceComputer* dc =
            sq.get_distance_computer(metric_type);
    dc->code_size = cs;

    return new OwningFlatCodesDistanceComputer(std::move(flat), dc);
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

    // Unpack codes for range search
    std::vector<uint8_t> flat_codes(ntotal * sq.code_size);
    {
        std::unique_ptr<CodePacker> packer(get_CodePacker());
        std::vector<uint8_t> scratch(M2 / 2);
        std::vector<int> values(d);
        for (idx_t i = 0; i < ntotal; i++) {
            fill_sa_code(
                    *packer,
                    i,
                    scratch.data(),
                    values.data(),
                    flat_codes.data() + i * sq.code_size);
        }
    }

    std::vector<RangeSearchPartialResult*> partial_results(
            omp_get_max_threads());

    size_t cs = sq.code_size;

#pragma omp parallel
    {
        int rank = omp_get_thread_num();
        RangeSearchPartialResult* pres = new RangeSearchPartialResult(result);
        partial_results[rank] = pres;

        std::vector<float> decoded(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const float* qi = x + i * d;
            RangeQueryResult& qres = pres->new_result(i);

            for (idx_t j = 0; j < ntotal; j++) {
                if (sel && !sel->is_member(j)) {
                    continue;
                }
                sq.decode(flat_codes.data() + j * cs, decoded.data(), 1);
                float dis;
                if (metric_type == METRIC_INNER_PRODUCT) {
                    dis = fvec_inner_product(qi, decoded.data(), d);
                    if (dis > radius) {
                        qres.add(dis, j);
                    }
                } else {
                    dis = fvec_L2sqr(qi, decoded.data(), d);
                    if (dis < radius) {
                        qres.add(dis, j);
                    }
                }
            }
        }
    }

    RangeSearchPartialResult::merge(partial_results);
}

// -----------------------------------------------------------------------
// remove_ids
// -----------------------------------------------------------------------

size_t IndexSQFastScan::remove_ids(const IDSelector& sel) {
    return IndexFastScan::remove_ids(sel);
}

// -----------------------------------------------------------------------
// check_compatible_for_merge
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
// merge_from
// -----------------------------------------------------------------------

void IndexSQFastScan::merge_from(Index& otherIndex, idx_t add_id) {
    check_compatible_for_merge(otherIndex);
    const IndexSQFastScan* other =
            static_cast<const IndexSQFastScan*>(&otherIndex);
    lo_codes.insert(
            lo_codes.end(), other->lo_codes.begin(), other->lo_codes.end());
    IndexFastScan::merge_from(otherIndex, add_id);
}

// -----------------------------------------------------------------------
// search1
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
