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

#include <faiss/IndexIVF.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/fast_scan/FastScanDistancePostProcessing.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/impl/fast_scan/sq_fastscan_lut.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
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

} // anonymous namespace

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
            is_native_4bit(qtype),
            "IndexSQFastScan only supports QT_4bit and QT_4bit_uniform. "
            "For higher-precision types, use "
            "IndexRefine(IndexSQFastScan(...), IndexScalarQuantizer(...)).");
    // M = d subquantizers accumulate into uint16 SIMD registers, each LUT
    // entry is a uint8 in [0, 255], so d * 255 must fit in a uint16.
    FAISS_THROW_IF_NOT_MSG(
            d_in <= 257,
            "IndexSQFastScan supports at most d = 257: the uint16 fast-scan "
            "accumulators would overflow for larger dimensions.");
    init_fastscan(d_in, d_in, 4, metric, bbs_in);
}

IndexSQFastScan::IndexSQFastScan() = default;

IndexSQFastScan::IndexSQFastScan(const IndexScalarQuantizer& orig, int bbs_in)
        : sq(orig.sq) {
    FAISS_THROW_IF_NOT_MSG(
            is_native_4bit(sq.qtype),
            "IndexSQFastScan conversion constructor only supports "
            "QT_4bit and QT_4bit_uniform.");
    // See the primary constructor: d * 255 must fit in a uint16 accumulator.
    FAISS_THROW_IF_NOT_MSG(
            orig.d <= 257,
            "IndexSQFastScan supports at most d = 257: the uint16 fast-scan "
            "accumulators would overflow for larger dimensions.");
    init_fastscan(orig.d, orig.d, 4, orig.metric_type, bbs_in);
    ntotal = orig.ntotal;
    is_trained = orig.is_trained;

    ntotal2 = roundup(ntotal, bbs_in);
    codes.resize(ntotal2 * M2 / 2);
    pq4_pack_codes(
            orig.codes.data(), ntotal, M, ntotal2, bbs_in, M2, codes.get());
}

// -----------------------------------------------------------------------
// train / add
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
    IndexFastScan::add(n, x);
}

// Note: no search() override. IndexFastScan::search is inherited; it throws
// on any SearchParameters (including an IDSelector), matching IndexPQFastScan.
// Selector-based search would require decoding the whole packed index, so use
// IndexScalarQuantizer for that instead.

// -----------------------------------------------------------------------
// compute_codes  -- called by IndexFastScan::add
// -----------------------------------------------------------------------

void IndexSQFastScan::compute_codes(uint8_t* out_codes, idx_t n, const float* x)
        const {
    sq.compute_codes(x, out_codes, n);
}

// -----------------------------------------------------------------------
// compute_float_LUT  -- builds the d x 16 distance LUT for vpshufb
// -----------------------------------------------------------------------

void IndexSQFastScan::compute_float_LUT(
        float* lut,
        idx_t n,
        const float* x,
        const FastScanDistancePostProcessing&) const {
    std::vector<float> recon_table;
    sq_fastscan::build_recon_table(sq, d, recon_table);
    sq_fastscan::fill_lut(recon_table.data(), x, lut, n, d, metric_type);
}

void IndexSQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    sq.decode(bytes, x, n);
}

void IndexSQFastScan::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::unique_ptr<CodePacker> packer(get_CodePacker());
    std::vector<uint8_t> flat_code(sq.code_size);
    packer->unpack_1(codes.data(), key, flat_code.data());
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
    std::vector<uint8_t> flat_code(sq.code_size);
    for (idx_t i = 0; i < ni; i++) {
        packer->unpack_1(codes.data(), i0 + i, flat_code.data());
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
    IndexFastScan::add(n, recon.data());
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
}

// Note: no get_distance_computer() override. Index::get_distance_computer is
// inherited: for METRIC_L2 it returns a GenericDistanceComputer that
// reconstructs vectors on demand via IndexFastScan::reconstruct (packed-layout
// aware), so nothing is decoded eagerly; for other metrics it throws. This
// matches IndexPQFastScan, which also does not implement a random-access
// distance computer over the packed codes.

// Note: no range_search() override; Index::range_search (throws) is inherited,
// as in IndexPQFastScan. Range search would require decoding the whole packed
// index; use IndexScalarQuantizer if range search is needed.

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
    IndexFastScan::merge_from(otherIndex, add_id);
}

// Note: no search1() override; Index::search1 (throws) is inherited, as in
// IndexPQFastScan. It depended on the removed get_distance_computer().

} // namespace faiss
