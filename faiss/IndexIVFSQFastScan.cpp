/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFSQFastScan.h>

#include <cstring>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/impl/fast_scan/sq_fastscan_lut.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

namespace {

size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

bool is_native_4bit(ScalarQuantizer::QuantizerType qtype) {
    return qtype == ScalarQuantizer::QT_4bit ||
            qtype == ScalarQuantizer::QT_4bit_uniform;
}

#define SQFS_UNSUPPORTED_MSG                                            \
    "IndexIVFSQFastScan only supports QT_4bit and QT_4bit_uniform. "    \
    "For higher-precision types, wrap a 4-bit IndexIVFSQFastScan with " \
    "IndexRefine, or use IndexIVFScalarQuantizer directly."

#define SQFS_OVERFLOW_MSG                                                \
    "IndexIVFSQFastScan supports at most d = 257: the uint16 fast-scan " \
    "accumulators would overflow for larger dimensions."

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
        int bbs_in)
        : IndexIVFFastScan(quantizer_in, d_in, nlist_in, 0, metric, false),
          sq(d_in, qtype) {
    FAISS_THROW_IF_NOT_MSG(is_native_4bit(qtype), SQFS_UNSUPPORTED_MSG);
    FAISS_THROW_IF_NOT_MSG(d_in <= 257, SQFS_OVERFLOW_MSG);
    // by_residual=false because the 2D LUT cannot handle per-probe residuals
    // (same rationale as IndexIVFPQFastScan).
    by_residual = false;
    init_fastscan(&sq, d_in, 4, nlist_in, metric, bbs_in, true);
}

IndexIVFSQFastScan::IndexIVFSQFastScan() {
    by_residual = false;
    bbs = 0;
    M2 = 0;
}

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
    FAISS_THROW_IF_NOT_MSG(is_native_4bit(sq.qtype), SQFS_UNSUPPORTED_MSG);
    FAISS_THROW_IF_NOT_MSG(orig.d <= 257, SQFS_OVERFLOW_MSG);
    FAISS_THROW_IF_NOT_MSG(
            !orig.by_residual,
            "IndexIVFSQFastScan: conversion from IndexIVFScalarQuantizer with "
            "by_residual=true is not supported. Set orig.by_residual=false and "
            "retrain, or construct IndexIVFSQFastScan directly.");

    by_residual = false;
    init_fastscan(&sq, orig.d, 4, orig.nlist, orig.metric_type, bbs_in, true);

    ntotal = orig.ntotal;
    is_trained = orig.is_trained;
    nprobe = orig.nprobe;

    // Native 4-bit SQ codes already use the nibble layout expected by
    // pq4_pack_codes, so re-pack them directly into the SIMD block layout.
    BlockInvertedLists* bil = dynamic_cast<BlockInvertedLists*>(invlists);
    FAISS_THROW_IF_NOT(bil);

    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = orig.invlists->list_size(list_no);
        if (list_size == 0) {
            continue;
        }
        InvertedLists::ScopedCodes orig_codes(orig.invlists, list_no);
        InvertedLists::ScopedIds orig_ids(orig.invlists, list_no);

        size_t nb2 = roundup(list_size, bbs);
        AlignedTable<uint8_t> packed(nb2 * M2 / 2);
        pq4_pack_codes(
                orig_codes.get(), list_size, M, nb2, bbs, M2, packed.get());
        bil->add_entries(list_no, list_size, orig_ids.get(), packed.get());
    }
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
    // by_residual is always false, so encode the raw vectors directly.
    sq.compute_codes(x, codes_out, n);

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
        const CoarseQuantized& /*cq*/,
        AlignedTable<float>& dis_tables,
        AlignedTable<float>& /*biases*/,
        const FastScanDistancePostProcessing&) const {
    // by_residual is always false, so codes store raw quantized values and the
    // LUT computes distances directly from the raw query.  No biases needed.
    std::vector<float> recon_table;
    sq_fastscan::build_recon_table(sq, d, recon_table);

    dis_tables.resize(n * d * sq_fastscan::ksub);
    sq_fastscan::fill_lut(
            recon_table.data(), x, dis_tables.get(), n, d, metric_type);
}

} // namespace faiss
