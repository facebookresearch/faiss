/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVFFastScan.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/sq_fastscan_utils.h>

namespace faiss {

/** Fast scan version of IndexIVFScalarQuantizer.
 *
 * Uses the PQ4 FastScan SIMD infrastructure (vpshufb) for accelerated
 * scanning within inverted lists.  Supported quantizer types:
 *
 *   - QT_4bit, QT_4bit_uniform: native 4-bit codes are packed directly
 *     into the SIMD block layout.  No precision loss.
 *
 *   - QT_6bit, QT_8bit, QT_8bit_uniform, QT_8bit_direct,
 *     QT_8bit_direct_signed: re-quantised to 4-bit for the SIMD scan,
 *     then reranked with exact original-precision distances.
 *
 *   - All other types (QT_fp16, QT_bf16, TurboQuant): fall back to the
 *     ScalarQuantizer's own SIMD-optimised InvertedListScanner (same
 *     behavior as IndexIVFScalarQuantizer, no fast-scan acceleration).
 *
 * For reranked types the block inverted lists hold the packed high nibbles
 * and a parallel ArrayInvertedLists (lo_codes_invlists) holds the low ones.
 */
struct IndexIVFSQFastScan : IndexIVFFastScan {
    ScalarQuantizer sq;

    /// Overselection ratio for reranking (default 2).
    float rerank_factor = 2;

    /// Parallel inverted lists holding the leftover low bits of each code.
    /// The top 4 bits of every code live in the packed scan blocks.
    /// Owned by this index. nullptr for native 4-bit and fallback types.
    InvertedLists* lo_codes_invlists = nullptr;

    /// Bytes per vector in lo_codes_invlists: the bits of each code left
    /// over once the scan has taken the top 4.
    size_t lo_code_size() const {
        return sq_fastscan::sq_rerank_size(d, sq.qtype);
    }

    /// Rebuild the code at (list_no, offset) from its two halves. `scratch`
    /// needs M2 / 2 bytes, `code_out` needs sq.code_size.
    void recombine_code(
            int64_t list_no,
            int64_t offset,
            uint8_t* scratch,
            int* values,
            uint8_t* code_out) const;

    IndexIVFSQFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            int bbs = 32,
            bool by_residual = true);

    IndexIVFSQFastScan();

    /// Build from an existing IndexIVFScalarQuantizer.
    explicit IndexIVFSQFastScan(
            const IndexIVFScalarQuantizer& orig,
            int bbs = 32);

    ~IndexIVFSQFastScan() override;

    size_t fast_scan_code_size() const override;

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listno = false) const override;

    bool lookup_table_is_3d() const override;

    void compute_LUT(
            size_t n,
            const float* x,
            const CoarseQuantized& cq,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases,
            const FastScanDistancePostProcessing& context) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters*) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void reset() override;
};

} // namespace faiss
