/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <faiss/IndexFastScan.h>
#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

/** Fast scan version of IndexScalarQuantizer.
 *
 * Supported quantizer types:
 *   - QT_4bit, QT_4bit_uniform: native 4-bit codes mapped onto the
 *     PQ4 FastScan SIMD infrastructure (vpshufb / equivalent).
 *     No precision loss.
 *   - QT_6bit, QT_8bit, QT_8bit_uniform, QT_8bit_direct,
 *     QT_8bit_direct_signed: re-quantised to 4-bit for the vpshufb
 *     fast scan (first pass), then reranked with exact
 *     original-precision distances (second pass).  Near-full
 *     precision with ~7–9× speedup.
 *   - All other types (QT_fp16, QT_bf16, …): fall back to the
 *     ScalarQuantizer's own SIMD-optimised scanner.  Same speed as
 *     IndexScalarQuantizer — no fast-scan benefit, but the unified
 *     interface is preserved.
 *
 * For 4-bit types, M = d subquantizers with 16 levels and uint16
 * SIMD accumulators (safe for d <= 257).
 *
 * For reranked types, both original codes (for reranking)
 * and packed 4-bit codes (for the SIMD scan) are stored.  The
 * overselection ratio is controlled by rerank_factor (default 4).
 */
struct IndexSQFastScan : IndexFastScan {
    ScalarQuantizer sq;

    /// Original full-precision codes (ntotal * sq.code_size bytes).
    /// Populated for ALL types: used for reranking, reconstruction,
    /// distance computation (HNSW), and range search.
    std::vector<uint8_t> codes_8bit;

    /// Overselection ratio for reranking.  The 4-bit fast scan
    /// retrieves k * rerank_factor candidates, then exact
    /// original-precision distances select the final top-k.
    /// Higher = better recall, slower.  Default 4.
    int rerank_factor = 4;

    /** Constructor.
     *
     * @param d       dimensionality of input vectors
     * @param qtype   any ScalarQuantizer::QuantizerType
     * @param metric  distance metric (METRIC_L2 or METRIC_INNER_PRODUCT)
     * @param bbs     block size for SIMD processing (multiple of 32)
     */
    IndexSQFastScan(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexSQFastScan();

    /// Build from an existing IndexScalarQuantizer
    explicit IndexSQFastScan(const IndexScalarQuantizer& orig, int bbs = 32);

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void compute_codes(uint8_t* codes, idx_t n, const float* x) const override;

    /// Validate compatibility before merge (checks qtype match).
    void check_compatible_for_merge(const Index& otherIndex) const override;

    void compute_float_LUT(
            float* lut,
            idx_t n,
            const float* x,
            const FastScanDistancePostProcessing& context) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /// Encode float vectors to SQ-format codes (not 4-bit packed).
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /// Batch reconstruction: decode contiguous range [i0, i0+ni).
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    /// Add pre-encoded SQ-format codes directly.
    void add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids)
            override;

    /// Reorder stored vectors by permutation.
    void permute_entries(const idx_t* perm);

    /// Distance computer operating on codes_8bit (for HNSW etc.).
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;

    DistanceComputer* get_distance_computer() const override;

    /// Range search with radius threshold.
    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    /// Remove vectors by selector; keeps codes_8bit in sync.
    size_t remove_ids(const IDSelector& sel) override;

    /// Merge another IndexSQFastScan into this one.
    void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    /// Single-query search with custom result handler.
    void search1(
            const float* x,
            ResultHandler& handler,
            SearchParameters* params = nullptr) const override;

    /// Standalone codec size: returns SQ code size (not 4-bit packed).
    size_t sa_code_size() const override;

    /// Packed code size: d dimensions / 2 (4-bit nibbles)
    size_t fast_scan_code_size() const override;
};

} // namespace faiss
