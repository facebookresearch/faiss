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

/** Fast scan version of IndexScalarQuantizer for native 4-bit types.
 *
 * Supported quantizer types:
 *   - QT_4bit, QT_4bit_uniform: native 4-bit codes mapped onto the
 *     PQ4 FastScan SIMD infrastructure (vpshufb / equivalent).
 *     No precision loss.
 *
 * For higher-precision types (QT_8bit, QT_6bit, etc.), use
 * IndexRefine(IndexSQFastScan(QT_4bit), IndexScalarQuantizer(QT_8bit))
 * to get fast-scan with reranking.  For fallback types (QT_fp16,
 * QT_bf16), use IndexScalarQuantizer directly.
 *
 * M = d subquantizers with 16 levels and uint16 SIMD accumulators
 * (safe for d <= 257).
 */
struct IndexSQFastScan : IndexFastScan {
    ScalarQuantizer sq;

    /** Constructor.
     *
     * @param d       dimensionality of input vectors
     * @param qtype   QT_4bit or QT_4bit_uniform only
     * @param metric  distance metric (METRIC_L2 or METRIC_INNER_PRODUCT)
     * @param bbs     block size for SIMD processing (multiple of 32)
     */
    IndexSQFastScan(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexSQFastScan();

    /// Build from an existing IndexScalarQuantizer (must be QT_4bit*)
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

    void check_compatible_for_merge(const Index& otherIndex) const override;

    void compute_float_LUT(
            float* lut,
            idx_t n,
            const float* x,
            const FastScanDistancePostProcessing& context) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids)
            override;

    void permute_entries(const idx_t* perm);

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;

    DistanceComputer* get_distance_computer() const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    size_t remove_ids(const IDSelector& sel) override;

    void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    void search1(
            const float* x,
            ResultHandler& handler,
            SearchParameters* params = nullptr) const override;

    size_t sa_code_size() const override;

    size_t fast_scan_code_size() const override;
};

} // namespace faiss
