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
 *     PQ4 FastScan SIMD (vpshufb / equivalent).
 *
 *   - QT_6bit, QT_8bit, QT_8bit_uniform, QT_8bit_direct,
 *     QT_8bit_direct_signed: re-quantised to 4-bit for the scan, reranked
 *     against the originals in orig_codes.
 *
 *   - All other types (QT_fp16, QT_bf16): use IndexScalarQuantizer directly.
 */
struct IndexSQFastScan : IndexFastScan {
    ScalarQuantizer sq;

    /// Overselection ratio for reranking.
    float rerank_factor = 2;

    /// Low nibbles of the 8-bit codes, two dimensions per byte, indexed by
    /// id. The high nibbles live in the packed scan codes. Empty for the
    /// native 4-bit types, which need no rerank.
    std::vector<uint8_t> lo_codes;

    /** Constructor.
     *
     * @param d       dimensionality of input vectors
     * @param qtype   any native 4-bit or reranked type (see above)
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

    void fill_sa_code(
            const CodePacker& packer,
            idx_t id,
            uint8_t* scratch,
            int* values,
            uint8_t* code_out) const;

    size_t fast_scan_code_size() const override;
};

} // namespace faiss
