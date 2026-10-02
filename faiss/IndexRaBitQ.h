/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/RaBitQuantizer.h>

namespace faiss {

enum RaBitQFullCodeMode : uint8_t {
    RABITQ_FULL_CODE_PACKED = 0,
    RABITQ_FULL_CODE_EXPANDED = 1,
    RABITQ_FULL_CODE_INT8 = 2,
    RABITQ_FULL_CODE_INT8_PACKED = 3,
    RABITQ_FULL_CODE_PROGRESSIVE = 4,
    // Research mode: keep the authoritative multibit row layout but score
    // only its shared sign prefix during graph navigation.
    RABITQ_NAVIGATION_1BIT_PACKED = 5,
    RABITQ_SPLIT4_NAVIGATION = 6,
    RABITQ_SPLIT4_ADC = 7,
    // Two-bit navigation prefix plus a coarse-conditioned 32-entry magnitude
    // codebook. The 5-bit local code is stored as low4 + high1 planes.
    RABITQ_FULL_CODE_NESTED_LUT7 = 8,
    // Four-bit nested code: a two-bit navigation prefix followed by a
    // two-bit local code. Full scores use a shared 2x4 magnitude LUT.
    RABITQ_FULL_CODE_NESTED_LUT4 = 9,
    // The same logical nested-LUT4 code stored as one contiguous nibble per
    // coordinate and used for full-code navigation.
    RABITQ_FULL_CODE_NIBBLE_LUT4 = 10,
};

struct RaBitQSearchParameters : SearchParameters {
    uint8_t qb = 4;
    bool centered = false;
};

struct IndexRaBitQ : IndexFlatCodes {
    RaBitQuantizer rabitq;

    // center of all points
    std::vector<float> center;

    // the default number of bits to quantize a query with.
    // use '0' to disable quantization and use raw fp32 values.
    // Note: qb=0 is NOT supported by FastScan variants, which require
    // quantized queries for SIMD lookup table construction.
    uint8_t qb = 4;

    // quantize the query with a zero-centered scalar quantizer.
    bool centered = false;

    /** Optional runtime cache of existing RaBitQ full levels.
     *
     * Each row contains d signed level bytes followed by the original
     * ExtraBitsFactors. Packed codes remain authoritative and unchanged.
     * The cache and selected mode are deliberately not serialized.
     */
    std::vector<uint8_t> expanded_codes;
    // Experimental four-bit split layout. Sign rows also contain the two
    // factors needed by one-bit navigation; tail rows contain three packed
    // magnitude bits and the two exact full-score factors. Together they use
    // 4*d/8 + 16 bytes per vector. A 12-byte factor variant changes fp32
    // rounding and must be evaluated separately.
    std::vector<uint8_t> split4_sign_codes;
    std::vector<uint8_t> split4_tail_codes;
    // Two ordered 32-entry magnitude codebooks, one per RQ2 coarse cell.
    std::vector<uint8_t> nested_lut7;
    // Two ordered four-entry magnitude codebooks, one per RQ2 coarse cell.
    // Both LUT4 physical layouts use this exact same logical codebook.
    std::vector<uint8_t> nested_lut4;
    // Runtime policy: use the full nested scorer during HNSW navigation.
    // The serialized row layout is unchanged; the default keeps RQ2 staging.
    bool nested_lut7_full_navigation = false;
    // Experimental zero-storage-cost middle stage. It reads the RQ2 prefix
    // plus the already stored high local bit (3 bits/dimension total).
    bool nested_lut7_mid_navigation = false;
    // Experimental exact-level variant for full-score navigation. The prefix
    // stores sign plus the high RQ7 magnitude bit and local5 stores the rest.
    bool nested_lut7_exact_encoding = false;
    // Experimental high-dimensional staged navigation. The sidecar stores
    // ||A_full * level_full - A_prefix * level_prefix|| for a probabilistic
    // projection bound. It is runtime-only and is not serialized.
    bool nested_adaptive_navigation = false;
    float nested_adaptive_sigma = 2.0f;
    std::vector<float> nested_adaptive_error_norms;
    RaBitQFullCodeMode full_code_mode = RABITQ_FULL_CODE_PACKED;

    IndexRaBitQ();

    explicit IndexRaBitQ(
            idx_t d,
            MetricType metric = METRIC_L2,
            uint8_t nb_bits = 1);

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;
    void add_sa_codes(idx_t n, const uint8_t* x, const idx_t* xids) override;
    void reset() override;
    size_t remove_ids(const IDSelector& sel) override;
    void merge_from(Index& other_index, idx_t add_id = 0) override;
    void permute_entries(const idx_t* perm);

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    // returns a quantized-to-qb bits DC if qb > 0
    // returns a default fp32-based DC if qb == 0
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    // returns a quantized-to-qb bits DC if qb_in > 0
    // returns a default fp32-based DC if qb_in == 0
    FlatCodesDistanceComputer* get_quantized_distance_computer(
            const uint8_t qb_in,
            bool centered) const;

    /** Select the full-code scorer. Integer modes support L2 and 2..8 total
     * RaBitQ bits. RABITQ_FULL_CODE_INT8 derives a d+8 byte cache per vector;
     * RABITQ_FULL_CODE_INT8_PACKED instead expands only the current scoring
     * batch into distance-computer-local scratch space.
     */
    void set_full_code_mode(uint8_t mode);

    size_t expanded_code_size() const;
    void rebuild_expanded_codes();
    bool expanded_integer_uses_native_dotprod() const;

    /** Convert authoritative legacy RQ4 rows into compact split planes and
     * release the legacy rows. Serialization is intentionally unsupported in
     * this research mode.
     */
    void prepare_split4_layout();

    /** Score an arbitrary candidate matrix with the selected full-code mode.
     * labels has n*k storage IDs and distances receives n*k scores. This is
     * intended for a small reranking set produced by a cheaper navigator.
     */
    void compute_distance_subset(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            const idx_t* labels) const;

    /** Materialize the logical two-bit prefix of this multibit code into an
     * empty 2-bit IndexRaBitQ. This is a research bridge for validating a
     * future progressive layout where the same prefix is read in place.
     */
    void derive_2bit_prefix(IndexRaBitQ& destination) const;

    /** Materialize the shared sign-bit navigation code of a multibit index.
     * This is exact: multibit RaBitQ and one-bit RaBitQ use the same sign code
     * and the same first two per-vector factors.
     */
    void derive_1bit_prefix(IndexRaBitQ& destination) const;

    /** Legacy packed rows cannot be losslessly converted to the residual
     * progressive layout because the original vectors are required.
     */
    void finalize_progressive_layout();

    /** Select nested progressive encoding for an empty trained index. New
     * vectors are encoded with a two-bit-optimized prefix and an embedded
     * refinement tail.
     */
    void prepare_progressive_layout();

    /** Select the 7-bit nested LUT layout for an empty trained index. It
     * retains the independently optimized 2-bit navigation prefix while
     * storing the full scorer in 7*d/8 + 12 bytes per vector.
     */
    void prepare_nested_lut7_layout();

    /** Select compact four-bit nested layouts. Both layouts encode identical
     * logical levels. The staged layout navigates with its two-bit prefix;
     * the nibble layout navigates with full four-bit table-lookup ADC.
     */
    void prepare_nested_lut4_layout();
    void prepare_nibble_lut4_layout();

    /** Build the runtime error-norm sidecar used by adaptive nested-LUT7
     * navigation. Existing packed rows remain unchanged.
     */
    void prepare_nested_adaptive_navigation(float sigma = 2.0f);

    // Don't rely on sa_decode(), bcz it is good for IP, but not for L2.
    //   As a result, use get_FlatCodesDistanceComputer() for the search.
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss
