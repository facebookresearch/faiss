/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <faiss/IndexFastScan.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/impl/RaBitQStats.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/simdlib.h>

namespace faiss {

// Import shared utilities from RaBitQUtils
using rabitq_utils::ExtraBitsFactors;
using rabitq_utils::QueryFactorsData;
using rabitq_utils::SignBitFactors;
using rabitq_utils::SignBitFactorsWithError;

/** Fast-scan version of RaBitQ index that processes 32 database vectors at a
 * time using SIMD operations. Similar to IndexPQFastScan but adapted for
 * RaBitQ's bit-level quantization with factors.
 *
 * The key differences from IndexRaBitQ:
 * - Processes vectors in batches of 32
 * - Uses 4-bit groupings for SIMD optimization (4 dimensions per 4-bit unit)
 * - Separates factors from quantized bits for efficient processing
 * - Leverages existing PQ4 FastScan infrastructure where possible
 */
struct IndexRaBitQFastScan : IndexFastScan {
    /// RaBitQ quantizer for encoding/decoding
    RaBitQuantizer rabitq;

    /// Center of all points (same as IndexRaBitQ)
    std::vector<float> center;

    /// Per-vector auxiliary data (1-bit codes stored separately in `codes`)
    ///
    /// 1-bit codes (sign bits) are stored in the inherited `codes` array from
    /// IndexFastScan in packed FastScan format for SIMD processing.
    ///
    /// This flat_storage holds per-vector factors and refinement-bit codes:
    /// Layout for 1-bit: [SignBitFactors (8 bytes)]
    /// Layout for multi-bit: [SignBitFactorsWithError
    /// (12B)][ref_codes][ExtraBitsFactors (8B)]
    std::vector<uint8_t> flat_storage;

    /// Default number of bits to quantize a query with
    uint8_t qb = 8;

    // quantize the query with a zero-centered scalar quantizer.
    bool centered = false;

    IndexRaBitQFastScan();

    explicit IndexRaBitQFastScan(
            idx_t d,
            MetricType metric = METRIC_L2,
            int bbs = 32,
            uint8_t nb_bits = 1);

    /// build from an existing IndexRaBitQ
    explicit IndexRaBitQFastScan(const IndexRaBitQ& orig, int bbs = 32);

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void compute_codes(uint8_t* codes, idx_t n, const float* x) const override;

    /// Compute storage size per vector in flat_storage
    size_t compute_per_vector_storage_size() const;

    void compute_float_LUT(
            float* lut,
            idx_t n,
            const float* x,
            const FastScanDistancePostProcessing& context) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Override to create RaBitQ-specific handlers
    SIMDResultHandlerToFloat* make_knn_handler(
            bool is_max,
            int /*impl*/,
            idx_t n,
            idx_t k,
            size_t /*ntotal*/,
            float* distances,
            idx_t* labels,
            const IDSelector* sel,
            const FastScanDistancePostProcessing& context) const override;
};

/** SIMD result handler for RaBitQ FastScan that applies distance corrections
 * and maintains heaps directly during SIMD operations.
 *
 * This handler processes batches of 32 distance computations from SIMD kernels,
 * applies RaBitQ-specific adjustments (factors and normalizers), and
 * immediately updates result heaps without intermediate storage. This
 * eliminates the need for post-processing and provides significant memory and
 * performance benefits.
 *
 * Key optimizations:
 * - Direct heap integration (no intermediate result storage)
 * - Batch-level computation of normalizers and query factors
 * - Preserves exact mathematical equivalence to original RaBitQ distances
 * - Runtime boolean for multi-bit support
 *
 * @tparam C Comparator type (CMin/CMax) for heap operations
 * @tparam with_id_map Whether to use id mapping (similar to HeapHandler)
 */
template <class C, bool with_id_map = false>
struct RaBitQHeapHandler
        : simd_result_handlers::ResultHandlerCompare<C, with_id_map> {
    using RHC = simd_result_handlers::ResultHandlerCompare<C, with_id_map>;
    using RHC::normalizers;

    const IndexRaBitQFastScan* rabitq_index;
    float* heap_distances; // [nq * k]
    int64_t* heap_labels;  // [nq * k]
    const size_t nq, k;
    const FastScanDistancePostProcessing&
            context;         // Processing context with query offset
    const bool is_multi_bit; // Runtime flag for multi-bit mode

    // Use float-based comparator for heap operations
    using Cfloat = typename std::conditional<
            C::is_max,
            CMax<float, int64_t>,
            CMin<float, int64_t>>::type;

    RaBitQHeapHandler(
            const IndexRaBitQFastScan* index,
            size_t nq_val,
            size_t k_val,
            float* distances,
            int64_t* labels,
            const IDSelector* sel_in,
            const FastScanDistancePostProcessing& context,
            bool multi_bit);

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) override;

    void begin(const float* norms);

    void end();

   private:
    /// Compute full multi-bit distance for a candidate vector (multi-bit only)
    float compute_full_multibit_distance(size_t db_idx, size_t q) const;

    /// Compute lower bound using 1-bit distance and error bound (multi-bit
    /// only)
    float compute_lower_bound(float dist_1bit, size_t db_idx, size_t q) const;
};

} // namespace faiss
