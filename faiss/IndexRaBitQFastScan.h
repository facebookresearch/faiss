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
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/simdlib.h>

namespace faiss {

// Import shared utilities from RaBitQUtils
using rabitq_utils::FactorsData;
using rabitq_utils::QueryFactorsData;

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

    /// Extracted factors storage for batch processing
    /// Size: ntotal, stores factors separately from packed codes
    std::vector<FactorsData> factors_storage;

    /// Default number of bits to quantize a query with
    uint8_t qb = 8;

    // quantize the query with a zero-centered scalar quantizer.
    bool centered = false;

    IndexRaBitQFastScan();

    explicit IndexRaBitQFastScan(
            idx_t d,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    /// build from an existing IndexRaBitQ
    explicit IndexRaBitQFastScan(const IndexRaBitQ& orig, int bbs = 32);

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void compute_codes(uint8_t* codes, idx_t n, const float* x) const override;

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
    void* make_knn_handler(
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
            context; // Processing context with query offset

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
            const FastScanDistancePostProcessing& context);

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final;

    void begin(const float* norms);

    void end();
};

} // namespace faiss
