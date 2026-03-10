/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <faiss/impl/CodePacker.h>
#include <faiss/impl/fast_scan/FastScanDistancePostProcessing.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>
#include <faiss/utils/Heap.h>

namespace faiss {

// Forward declaration — full definition needed only in implementation
struct IndexIVFRaBitQFastScan;

namespace simd_result_handlers {

/** SIMD result handler for IndexIVFRaBitQFastScan that applies
 * RaBitQ-specific distance corrections during batch processing.
 *
 * This handler processes batches of 32 distance computations from SIMD
 * kernels, applies RaBitQ distance formula adjustments (factors and
 * normalizers), and immediately updates result heaps. This eliminates the
 * need for post-processing and provides significant performance benefits.
 *
 * Key optimizations:
 * - Direct heap integration with no intermediate result storage
 * - Batch-level computation of normalizers and query factors
 * - Specialized handling for both centered and non-centered quantization
 * modes
 * - Efficient inner product metric corrections
 * - Uses runtime boolean for multi-bit mode
 *
 * @tparam C Comparator type (CMin/CMax) for heap operations
 * @tparam SL SIMD level for dynamic dispatch
 */
template <class C, SIMDLevel SL = SINGLE_SIMD_LEVEL_256>
struct IVFRaBitQHeapHandler : ResultHandlerCompare<C, true, SL> {
    using RHC = ResultHandlerCompare<C, true, SL>;
    using typename RHC::simd16uint16;

    const IndexIVFRaBitQFastScan* index;
    float* heap_distances; // [nq * k]
    int64_t* heap_labels;  // [nq * k]
    const size_t nq, k;
    size_t current_list_no = 0;
    const uint8_t* list_codes_ptr = nullptr; // raw block data for list
    std::vector<int>
            probe_indices; // probe index for each query in current batch
    const FastScanDistancePostProcessing*
            context;        // Processing context with query factors
    const bool is_multibit; // Whether to use multi-bit two-stage search
    size_t nup = 0;         // Number of heap updates

    // Cached block-layout constants (invariant for handler lifetime)
    const size_t storage_size;
    const size_t packed_block_size;
    const size_t full_block_size;
    std::unique_ptr<CodePacker> packer; // cached for unpack in hot path

    // Use float-based comparator for heap operations
    using Cfloat = typename std::conditional<
            C::is_max,
            CMax<float, int64_t>,
            CMin<float, int64_t>>::type;

    IVFRaBitQHeapHandler(
            const IndexIVFRaBitQFastScan* idx,
            size_t nq_val,
            size_t k_val,
            float* distances,
            int64_t* labels,
            const FastScanDistancePostProcessing* ctx = nullptr,
            bool multibit = false);

    void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) override;

    /// Override base class virtual method to receive context information
    void set_list_context(size_t list_no, const std::vector<int>& probe_map)
            override;

    void begin(const float* norms) override;

    void end() override;

    size_t num_updates() override {
        return nup;
    }

   private:
    /// Compute full multi-bit distance for a candidate vector (multi-bit
    /// only)
    /// @param db_idx Global database vector index
    /// @param local_q Batch-local query index (for probe_indices access)
    /// @param global_q Global query index (for storage indexing)
    /// @param local_offset Offset within the current inverted list
    float compute_full_multibit_distance(
            size_t /*db_idx*/,
            size_t local_q,
            size_t global_q,
            size_t local_offset) const;
};

} // namespace simd_result_handlers

} // namespace faiss
